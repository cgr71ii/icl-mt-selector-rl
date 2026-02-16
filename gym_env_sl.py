
# 1. Generate offline dataset with a random policy:
# 1.1. state: LLM representation of the source sentence and the list of selected ICL examples. Store also the strings, not just the LLM representations
# 1.2. action: SONAR representation of the ICL example
# 1.3. reward: the specific metric (e.g., CHRF), but also the discretization of the reward to N values so we can generate a token for each reward value
# 1.3.1. The reward of a trajectory that has not finished (i.e., not all the ICL examples have been selected) is not 0, but the reward of the whole trajectory (i.e., return value, monte carlo return)
# 1.4. Although we can store the trajectories, we will store the transitions (s, a, r) to train the supervised encoder-only model
# 2. Train the supervised encoder-only model with the offline dataset for a certain number of epochs, with early stopping based on the validation set
# 3. Evaluate the best trained model with a greedy policy (i.e., select the ICL example with the highest reward value) on the validation and test sets

import os
import sys
import random
import logging
from datetime import datetime
import warnings

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

import utils
from gym_env_v1 import MTICLEnv
from gym_env_v1_eval import MTICLEvalEnv

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, is_vecenv_wrapped, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.transformer_layers import TransformerModel
import joblib
import pandas as pd

def evaluate_policy_custom(
    model,
    env,
    n_eval_episodes=10,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
    warn=True,
    device="cpu",
    reward_idx=None,
    logger=None,
):
    # Code adapter from stable-baselines3/stable_baselines3/common/evaluation.py

    assert reward_idx is not None
    assert isinstance(reward_idx, int) and reward_idx >= 0, f"reward_idx must be a non-negative integer, got {reward_idx}"
    assert logger is not None

    logger.info("Evaluating policy for %d episodes with reward_idx: %d", n_eval_episodes, reward_idx)

    is_monitor_wrapped = False

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    additional_embedding_idxs = [reward_idx] if reward_idx is not None else None

    assert n_envs == 1, n_envs # we assume only one environment for evaluation, to simplify the evaluation loop

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset() # TODO this returns the state of the environment, not just the representation from the LLM
    observations = torch.from_numpy(observations).to(device) if isinstance(observations, np.ndarray) else observations.to(device)
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions = model(observations, additional_embedding_idxs=additional_embedding_idxs)
        actions = actions.cpu().detach().numpy() if isinstance(actions, torch.Tensor) else actions
        new_observations, rewards, dones, infos = env.step(actions) # TODO this returns the state of the environment, not just the representation from the LLM
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations
        observations = torch.from_numpy(observations).to(device) if isinstance(observations, np.ndarray) else observations.to(device)

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

def make_env(rank, env_cls, env_args, env_kwargs, seed=None, seed_add_rank=False):
    def _init():
        sys.stderr.flush()
        env = env_cls(*env_args, **{"_seed": seed + (rank if seed_add_rank else 0) if seed is not None else rank, **env_kwargs})

        #env.reset(seed=seed + rank, options={"soft_reset_after_hard_reset": False})

        return env

    return _init

def make_batches(training_dataset, batch_size):
    assert batch_size > 0, f"Invalid batch size: {batch_size}"
    assert isinstance(training_dataset, list), f"Expected training_dataset to be a list, but got {type(training_dataset)}"

    for i in range(0, len(training_dataset), batch_size):
        yield training_dataset[i:i + batch_size]

def get_random_action(data_entry, env_training_dummy, max_icl_examples, data_icl_examples_training, num_labels_reward, logger, api_idx=None):
    src_sentence, reference = data_entry.split('\t')
    trajectory_states = []
    trajectory_actions = []
    trajectory_rewards = []
    icl_examples = []
    state = env_training_dummy.get_state_representation([src_sentence])[0]
    idx_icl_example = 0

    trajectory_states.append(state) # s_t

    while idx_icl_example < max_icl_examples:
        icl_example = random.choice(data_icl_examples_training)

        assert icl_example in env_training_dummy.str2representation, icl_example

        src_icl_example, trg_icl_example = icl_example.split('\t')

        if src_icl_example == src_sentence:
            logger.debug("Skipping ICL example with the same source sentence as the data entry: %s", icl_example)

            continue

        action = env_training_dummy.str2representation[icl_example] # a_t

        icl_examples.append(icl_example.split('\t'))
        trajectory_actions.append(action)

        translation = None if idx_icl_example < max_icl_examples - 1 else env_training_dummy.get_translations([src_sentence], icl_examples=[icl_examples], api_idx=api_idx)[0] # only get the translation for the last state, to save time
        reward = env_training_dummy.get_reward(src_sentence, reference, translation=translation, icl_examples=icl_examples) # TODO something might not work as intended as it assumes that environment is being used...

        trajectory_rewards.append(reward) # r_t

        state = env_training_dummy.get_state_representation([src_sentence], icl_examples=[icl_examples])[0]

        trajectory_states.append(state) # s_{t+1}

        idx_icl_example += 1

    assert len(icl_examples) == max_icl_examples
    assert len(trajectory_states) - 1 == len(trajectory_actions) == len(trajectory_rewards) == max_icl_examples

    trajectory_states.pop() # remove the last state, that we do not need for training the supervised model (we only need the transitions with (s, a, r), and the last state doesn't have an action)

    return_value = sum(trajectory_rewards)

    # TODO we assume r_0, r_1, ..., r_{t-1} = 0, and r_T = return_value

    for r in trajectory_rewards[:-1]:
        assert r == 0.0

    assert return_value >= 0 and return_value <= 100, f"Reward out of expected range [0, 100]: {return_value}"

    reward_label_position = min(int((return_value / 100) * num_labels_reward), num_labels_reward - 1) # discretize the reward into num_labels_reward values; lower is worse, higher is better

    logger.debug("Trajectory states: %s, actions: %s, return_value: %s, reward_label: %s", trajectory_states[-1].shape, trajectory_actions[-1].shape, return_value, reward_label_position)
    logger.debug("Trajectory icl_examples: %s", icl_examples)

    return {
        "src_sentence": src_sentence,
        "reference": reference,
        "trajectory_states": trajectory_states,
        "trajectory_actions": trajectory_actions,
        "trajectory_rewards": trajectory_rewards,
        "icl_examples": icl_examples,
        "return_value": return_value,
        "reward_label_position": reward_label_position,
    }

def main():
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.rl_experiments"), level=logging.DEBUG)

    logger.info("Provided args: %s", sys.argv)
    logger.info("In this script we assume that data (both translation sentences and ICL examples) and kNN elements are shared among all training environments and evaluation environment")

    # args
    src_lang = sys.argv[1].split(':')
    trg_lang = sys.argv[2].split(':')
    file_data = sys.argv[3].split(':')
    file_data_icl_examples = sys.argv[4].split(':')
    parsed_kwargs = utils.parse_args(sys.argv[5:])

    assert len(file_data) in (1, 3), f"Expected 1 or 3 file paths for training, dev, and test sets, but got {len(file_data)}"
    assert len(file_data_icl_examples) in (1, 3), f"Expected 1 or 3 file paths for ICL examples, but got {len(file_data_icl_examples)}"

    # parse args
    src_lang_training, src_lang_dev, src_lang_test = src_lang if len(src_lang) == 3 else (src_lang[0],) * 3
    trg_lang_training, trg_lang_dev, trg_lang_test = trg_lang if len(trg_lang) == 3 else (trg_lang[0],) * 3
    file_data_training, file_data_dev, file_data_test = file_data if len(file_data) == 3 else (file_data[0],) * 3
    file_data_icl_examples_training, file_data_icl_examples_dev, file_data_icl_examples_test = file_data_icl_examples if len(file_data_icl_examples) == 3 else (file_data_icl_examples[0],) * 3

    # read data
    data_to_be_translated_training, data_to_be_translated_dev, data_to_be_translated_test = [], [], []
    data_icl_examples_training, data_icl_examples_dev, data_icl_examples_test = [], [], []

    for _file_data, data_to_be_translated in ((file_data_training, data_to_be_translated_training), (file_data_dev, data_to_be_translated_dev), (file_data_test, data_to_be_translated_test)):
        with open(_file_data, "rt") as fd:
            for line in fd:
                data_to_be_translated.append(line.rstrip("\r\n"))

    for _file_data, data_icl_examples in ((file_data_icl_examples_training, data_icl_examples_training), (file_data_icl_examples_dev, data_icl_examples_dev), (file_data_icl_examples_test, data_icl_examples_test)):
        with open(_file_data, "rt") as fd:
            for line in fd:
                data_icl_examples.append(line.rstrip("\r\n"))

    # default values
    num_envs = int(max(1, parsed_kwargs.pop("num_envs", 8)))
    device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    max_icl_examples = max(int(parsed_kwargs.get("max_icl_examples", 5)), 1)
    dataset_rollouts_per_data_entry = max(1, int(parsed_kwargs.pop("dataset_rollouts_per_data_entry", 1)))
    num_labels_reward = max(int(parsed_kwargs.get("num_labels_reward", 5)), 2) # number of discrete reward labels (e.g., 5 means we discretize the reward into 5 values, and we can represent each value with a token)

    if "_seed" in parsed_kwargs or "seed" in parsed_kwargs:
        seed = parsed_kwargs.pop("_seed", None)

        if seed is None:
            seed = parsed_kwargs.pop("seed")

        seed = int(seed)
    else:
        seed = 42

    utils.set_random_seed(seed)

    randint_values = (1, 1000)
    env_seeds = [random.randint(*randint_values) for _ in range(num_envs)]

    for rank in range(1, num_envs):
        while env_seeds[rank] in env_seeds[:rank]:
            env_seeds[rank] = random.randint(*randint_values)

    logger.info("Seed: %s (env_seeds: %s)", seed, env_seeds)

    assert len(env_seeds) == len(set(env_seeds)) == num_envs

    # set defaults in case they are not provided
    max_data_entries = int(parsed_kwargs.get("max_data_entries", -1)) # load all data (default value)
    max_data_icl_examples_entries = int(parsed_kwargs.get("max_data_icl_examples_entries", -1)) # load all data (default value)
    #max_data_entries = 7 # TODO remove
    #max_data_icl_examples_entries = 8 # TODO remove
    state_representation = parsed_kwargs.get("state_representation", "representation_per_token_with_features")
    parsed_kwargs["device"] = device
    parsed_kwargs["max_icl_examples"] = max_icl_examples
    parsed_kwargs["max_data_entries"] = max_data_entries
    parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    parsed_kwargs["state_representation"] = state_representation
    parsed_kwargs["eval_strategy_training"] = parsed_kwargs.get("eval_strategy_training", "chrf2")
    parsed_kwargs["eval_strategy_eval"] = parsed_kwargs.get("eval_strategy_eval", "chrf2")
    parsed_kwargs["repeat_translation_candidates"] = parsed_kwargs.get("repeat_translation_candidates", True)
    parsed_kwargs["repeat_translation_candidates_times"] = parsed_kwargs.get("repeat_translation_candidates_times", 1)
    parsed_kwargs["knn_always_add_eos_action"] = parsed_kwargs.get("knn_always_add_eos_action", True)
    parsed_kwargs["enable_eos_action"] = parsed_kwargs.get("enable_eos_action", False)
    parsed_kwargs["state_window_length"] = int(parsed_kwargs.get("state_window_length", 1024)) + 3
    parsed_kwargs["action_representation"] = parsed_kwargs.get("action_representation", "src_embedding:SONAR")
    parsed_kwargs["model_hidden_size_action_src_sentence"] = parsed_kwargs.get("model_hidden_size_action_src_sentence", 1024)
    parsed_kwargs["actions_without_replacement"] = parsed_kwargs.get("actions_without_replacement", False) # allow/disallow selecting the same ICL example more than once in the same trajectory
    parsed_kwargs["observation_skip_first_n"] = int(parsed_kwargs.get("observation_skip_first_n", 3))
    data_to_be_translated_training = data_to_be_translated_training[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_dev = data_to_be_translated_dev[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_test = data_to_be_translated_test[:max_data_entries if max_data_entries > 0 else None]
    data_icl_examples_training = data_icl_examples_training[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_dev = data_icl_examples_dev[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_test = data_icl_examples_test[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]

    logger.info("For each data entry, we generate %d training rollouts: %d entries * %d rollouts * %d ICL examples = %d total training size", dataset_rollouts_per_data_entry, len(data_to_be_translated_training), dataset_rollouts_per_data_entry, max_icl_examples, len(data_to_be_translated_training) * dataset_rollouts_per_data_entry * max_icl_examples)

    assert parsed_kwargs["enable_eos_action"] is False, "This script assumes no EoS so far"

    assert parsed_kwargs["observation_skip_first_n"] == 3, "hot fix to address 'representation_per_token_with_features'" # TODO fix

    # Other kwargs
    parsed_kwargs_training = {}
    #parsed_kwargs_training["initial_time_sleep"] = num_envs * 2 # sleep to synchronize all environments
    parsed_kwargs_training_dummy = {}

    logger.info("parsed_kwargs: %s", parsed_kwargs)

    # Other values
    filename_time = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f"./rl_models_{filename_time}/"
    name_prefix = f"rl_{filename_time}"
    best_model_path = f"{name_prefix}_best_model.pt"
    save_model_path = os.path.join(save_path, best_model_path)
    monitor_filename = None # pickle serialization doesn't allow to have an opened file descriptor (EvalCallback)
    patience = 6 # early stopping patience (number of evals with no improvement)

    assert not os.path.exists(save_path), f"Save path already exists: {save_path}"

    os.makedirs(save_path, exist_ok=False)

    if patience < 0:
        logger.info("Early stopping disabled (patience < 0)")
    else:
        logger.info("Early stopping enabled (patience: %d evals with no improvement)", patience)

    logger.info("Save path: %s", save_path)

    # Environment
    env_class = MTICLEnv
    env_eval_dev_class = MTICLEvalEnv
    env_eval_test_class = MTICLEvalEnv
    vec_env_class = SubprocVecEnv
    vec_env_kwargs = {"start_method": "forkserver"} if vec_env_class is SubprocVecEnv else {}
    batch_size = max(1, int(parsed_kwargs.pop("sl_batch_size", 32)))
    learning_rate = 1e-5
    n_jobs = num_envs # TODO fix

    env_args = [src_lang_training, trg_lang_training, file_data_training, file_data_icl_examples_training]
    env_kwargs = {"gym_logger_level": gym.logger.DEBUG, **parsed_kwargs}
    env_training_dummy = env_class(*list(env_args), **dict({"custom_env_id": "training_dummy", **env_kwargs, **parsed_kwargs_training_dummy}))

    env_training_dummy._init_load_data_and_populate_knn_pool(options={}) # env_training_dummy.get_closest_neighbors_urls() is available

    #parsed_kwargs_training["initial_sample_list_actions"] = [(k, env_training_dummy.str2representation[k]) for k in env_training_dummy.str2representation_valid_actions_k] # initial random action sampling
    #env = vec_env_class([make_env(rank, env_class, list(env_args), dict({"custom_env_id": str(rank), **env_kwargs, **parsed_kwargs_training}), seed=env_seeds[rank]) for rank in range(num_envs)], **vec_env_kwargs)
    env_eval_dev = Monitor(env_eval_dev_class(src_lang_dev, trg_lang_dev, file_data_dev, file_data_icl_examples_dev, gym_logger_level=gym.logger.INFO, custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs), filename=monitor_filename, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={}) # env_eval_dev.get_closest_neighbors_urls() is available

    #retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=1, get_representations_instead_of_embeddings=False, observations=observations, debug=False)[0]

    # Generate trainig set rollouts with a random policy and store the transitions (s, a, r) for supervised learning
    # dataset_rollouts_per_data_entry
    training_dataset = []

    # TODO we assume reward in the range [0, 100]
    best_rewards = {data_entry: 0.0 for data_entry in data_to_be_translated_training} # best reward observed for each reward label, to monitor the training progress
    seen_data = set()

    logger.info("Generationg random rollouts for training dataset with %d parallel jobs", n_jobs)

    for rollout_idx in range(dataset_rollouts_per_data_entry):
#        for idx, data_entry in enumerate(data_to_be_translated_training):
#            src_sentence, reference = data_entry.split('\t')
#            action_data = get_random_action(data_entry, api_idx=idx)
#
#            # In case we need the trajectories:
#            #training_dataset.append({
#            #    "src_sentence": src_sentence,
#            #    "reference": reference,
#            #    "states": trajectory_states,
#            #    "actions": trajectory_actions,
#            #    "rewards": trajectory_rewards,
#            #    "icl_examples": icl_examples,
#            #    "return_value": return_value,
#            #    "reward_label": reward_label_position,
#            #})
#
#            trajectory_states = action_data["trajectory_states"]
#            trajectory_actions = action_data["trajectory_actions"]
#            trajectory_rewards = action_data["trajectory_rewards"]
#            icl_examples = action_data["icl_examples"]
#            return_value = action_data["return_value"]
#            reward_label_position = action_data["reward_label_position"]
#
#            # We store the transitions (s, a, r) for supervised learning
#            for s, a, r, icl_example in zip(trajectory_states, trajectory_actions, trajectory_rewards, icl_examples):
#                training_dataset.append({
#                    "src_sentence": src_sentence,
#                    "reference": reference,
#                    "state": s,
#                    "action": a,
#                    "reward": r,
#                    "icl_example": icl_example,
#                    "return_value": return_value,
#                    "reward_label": reward_label_position,
#                })
        result = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_random_action)(data_entry, env_training_dummy, max_icl_examples, data_icl_examples_training, num_labels_reward, logger, api_idx=idx) for idx, data_entry in enumerate(data_to_be_translated_training))

        assert len(result) == len(data_to_be_translated_training)

        for action_data, data_entry in zip(result, data_to_be_translated_training): # "The order of the outputs always matches the order the inputs have been submitted with"
            src_sentence, reference = data_entry.split('\t')
            _src_sentence = action_data["src_sentence"]
            _reference = action_data["reference"]
            trajectory_states = action_data["trajectory_states"]
            trajectory_actions = action_data["trajectory_actions"]
            trajectory_rewards = action_data["trajectory_rewards"]
            icl_examples = action_data["icl_examples"]
            return_value = action_data["return_value"]
            reward_label_position = action_data["reward_label_position"]
            icl_examples_str = '\t'.join(['\t'.join(icl_example) for icl_example in icl_examples])
            seen_key = hash(f"{src_sentence}\t{reference}\t{icl_examples_str}")

            assert src_sentence == _src_sentence, f"Expected src_sentence: {src_sentence}, but got: {_src_sentence}" # check order is correct
            assert reference == _reference, f"Expected reference: {reference}, but got: {_reference}"

            if seen_key in seen_data:
                continue

            seen_data.add(seen_key)

            # We store the transitions (s, a, r) for supervised learning
            for s, a, r, icl_example in zip(trajectory_states, trajectory_actions, trajectory_rewards, icl_examples):
                training_dataset.append({
                    "src_sentence": src_sentence,
                    "reference": reference,
                    "state": s,
                    "action": a,
                    "reward": r,
                    "icl_example": icl_example,
                    "return_value": return_value,
                    "reward_label": None, # we will assign the reward labels later, after we have generated all the trajectories and we can analyze the distribution of return values to assign the reward labels in a more informed way (e.g., using quantiles)
                })

            best_rewards[data_entry] = max(best_rewards[data_entry], action_data["return_value"])

        logger.info("Rollout %d: Best rewards observed (mean for %d entries): %s", rollout_idx + 1, len(best_rewards), sum(best_rewards.values()) / len(best_rewards) if len(best_rewards) > 0 else -1)

    #assert len(training_dataset) == len(data_to_be_translated_training) * dataset_rollouts_per_data_entry
    assert len(training_dataset) == len(data_to_be_translated_training) * dataset_rollouts_per_data_entry * max_icl_examples, f"Expected training dataset size: {len(data_to_be_translated_training) * dataset_rollouts_per_data_entry * max_icl_examples}, but got {len(training_dataset)}"

    logger.info("Generated training dataset with %d transitions (s, a, r)", len(training_dataset))
    logger.info("Best rewards observed (mean for %d entries): %s", len(best_rewards), sum(best_rewards.values()) / len(best_rewards) if len(best_rewards) > 0 else -1)

    #reward_intervals = {i: f"[{i * (100 / num_labels_reward)}, {(i + 1) * (100 / num_labels_reward)}]" for i in range(num_labels_reward)} # intervals for each reward label
    reward_intervals = pd.qcut([d["return_value"] for d in training_dataset], num_labels_reward, duplicates='drop')
    reward_intervals_codes = reward_intervals.codes.tolist()

    logger.info("Reward label description (total: %d):\n%s", len(reward_intervals), reward_intervals.describe())

    assert len(reward_intervals_codes) == len(training_dataset)

    for idx in range(len(training_dataset)):
        training_dataset[idx]["reward_label"] = reward_intervals_codes[idx]

    random.shuffle(training_dataset)

    action_dim = env_training_dummy.action_dim
    state_dim_per_token = env_training_dummy.state_dim_per_token
    state_window_length = env_training_dummy.state_window_length
    model_kwargs = {
        "d_model": 512,
        "nhead": 4,
        "dim_feedforward": 2048,
        "nlayers": 3,
        "projection_in": state_dim_per_token,
        "projection_out": action_dim,
        "max_seq_len": 8192,
        "embedding_dropout": 0.1,
        "dropout_p": 0.1,
        "projection_out_dropout_p": 0.1,
        "initial_layer_norm": False,
        "initial_layer_norm_first": False,
        "activation": "relu",
        "bias": True,
        "norm_first": True,
        "l2_norm": False,
        "skip_n_word_embeddings_from_observation": "0:0",
        "str_id": "none",
        "expected_seq_len": None,
        "last_layer_norm": False,
        "mean_pooling": False,
        "last_linear_layer": True,
        "additional_embeddings": num_labels_reward,
    }
    model = TransformerModel(**model_kwargs)
    model = model.train()
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad_(True)

    train_until_patience = True
    epochs = 1000
    training_steps_per_epoch = len(training_dataset) // batch_size + (0 if len(training_dataset) % batch_size == 0 else 1)
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to patience
    model_parameters = list(filter(lambda d: d.requires_grad, [p for k, p in model.named_parameters()]))
    optimizer_args_params = [{"params": model_parameters, "lr": learning_rate}]

    logger.info("Parameters with requires_grad=True: %d", len(model_parameters))

    optimizer, scheduler =\
        utils.get_lr_scheduler_and_optimizer_using_argparse_values("adamw", "inverse_sqrt", [0.9, 0.999, 1e-08, 0.01], ["400"], optimizer_args_params, learning_rate, training_steps, training_steps_per_epoch, logger)

    # training args
    current_patience = 0
    epoch = 0
    do_training = epoch < epochs or train_until_patience
    loss_function = nn.MSELoss(reduction="none")
    log_steps = 100 # TODO argument
    sum_epoch_loss = np.inf
    early_stopping_best_loss = np.inf
    early_stopping_best_result_dev = -np.inf # higher is better
    gradient_accumulation = 1
    debug = True
    early_stopping_metric_dev = early_stopping_best_result_dev

    while do_training:
        epoch_loss = []
        epoch_loss1 = []

        logger.info("Epoch #%d", epoch + 1)

        # Eval
        if epoch > 0:
            # dev_results = eval(model, # TODO
            mean_reward, std_reward = evaluate_policy_custom(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated_dev), render=False, device=device, reward_idx=num_labels_reward - 1, logger=logger)
            early_stopping_metric_dev = mean_reward

            logger.info("Dev eval: %s +- %s", mean_reward, std_reward)

        if len(epoch_loss) > 0 and sum_epoch_loss < early_stopping_best_loss:
            logger.info("Better loss result: %s -> %s", early_stopping_best_loss, sum_epoch_loss)

            early_stopping_best_loss = sum_epoch_loss

        if early_stopping_metric_dev > early_stopping_best_result_dev:
            logger.info("Patience better dev result: %s -> %s", early_stopping_best_result_dev, early_stopping_metric_dev)

            current_patience = 0
            early_stopping_best_result_dev = early_stopping_metric_dev

            if save_model_path:
                logger.info("Saving best model: %s", save_model_path)

                torch.save(model.state_dict(), save_model_path)
        elif epoch > 0 and patience > 0:
            current_patience += 1

            logger.info("Exhausting patience... %d / %d", current_patience, patience)

        if patience > 0 and current_patience >= patience:
            logger.info("Patience is over ...")

            do_training = False

            break # we need to force the break to avoid the training of the current epoch

        # Training loop
        model.zero_grad()
        final_loss = None
        final_loss1 = 0.0
        loss_elements1 = 0

        for batch_idx, batch in enumerate(make_batches(training_dataset, batch_size), 1):
            reward_labels = [item["reward_label"] for item in batch]
            states = [item["state"] for item in batch]
            actions = [item["action"] for item in batch]

            assert len(states) == len(reward_labels) == len(batch), f"{len(states)} vs {len(reward_labels)} vs {len(batch)}"

            states = np.array(states)
            states = torch.from_numpy(states).to(device) # input
            actions = np.array(actions)
            actions = torch.from_numpy(actions).to(device) # output

            assert len(states.shape) == 2, states.shape
            assert len(actions.shape) == 2, actions.shape

            model_output = model(states, additional_embedding_idxs=reward_labels) # logits

            assert len(model_output.shape) == 2, model_output.shape
            assert model_output.shape[0] == len(batch), model_output.shape
            assert model_output.shape[1] == action_dim, model_output.shape
            assert model_output.shape == actions.shape, f"{model_output.shape} vs {actions.shape}"

            _loss = loss_function(model_output, actions)
            loss_elements1 += _loss.numel()

            assert len(_loss.shape) == 2, _loss.shape

            _loss = _loss.sum(dim=1) # sum the loss for each element in the batch

            assert len(_loss.shape) == 1, _loss.shape
            assert _loss.shape == (len(batch),), f"{_loss.shape} vs {(len(batch),)}"

            final_loss1 += torch.sum(_loss).cpu().detach().item()

            assert len(_loss.shape) == 1, _loss.shape

            if final_loss is None:
                final_loss = torch.sum(_loss)
            else:
                final_loss += torch.sum(_loss)

            # loss
            if batch_idx % gradient_accumulation == 0 or batch_idx == training_steps_per_epoch:
                assert final_loss is not None

                loss = final_loss / loss_elements1
                loss1 = final_loss1 / (loss_elements1 if loss_elements1 > 0. else 1.)
                final_loss = None
                loss_elements1 = 0
                final_loss1 = 0.0

                epoch_loss.append(loss.cpu().detach().item())
                epoch_loss1.append(loss1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if debug and batch_idx % 50 == 0:
                    # Grad
                    _model_grad_sum = sum([p.grad.sum().item() for p in model.parameters() if p.grad is not None])
                    _model_projection_grad_sum = sum([p.grad.sum().item() for p in model.projection.parameters() if model.projection is not None and p.grad is not None])

                    logger.debug("Grad sum (model, projection): %s %s", _model_grad_sum, _model_projection_grad_sum)

                optimizer.step()
                scheduler.step()

                model.zero_grad()

            if (batch_idx % (log_steps * gradient_accumulation)) == 0:
                sum_partial_loss = sum(epoch_loss[-1 * log_steps:]) # no: -1 * log_steps * gradient_accumulation!
                sum_loss = sum(epoch_loss)

                logger.info("Batch #%d: %s (last %d steps: %s)", batch_idx, sum_loss, log_steps * gradient_accumulation, sum_partial_loss)

                sys.stdout.flush()

        assert batch_idx == training_steps_per_epoch, f"{batch_idx} vs {training_steps_per_epoch}"

        sum_epoch_loss = sum(epoch_loss)

        logger.info("Epoch loss: %s", sum_epoch_loss)

        assert str(sum_epoch_loss) != "nan", "Some values in the input data are NaN"

        sys.stdout.flush()

        epoch += 1
        do_training = epoch < epochs or train_until_patience

if __name__ == "__main__":
    main()
