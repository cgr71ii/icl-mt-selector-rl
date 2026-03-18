
# 1. Generate offline dataset with a random policy:
# 1.1. state: LLM representation of the source sentence and the list of selected ICL examples. Store also the strings, not just the LLM representations
# 1.2. action: SONAR representation of the ICL example
# 1.3. reward: the specific metric (e.g., CHRF), but also the discretization of the reward to N values so we can generate a token for each reward value
# 1.3.1. The reward of a trajectory (=rollout) that has not finished (i.e., not all the ICL examples have been selected) is not 0, but the reward of the whole trajectory (i.e., return value, monte carlo return)
# 1.4. Although we can store the trajectories, we will store the transitions (s, a, r) to train the supervised encoder-only model
# 2. Train the supervised encoder-only model with the offline dataset for a certain number of epochs, with early stopping based on the validation set
# 3. Evaluate the best trained model with a greedy policy (i.e., select the ICL example with the highest reward value) on the validation and test sets

import os
import sys
import random
import logging
from datetime import datetime
import warnings
import pickle

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
    max_icl_examples=None,
    l2_norm_model_output=False,
):
    # Code adapter from stable-baselines3/stable_baselines3/common/evaluation.py

    assert reward_idx is not None
    assert isinstance(reward_idx, int) and reward_idx >= 0, f"reward_idx must be a non-negative integer, got {reward_idx}"
    assert logger is not None
    assert max_icl_examples is not None

    logger.info("Evaluating policy for %d episodes with reward_idx: %d", n_eval_episodes, reward_idx)

    training = model.training
    is_monitor_wrapped = False
    env_skip = env.unwrapped.action_dim + env.unwrapped.state_dim_per_token + env.unwrapped.state_dim_per_token
    env_state_dim = env.unwrapped.state_dim

    model.eval()

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
    reward_embedding_idxs = [reward_idx] if reward_idx is not None else None

    assert reward_embedding_idxs is not None
    assert n_envs == 1, n_envs # we assume only one environment for evaluation, to simplify the evaluation loop

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    observations = torch.from_numpy(observations) if isinstance(observations, np.ndarray) else observations
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    idx = 0

    assert observations.shape[-1] == env_state_dim, f"Expected observation shape: (*, {env_state_dim}), but got: {observations.shape}"

    while (episode_counts < episode_count_targets).any():
        assert len(observations.shape) == 2, observations.shape
        #assert observations.shape[1] == 1024 + 1024 # TODO fix
        # Since the environment returns the state of the environment, not just the representation from the LLM, we need to fix it:
        #logger.error("debug: %s %s %s", observations.shape, env_skip, env_state_dim)
        _observations = torch.zeros_like(observations)
        _observations[:, :env_state_dim - env_skip] = observations[:, env_skip:] # TODO fix. Remove source sentence representation and reward info in 2 first tokens
        #_observations[:, 4:] = _observations[:, :-4].clone() # TODO remove
        #_observations[:, :4] = torch.ones_like(_observations[:, :4]) * (idx + 1) / 10 # TODO remove
        #_observations[:, 0] = 0.0 # TODO remove
        #_observations[:, 4:] = 0.0 # TODO remove
        #_observations[...] = 0.0 # TODO remove
        observations = _observations
        observations = observations.to(device)

        #logger.debug("debug: %s: %s: %s)", observations.shape, torch.sum((observations.reshape(1, -1, 4) == 0).all(dim=2)), observations.reshape(1, -1, 4))

        actions = model(observations, reward_embedding_idxs=reward_embedding_idxs, step_embedding_idxs=[idx % max_icl_examples])

        if l2_norm_model_output:
            actions = utils.l2_normalize(actions)

        actions = actions.cpu().detach().numpy() if isinstance(actions, torch.Tensor) else actions
        new_observations, rewards, dones, infos = env.step(actions)
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
        observations = torch.from_numpy(observations) if isinstance(observations, np.ndarray) else observations
        idx += 1

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if training:
        model.train()

    logger.info("Eval finished (training: %s)", training)

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

def make_batches(training_dataset, batch_size, sample=False, replacement=False, replacement_assert_mod_n=None):
    assert batch_size > 0, f"Invalid batch size: {batch_size}"
    assert isinstance(training_dataset, list), f"Expected training_dataset to be a list, but got {type(training_dataset)}"

    if not sample:
        for i in range(0, len(training_dataset), batch_size):
            yield training_dataset[i:i + batch_size]
    else:
        if replacement:
            if replacement_assert_mod_n is not None:
                assert isinstance(replacement_assert_mod_n, int), type(replacement_assert_mod_n)
                assert replacement_assert_mod_n > 0

            n = len(training_dataset)
            extra_n = 0
            len_l = len(range(0, n, batch_size))

            while (len_l % replacement_assert_mod_n) != 0:
                n += batch_size
                extra_n += 1
                new_len_l = len(range(0, n, batch_size))

                assert len_l + extra_n == new_len_l, f"Expected new_len_l to be equal to len_l + extra_n, but got: {len_l} + {extra_n} vs {new_len_l}"

            for _ in range(0, n, batch_size):
                yield random.choices(training_dataset, k=batch_size)
        else:
            indices = list(range(len(training_dataset)))

            random.shuffle(indices)

            for i in range(0, len(training_dataset), batch_size):
                batch_indices = indices[i:i + batch_size]

                yield [training_dataset[i] for i in batch_indices]

def generate_rollouts(joblib_idx, data_entry, env_training_dummy, max_icl_examples, data_icl_examples_training, num_labels_reward, logger,
                      api_idx=None, model=None, k=1, device="cpu", icl_examples_prepend=True, l2_norm_model_output=False):
    src_sentence, reference = data_entry.split('\t')
    rollout_states = []
    rollout_actions = []
    rollout_rewards = []
    icl_examples = []
    rollout_steps = []
    state = env_training_dummy.get_state_representation([src_sentence], api_idx=api_idx)[0]
    idx_icl_example = 0

    assert isinstance(k, int)
    assert k > 0, k

    #state[4:] = state[:-4] # TODO remove
    #state[:4] = np.ones_like(state[:4]) * (idx_icl_example + 1) / 10 # TODO remove
    #state[0] = 0.0 # TODO remove
    #state[4:] = 0.0 # TODO remove
    #state[...] = 0.0 # TODO remove
    rollout_states.append(state) # s_t

    while idx_icl_example < max_icl_examples:
        if model is not None:
            observation = np.array(rollout_states[-1])
            observation = torch.from_numpy(observation).unsqueeze(0).to(device)
            proto_action = model(observation, reward_embedding_idxs=[num_labels_reward - 1], step_embedding_idxs=[idx_icl_example]).cpu().detach().numpy()

            if l2_norm_model_output:
                proto_action = utils.l2_normalize(proto_action)

            _icl_examples = env_training_dummy.get_closest_neighbors_urls(proto_action, k=k, get_representations_instead_of_embeddings=True, debug=False)[0]

            assert len(_icl_examples) == 1, len(_icl_examples)
            assert len(_icl_examples[0]) == k, len(_icl_examples[0])

            _icl_examples = _icl_examples[0]
            remove_idx = None

            for idx, icl_example in enumerate(_icl_examples):
                src_icl_example, trg_icl_example = icl_example.split('\t')

                if src_icl_example == src_sentence:
                    assert remove_idx is None

                    remove_idx = idx

            if remove_idx is not None:
                del _icl_examples[remove_idx]

            if len(_icl_examples) == 0:
                logger.debug("kNN again to look for a different NN")

                assert k == 1, k

                _icl_examples = env_training_dummy.get_closest_neighbors_urls(proto_action, k=k + 1, get_representations_instead_of_embeddings=True, debug=False)[0]
                _icl_examples = _icl_examples[0]

                assert len(_icl_examples) == 2, len(_icl_examples)

                src_icl_example, trg_icl_example = _icl_examples[0].split('\t')

                assert src_icl_example == src_sentence

                del _icl_examples[0]

                src_icl_example, trg_icl_example = _icl_examples[1].split('\t')

                assert src_icl_example != src_sentence

            # The result action will be randomly sampled
        else:
            # Random sampling from the training set of ICL examples
            _icl_examples = data_icl_examples_training

        assert isinstance(_icl_examples, list), type(_icl_examples)
        assert len(_icl_examples) > 0
        assert isinstance(_icl_examples[0], str), type(_icl_examples[0])

        icl_example = random.choice(_icl_examples)

        assert icl_example in env_training_dummy.str2representation, icl_example

        src_icl_example, trg_icl_example = icl_example.split('\t')

        if src_icl_example == src_sentence:
            assert model is None

            logger.debug("Skipping ICL example with the same source sentence as the data entry: %s", icl_example)

            continue

        action = env_training_dummy.str2representation[icl_example] # a_t

        if icl_examples_prepend:
            icl_examples.insert(0, icl_example.split('\t'))
        else:
            icl_examples.append(icl_example.split('\t'))

        rollout_actions.append(action)

        translation = None if idx_icl_example < max_icl_examples - 1 else env_training_dummy.get_translations([src_sentence], icl_examples=[icl_examples], api_idx=api_idx)[0] # only get the translation for the last state, to save time
        reward = env_training_dummy.get_reward(src_sentence, reference, translation=translation) # TODO something might not work as intended as it assumes that environment is being used...

        rollout_rewards.append(reward) # r_t

        state = env_training_dummy.get_state_representation([src_sentence], icl_examples=[icl_examples], api_idx=api_idx)[0]

        #state[4:] = state[:-4] # TODO remove
        #state[:4] = np.ones_like(state[:4]) * (idx_icl_example + 2) / 10 # TODO remove
        #state[0] = 0.0 # TODO remove
        #state[4:] = 0.0 # TODO remove
        #state[...] = 0.0 # TODO remove
        rollout_states.append(state) # s_{t+1}
        rollout_steps.append(idx_icl_example)

        idx_icl_example += 1

    assert len(icl_examples) == max_icl_examples
    assert len(rollout_states) - 1 == len(rollout_actions) == len(rollout_rewards) == max_icl_examples == len(rollout_steps), f"Expected lengths: {max_icl_examples}, but got: {len(rollout_states) - 1}, {len(rollout_actions)}, {len(rollout_rewards)}, {len(rollout_steps)}"

    rollout_states.pop() # remove the last state, that we do not need for training the supervised model (we only need the transitions with (s, a, r), and the last state doesn't have an action)

    return_value = sum(rollout_rewards)

    # TODO we assume r_0, r_1, ..., r_{t-1} = 0, and r_T = return_value

    for r in rollout_rewards[:-1]:
        assert r == 0.0

    assert return_value >= 0 and return_value <= 100, f"Reward out of expected range [0, 100]: {return_value}"

    reward_label_position = min(int((return_value / 100) * num_labels_reward), num_labels_reward - 1) # discretize the reward into num_labels_reward values; lower is worse, higher is better

    if icl_examples_prepend:
        icl_examples = list(reversed(icl_examples)) # reverse to have the order of the ICL examples in the same order as they are selected

    logger.debug("rollout states: %s, actions: %s, return_value: %s, reward_label: %s, icl_examples: %s", rollout_states[-1].shape, rollout_actions[-1].shape, return_value, reward_label_position, icl_examples)

    return joblib_idx, {
        "src_sentence": src_sentence,
        "reference": reference,
        "rollout_states": rollout_states,
        "rollout_actions": rollout_actions,
        "rollout_rewards": rollout_rewards,
        "icl_examples": icl_examples,
        "return_value": return_value,
        "reward_label_position": reward_label_position,
        "rollout_steps": rollout_steps,
    }

def do_generate_rollouts(data_to_be_translated_training, dataset_rollouts_per_data_entry, env_training_dummy, max_icl_examples, data_icl_examples_training, num_labels_reward, logger, n_jobs, best_rewards,
                         model=None, k=1, device="cpu", icl_examples_prepend=True, initial_rollout_idx=0, l2_norm_model_output=False):
    # TODO we assume reward in the range [0, 100]
    training_dataset = []
    seen_data = set()
    training = model.training if model is not None else False
    rollout_unique_idx = initial_rollout_idx

    if model is not None:
        model.eval()

    logger.info("Generating random rollouts for training dataset with %d parallel jobs", n_jobs)

    for rollout_idx in range(dataset_rollouts_per_data_entry):
        # prefer="threads" is better in this case due to API calls and model usage that can raise exceptions when using multiprocessing, since these use cases release the GIL, and using threads allows to avoid some of these exceptions while still providing parallelism (not really need to use multiprocessing)
        with torch.no_grad():
            result = joblib.Parallel(n_jobs=n_jobs, timeout=999999, prefer="threads")(joblib.delayed(generate_rollouts)(idx, data_entry, env_training_dummy, max_icl_examples, data_icl_examples_training, num_labels_reward, logger, api_idx=idx, model=model, k=k, device=device, icl_examples_prepend=icl_examples_prepend, l2_norm_model_output=l2_norm_model_output) for idx, data_entry in enumerate(data_to_be_translated_training))

        assert len(result) == len(data_to_be_translated_training)

        for action_idx, (_action_data, data_entry) in enumerate(zip(result, data_to_be_translated_training)): # "The order of the outputs always matches the order the inputs have been submitted with"
            joblib_idx, action_data = _action_data # unpack the joblib_idx that was returned by generate_rollouts

            assert joblib_idx == action_idx, f"Expected joblib_idx: {action_idx}, but got: {joblib_idx}" # check order is correct

            src_sentence, reference = data_entry.split('\t')
            _src_sentence = action_data["src_sentence"]
            _reference = action_data["reference"]
            rollout_states = action_data["rollout_states"]
            rollout_actions = action_data["rollout_actions"]
            rollout_rewards = action_data["rollout_rewards"]
            icl_examples = action_data["icl_examples"]
            return_value = action_data["return_value"]
            rollout_steps = action_data["rollout_steps"]
            icl_examples_str = '\t'.join(['\t'.join(icl_example) for icl_example in icl_examples])
            seen_key = hash(f"{src_sentence}\t{reference}\t{icl_examples_str}")

            assert src_sentence == _src_sentence, f"Expected src_sentence: {src_sentence}, but got: {_src_sentence}" # check order is correct
            assert reference == _reference, f"Expected reference: {reference}, but got: {_reference}"

            if seen_key in seen_data:
                continue

            seen_data.add(seen_key)

            # We store the transitions (s, a, r) for supervised learning
            for s, a, r, icl_example, step in zip(rollout_states, rollout_actions, rollout_rewards, icl_examples, rollout_steps):
                training_dataset.append({
                    "src_sentence": src_sentence,
                    "reference": reference,
                    "state": s,
                    "action": a,
                    "reward": r,
                    "icl_example": icl_example,
                    "return_value": return_value,
                    "reward_label": None, # we will assign the reward labels later, after we have generated all the trajectories and we can analyze the distribution of return values to assign the reward labels in a more informed way (e.g., using quantiles)
                    "step": step,
                    "icl_examples_prepend": icl_examples_prepend,
                    "rollout_id": rollout_unique_idx,
                })

            best_rewards[data_entry] = max(best_rewards[data_entry], action_data["return_value"])
            rollout_unique_idx += 1

        logger.info("Iteration %d: Best rewards observed (mean for %d entries): %s", rollout_idx + 1, len(best_rewards), sum(best_rewards.values()) / len(best_rewards) if len(best_rewards) > 0 else -1)

    if model is not None:
        if training:
            model.train()

        logger.info("Model training: %s", training)

    return training_dataset

def update_training_dataset(training_dataset, best_rewards, num_labels_reward, epoch, logger):
    assert isinstance(epoch, int)
    assert epoch > 0, epoch

    # We remove duplicated steps, so it is not trivial to anticipate the expected length for the training set
    #assert len(training_dataset) == len(data_to_be_translated_training) * dataset_rollouts_per_data_entry * max_icl_examples * epoch, f"Expected training dataset size: {len(data_to_be_translated_training) * dataset_rollouts_per_data_entry * max_icl_examples}, but got {len(training_dataset)}"

    logger.info("Best rewards observed (mean for %d entries): %s", len(best_rewards), sum(best_rewards.values()) / len(best_rewards) if len(best_rewards) > 0 else -1)
    logger.info("Training dataset length: %d transitions (s, a, r)", len(training_dataset))

    reward_intervals = pd.qcut([d["return_value"] for d in training_dataset], num_labels_reward, duplicates='drop')
    reward_intervals_codes = reward_intervals.codes.tolist()
    max_reward_label_code = max(reward_intervals_codes)
    reward_intervals_codes_offset = 0

    if max_reward_label_code < num_labels_reward - 1:
        logger.warning("Max reward label code is less than num_labels_reward - 1: %d < %d", max_reward_label_code, num_labels_reward - 1)

        reward_intervals_codes_offset = num_labels_reward - 1 - max_reward_label_code

    logger.info("Reward label description (total: %d):\n%s", len(reward_intervals), reward_intervals.describe())

    assert len(reward_intervals_codes) == len(training_dataset)

    translation2reward_label2rollout_id_and_return = {}
    rollout_id2remove = set()
    categories_count = {label: 0 for label in range(num_labels_reward)}

    for idx in range(len(training_dataset)):
        _training_dataset = training_dataset[idx]

        if epoch == 1:
            assert _training_dataset["reward_label"] is None, _training_dataset["reward_label"]
        #else:
        #    assert isinstance(_training_dataset["reward_label"], int), type(_training_dataset["reward_label"]) # this may not be true if new data is added in each epoch

        _training_dataset["reward_label"] = reward_intervals_codes[idx] + reward_intervals_codes_offset

        assert isinstance(training_dataset[idx]["reward_label"], int), type(training_dataset[idx]["reward_label"])

        # Update translation2reward_label2rollout_id_and_return mapping
        translation = f"{_training_dataset['src_sentence']}\t{_training_dataset['reference']}"
        reward_label = _training_dataset["reward_label"]
        rollout_id = _training_dataset["rollout_id"]
        return_value = _training_dataset["return_value"]

        if translation not in translation2reward_label2rollout_id_and_return:
            translation2reward_label2rollout_id_and_return[translation] = {}

        if reward_label not in translation2reward_label2rollout_id_and_return[translation]:
            translation2reward_label2rollout_id_and_return[translation][reward_label] = (rollout_id, return_value)

        _rollout_id, _return_value = translation2reward_label2rollout_id_and_return[translation][reward_label]

        if rollout_id != _rollout_id:
            logger.debug("Found different rollout_id for the same translation and reward label: reward_label: %s, rollout_id: %s, return_value: %s vs rollout_id: %s, return_value: %s. Removing rollout with lower return value", reward_label, rollout_id, return_value, _rollout_id, _return_value)

            if return_value > _return_value:
                translation2reward_label2rollout_id_and_return[translation][reward_label] = (rollout_id, return_value)

                rollout_id2remove.add(_rollout_id)
            else:
                rollout_id2remove.add(rollout_id)

        categories_count[reward_label] += 1

        #logger.error("debug (label: %s): %s\n%s\n%s\n%s\n%s\n%s\n%s",
        #             _training_dataset["reward_label"],
        #             _training_dataset["src_sentence"],
        #             _training_dataset["icl_example"],
        #             torch.sum((torch.from_numpy(_training_dataset["state"]).reshape(1, -1, 4) == 0).all(dim=2)),
        #             torch.from_numpy(_training_dataset["state"]).reshape(-1, 4),
        #             torch.sum(torch.from_numpy(_training_dataset["state"]).reshape(-1, 4), dim=(0, 1)),
        #             _training_dataset["action"],
        #             torch.sum(torch.from_numpy(_training_dataset["action"]), dim=0)
        #            )

    logger.info("Categories count: %s", categories_count)

    for k, v in categories_count.items():
        if v == 0:
            logger.warning("Category %s has 0 samples", k)

    if len(rollout_id2remove) > 0:
        # Remove transitions from the training dataset that correspond to rollouts that have a different rollout with the same translation and reward label
        _training_dataset = [d for d in training_dataset if d["rollout_id"] not in rollout_id2remove]

        logger.info("Removing transitions from the training dataset that correspond to rollouts that have a different rollout with the same translation and reward label: %d -> %d training transitions (%s %% removed from the original training set)", len(training_dataset), len(_training_dataset), 100 * (len(training_dataset) - len(_training_dataset)) / len(training_dataset))

        training_dataset = _training_dataset

    translation2reward_label2rollout_id_and_return = {}
    new_categories_count = {label: 0 for label in range(num_labels_reward)}

    for idx in range(len(training_dataset)):
        _training_dataset = training_dataset[idx]

        assert isinstance(_training_dataset["reward_label"], int), type(_training_dataset["reward_label"])

        translation = f"{_training_dataset['src_sentence']}\t{_training_dataset['reference']}"
        reward_label = _training_dataset["reward_label"]
        rollout_id = _training_dataset["rollout_id"]
        return_value = _training_dataset["return_value"]

        if translation not in translation2reward_label2rollout_id_and_return:
            translation2reward_label2rollout_id_and_return[translation] = {}

        if reward_label not in translation2reward_label2rollout_id_and_return[translation]:
            translation2reward_label2rollout_id_and_return[translation][reward_label] = (rollout_id, return_value)

        new_categories_count[reward_label] += 1

        _rollout_id, _return_value = translation2reward_label2rollout_id_and_return[translation][reward_label]

        assert rollout_id == _rollout_id

    random.shuffle(training_dataset)

    logger.info("New categories count: %s", new_categories_count)

    for k, v in new_categories_count.items():
        if v == 0:
            logger.warning("New category %s has 0 samples", k)

    return training_dataset

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
    pre_k = parsed_kwargs.pop("k", "2")
    additional_parsed_kwargs_dev = {}

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
    #max_data_entries = 2 # TODO remove
    #max_data_entries = 20 # TODO remove
    #max_data_icl_examples_entries = 8 # TODO remove
    max_data_entries_dev = max_data_entries
    #max_data_entries_dev = 20 # TODO remove
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
    parsed_kwargs["knn_distance_ip"] = parsed_kwargs.get("knn_distance_ip", True)
    parsed_kwargs["current_icl_examples_prepend"] = parsed_kwargs.get("current_icl_examples_prepend", True)
    data_to_be_translated_training = data_to_be_translated_training[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_dev = data_to_be_translated_dev[:max_data_entries_dev if max_data_entries_dev > 0 else None]
    data_to_be_translated_test = data_to_be_translated_test[:max_data_entries if max_data_entries > 0 else None]
    data_icl_examples_training = data_icl_examples_training[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_dev = data_icl_examples_dev[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_test = data_icl_examples_test[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    pre_k_is_float = '.' in pre_k
    k_training = min(max(1, int(float(pre_k) * len(data_icl_examples_training)) if pre_k_is_float else int(pre_k)), len(data_icl_examples_training))
    knn_distance_ip = parsed_kwargs["knn_distance_ip"]
    current_icl_examples_prepend = parsed_kwargs["current_icl_examples_prepend"]
    disable_l2_norm_in_model = parsed_kwargs.pop("disable_l2_norm_in_model", False)

    if knn_distance_ip:
        logger.warning("Using inner product as distance for kNN: using l2-normalization in the model and cosine similarity for kNN. The MSE loss will be computed on the l2-normalized representations")

        if disable_l2_norm_in_model:
            logger.info("Disabling L2 normalization in the model, but using L2 normalization during inference")

    logger.info("For each data entry, we generate %d training rollouts: %d entries * %d rollouts * %d ICL examples = %d total training size", dataset_rollouts_per_data_entry, len(data_to_be_translated_training), dataset_rollouts_per_data_entry, max_icl_examples, len(data_to_be_translated_training) * dataset_rollouts_per_data_entry * max_icl_examples)
    logger.info("k=%d", k_training)

    assert parsed_kwargs["enable_eos_action"] is False, "This script assumes no EoS so far"
    assert state_representation == "representation_per_token_with_features", f"This script assumes state_representation: representation_per_token_with_features, but got: {state_representation}"

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
    training_dataset_path = parsed_kwargs.pop("training_dataset_path", None)
    #patience = 6 # early stopping patience (number of evals with no improvement)
    #patience = 100 # TODO remove
    #patience = -1 # disable early stopping # TODO remove?
    #patience = 10
    patience = 20
    #patience = 200 # TODO remove
    #eval_freq_epochs = 1
    #eval_freq_epochs = 5  # TODO remove
    eval_freq_epochs = 20
    #generate_new_samples_for_training_set_every_epochs = 1
    #generate_new_samples_for_training_set_every_epochs = 0 # TODO remove
    generate_new_samples_for_training_set_every_epochs = max(eval_freq_epochs // 2, 1)
    #initial_epochs_without_eval_nor_generation = 0
    initial_epochs_without_eval_nor_generation = 50 # ignored if min_loss_start_training is enabled
    #initial_epochs_without_eval_nor_generation = 1000 # TODO remove
    min_loss_start_training = -1.0 # negative is disabled
    min_loss_start_training = 0.7 if knn_distance_ip else 0.001
    #min_loss_start_training = 0.01
    min_loss_start_training = float(min_loss_start_training)

    if min_loss_start_training >= 0:
        logger.info("Minimum loss threshold for starting training: %f", min_loss_start_training)

        if initial_epochs_without_eval_nor_generation > 0:
            logger.warning("Minimum loss threshold for starting training is enabled, but initial_epochs_without_eval_nor_generation > 0: initial_epochs_without_eval_nor_generation = 0")

            initial_epochs_without_eval_nor_generation = 0

    assert eval_freq_epochs > 0
    assert generate_new_samples_for_training_set_every_epochs >= 0
    assert initial_epochs_without_eval_nor_generation >= 0
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
    #batch_size = max(1, int(parsed_kwargs.pop("sl_batch_size", 32)))
    batch_size = max(1, int(parsed_kwargs.pop("sl_batch_size", 512)))
    load_model_path = parsed_kwargs.pop("sl_load_model_path", None)
    #learning_rate = 1e-5
    learning_rate = 1e-4
    #learning_rate = 1e-3
    n_jobs = num_envs # TODO fix
    epochs = 200

    logger.info("Batch size: %d, learning_rate: %s", batch_size, learning_rate)

    ## create environments
    env_args = [src_lang_training, trg_lang_training, file_data_training, file_data_icl_examples_training]
    env_kwargs = {"gym_logger_level": gym.logger.DEBUG, **parsed_kwargs}
    env_training_dummy = env_class(*list(env_args), **dict({"custom_env_id": "training_dummy", **env_kwargs, **parsed_kwargs_training_dummy}))

    env_training_dummy._init_load_data_and_populate_knn_pool(options={}) # env_training_dummy.get_closest_neighbors_urls() is available

    additional_parsed_kwargs_dev["max_data_entries"] = max_data_entries_dev
    parsed_kwargs.pop("max_data_entries")

    logger.info("additional_parsed_kwargs_dev: %s", additional_parsed_kwargs_dev)

    #parsed_kwargs_training["initial_sample_list_actions"] = [(k, env_training_dummy.str2representation[k]) for k in env_training_dummy.str2representation_valid_actions_k] # initial random action sampling
    #env = vec_env_class([make_env(rank, env_class, list(env_args), dict({"custom_env_id": str(rank), **env_kwargs, **parsed_kwargs_training}), seed=env_seeds[rank]) for rank in range(num_envs)], **vec_env_kwargs)
    env_eval_dev = Monitor(env_eval_dev_class(src_lang_dev, trg_lang_dev, file_data_dev, file_data_icl_examples_dev, gym_logger_level=gym.logger.INFO, custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs, **additional_parsed_kwargs_dev), filename=monitor_filename, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={}) # env_eval_dev.get_closest_neighbors_urls() is available

    #retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=1, get_representations_instead_of_embeddings=False, observations=observations, debug=False)[0]

    action_dim = env_training_dummy.action_dim
    state_dim_per_token = env_training_dummy.state_dim_per_token
    state_window_length = env_training_dummy.state_window_length
    use_first_n_tokens = -1 # -1 means use all tokens

    if use_first_n_tokens > 0:
        logger.info("Using only the first %d tokens of the state representation as input to the model", use_first_n_tokens)

    model_kwargs = {
        #"d_model": 512,
        #"d_model": 128,
        "d_model": 256,
        "nhead": 4,
        #"dim_feedforward": 2048,
        #"dim_feedforward": 512,
        "dim_feedforward": 1024,
        #"nlayers": 3,
        "nlayers": 6,
        "projection_in": state_dim_per_token,
        "projection_out": action_dim,
        "max_seq_len": 8192,
        "embedding_dropout": 0.1,
        "dropout_p": 0.1,
        "projection_out_dropout_p": 0.1,
        #"initial_layer_norm": False,
        "initial_layer_norm": True, # needed for stable training
        #"initial_layer_norm_first": False,
        "initial_layer_norm_first": True,
        "initial_layer_norm_first_additional_layer_norm_after_projection": True, # needed to have std close to 1.0
        #"activation": "relu",
        "activation": "gelu",
        "bias": True,
        "norm_first": True,
        #"norm_first": False,
        #"l2_norm": False,
        "l2_norm": True if knn_distance_ip and not disable_l2_norm_in_model else False,
        "skip_n_word_embeddings_from_observation": "0:0",
        "str_id": "none",
        "expected_seq_len": None,
        "last_layer_norm": False,
        #"mean_pooling": False,
        "mean_pooling": True,
        "last_linear_layer": True,
        "reward_embeddings": num_labels_reward,
        "remove_first_column_of_zeros": True,
        "step_embeddings": max_icl_examples, # we can also use learnable step embeddings to represent the position in the trajectory
        "use_first_n_tokens": use_first_n_tokens,
    }
    model = TransformerModel(**model_kwargs)

    if load_model_path is not None and os.path.exists(load_model_path):
        logger.info("Loading model from: %s", load_model_path)

        model_weights = torch.load(load_model_path)

        model.load_state_dict(model_weights)

    model = model.to(device)

    model.train()

    for p in model.parameters():
        p.requires_grad_(True)

    train_until_patience = True if patience >= 0 else False
    model_parameters = list(filter(lambda d: d.requires_grad, [p for k, p in model.named_parameters()]))
    optimizer_args_params = [{"params": model_parameters, "lr": learning_rate}]

    logger.info("Parameters with requires_grad=True: %d", len(model_parameters))

    # Generate trainig set rollouts with a random policy and store the transitions (s, a, r) for supervised learning
    # dataset_rollouts_per_data_entry
    training_dataset = []
    best_rewards = {data_entry: 0.0 for data_entry in data_to_be_translated_training} # best reward observed for each reward label, to monitor the training progress
    load_initial_training_dataset = False

    if training_dataset_path is not None and os.path.exists(training_dataset_path):
        load_initial_training_dataset = True

    #load_initial_training_dataset = False # TODO remove?

    # BEGIN generate training set
    if not load_initial_training_dataset:
        training_dataset_path_store = os.path.join(save_path, f"training_dataset_initial.pkl")

        assert not os.path.exists(training_dataset_path_store), f"Training dataset path already exists: {training_dataset_path_store}"

        _training_dataset = do_generate_rollouts(
            data_to_be_translated_training,
            dataset_rollouts_per_data_entry,
            env_training_dummy,
            max_icl_examples,
            data_icl_examples_training,
            num_labels_reward,
            logger,
            n_jobs,
            best_rewards,
            model=None,
            #model=model, # TODO remove. debug
            k=k_training,
            device=device,
            icl_examples_prepend=current_icl_examples_prepend,
            initial_rollout_idx=max([t["rollout_id"] for t in training_dataset]) + 1 if len(training_dataset) > 0 else 0,
            l2_norm_model_output=knn_distance_ip and disable_l2_norm_in_model,
        )

        training_dataset.extend(_training_dataset)

        logger.info("Storing training dataset (length: %d): %s", len(training_dataset), training_dataset_path_store)

        with open(training_dataset_path_store, "wb") as fd:
            pickle.dump(training_dataset, fd)
    else:
        logger.info("Loading training dataset: %s", training_dataset_path)

        with open(training_dataset_path, "rb") as fd:
            training_dataset = pickle.load(fd)

        trajectory = {
            "rollout_states": [],
            "rollout_actions": [],
            "rollout_rewards": [],
            "icl_examples": [],
        }

        assert len(training_dataset) % max_icl_examples == 0, f"Expected training dataset size to be a multiple of max_icl_examples: {max_icl_examples}, but got: {len(training_dataset)}"
        assert max_data_icl_examples_entries < 0, f"Expected max_data_icl_examples_entries to be negative (no limit) when loading the training dataset, but got: {max_data_icl_examples_entries}"

        icl_examples_prepend = None
        seen_data_entries = set()
        remove_idxs = set()

        for action_idx, action_data in enumerate(training_dataset):
            src_sentence = action_data["src_sentence"]
            reference = action_data["reference"]
            data_entry = f"{src_sentence}\t{reference}"
            step = action_data["step"]

            if max_data_entries > 0 and len(seen_data_entries) >= max_data_entries and data_entry not in seen_data_entries:
                remove_idxs.add(action_idx)
                continue

            seen_data_entries.add(data_entry)

            if icl_examples_prepend is None:
                if "icl_examples_prepend" in action_data:
                    icl_examples_prepend = action_data["icl_examples_prepend"]
                else:
                    icl_examples_prepend = current_icl_examples_prepend

            if "icl_examples_prepend" in action_data:
                assert action_data["icl_examples_prepend"] == icl_examples_prepend, f"Expected icl_examples_prepend to be the same for all transitions in the training dataset, but got both True and False"
            else:
                action_data["icl_examples_prepend"] = icl_examples_prepend

            if "rollout_id" not in action_data:
                action_data["rollout_id"] = action_idx // max_icl_examples

            assert icl_examples_prepend == current_icl_examples_prepend, f"Expected icl_examples_prepend in the training dataset to be the same as the current_icl_examples_prepend used for generating the rollouts, but got: {icl_examples_prepend} vs {current_icl_examples_prepend}"

            trajectory["rollout_states"].append(action_data["state"])
            trajectory["rollout_actions"].append(action_data["action"])
            trajectory["rollout_rewards"].append(action_data["reward"])
            trajectory["icl_examples"].append(action_data["icl_example"])

            assert action_idx % max_icl_examples == step, f"Expected step: {action_idx % max_icl_examples}, but got: {step}"

            if step + 1 == max_icl_examples:
                logger.debug("rollout states: %s, actions: %s, return_value: %s, reward_label: %s, icl_examples: %s", trajectory["rollout_states"][-1].shape, trajectory["rollout_actions"][-1].shape, action_data["return_value"], action_data["reward_label"], trajectory["icl_examples"][-1])

                best_rewards[data_entry] = max(best_rewards[data_entry], action_data["return_value"])
                trajectory = {
                    "rollout_states": [],
                    "rollout_actions": [],
                    "rollout_rewards": [],
                    "icl_examples": [],
                }

        if len(remove_idxs) > 0:
            training_dataset = [d for idx, d in enumerate(training_dataset) if idx not in remove_idxs]

        assert len(trajectory["rollout_states"]) == 0, f"Expected trajectory to be empty after processing the training dataset, but got: {len(trajectory['rollout_states'])}"

        logger.info("Iteration -: Best rewards observed (mean for %d entries): %s", len(best_rewards), sum(best_rewards.values()) / len(best_rewards) if len(best_rewards) > 0 else -1)

    #training_dataset *= 20 # TODO remove. debug

    training_dataset = update_training_dataset(
        training_dataset,
        best_rewards,
        num_labels_reward,
        1,
        logger
    )

    training_steps_per_epoch = len(training_dataset) // batch_size + (0 if len(training_dataset) % batch_size == 0 else 1)
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to patience
    # END generate training set

    gradient_accumulation = 1
    #gradient_accumulation = 64

    logger.info("Gradient accumulation steps: max(min(gradient_accumulation=%d, ((len(training_dataset)=%d + batch_size=%d - 1) // batch_size)=%d), 1) = %d",
                gradient_accumulation, len(training_dataset), batch_size, (len(training_dataset) + batch_size - 1) // batch_size,
                max(min(gradient_accumulation, (len(training_dataset) + batch_size - 1) // batch_size), 1))

    gradient_accumulation = max(min(gradient_accumulation, (len(training_dataset) + batch_size - 1) // batch_size), 1) # + batch_size - 1 to round up (-1 to avoid counting an extra step when len(training_dataset) is divisible by batch_size)
    training_steps_per_epoch_lr_scheduler = len(training_dataset) // (batch_size * gradient_accumulation) + (0 if len(training_dataset) % (batch_size * gradient_accumulation) == 0 else 1)
    training_steps_lr_scheduler = training_steps_per_epoch_lr_scheduler * epochs

    #for t in training_dataset: # TODO remove
    #    logger.debug("debug: %s: %s\n%s", t["state"].shape, torch.sum((torch.from_numpy(t["state"]).reshape(1, -1, 4) == 0).all(dim=2)), torch.from_numpy(t["state"]).reshape(1, -1, 4))

    #warmup_steps = 1000 if patience >= 0 else "10%"
    warmup_steps = "10%"
    #lr_scheduler_str = "inverse_sqrt" if patience >= 0 else "linear"
    lr_scheduler_str = "linear"
    lr_scheduler_args = [str(warmup_steps)]
    lr_scheduler_restart_every_steps = -1 # negative value means no restart
    lr_scheduler_restart_every_steps = training_steps_lr_scheduler
    optimizer, scheduler =\
        utils.get_lr_scheduler_and_optimizer_using_argparse_values("adamw", lr_scheduler_str, [0.9, 0.999, 1e-08, 0.01], lr_scheduler_args, optimizer_args_params, learning_rate, training_steps_lr_scheduler, training_steps_per_epoch_lr_scheduler, logger)

    if generate_new_samples_for_training_set_every_epochs > 0:
        logger.warning("LR scheduler will be desynced with the training steps due to the growth of the training set (training_steps_per_epoch_lr_scheduler=%d) if generate_new_samples_for_training_set_every_epochs=%d > 0. It will not be a problem as long as the LR scheduler is restarted (lr_scheduler_restart_every_steps > 0): %s", training_steps_per_epoch_lr_scheduler, generate_new_samples_for_training_set_every_epochs, lr_scheduler_restart_every_steps)

    if train_until_patience:
        logger.info("Training until patience is exhausted, with patience: %d", patience)
    else:
        logger.info("Training for a fixed number of epochs: %d", epochs)

    if lr_scheduler_restart_every_steps > 0:
        logger.info("LR scheduler will restart every %d steps (when restarted, warmup is disabled)", lr_scheduler_restart_every_steps)

    # training args
    current_patience = 0
    epoch = 0
    do_training = epoch < epochs or train_until_patience
    #loss_function = nn.MSELoss(reduction="none")
    #loss_function = nn.MSELoss(reduction="mean")
    loss_function = nn.CosineEmbeddingLoss(reduction="mean") if knn_distance_ip else nn.MSELoss(reduction="mean")
    log_steps = 100 # TODO argument
    sum_epoch_loss = np.inf
    mean_epoch_loss = np.inf
    early_stopping_best_loss = np.inf
    early_stopping_best_result_dev = -np.inf # higher is better
    debug = True
    early_stopping_metric_dev = early_stopping_best_result_dev
    force_eval_msg = False
    times_optimizer_step_global = 0

    logger.info("Loss function: %s", loss_function)

    while do_training:
        epoch_loss = []
        epoch_cos_similarity = []
        epoch_cos_similarity_per_label = {label: [] for label in range(num_labels_reward)}
        times_optimizer_step = 0
        loss_accumulated_steps = 0

        # Eval and generate new samples
        epoch_number_fix_eval = epoch - initial_epochs_without_eval_nor_generation
        force_eval = os.path.exists("./bypass_eval")

        if force_eval_msg and not force_eval:
            logger.warning("Force eval disabled due to the absence of the ./bypass_eval file")

            force_eval_msg = False

        if not force_eval_msg and force_eval:
            logger.warning("Force eval enabled due to the presence of the ./bypass_eval file")

            force_eval_msg = True

        if force_eval or (min_loss_start_training < 0.0 and epoch_number_fix_eval > 0) or (min_loss_start_training >= 0.0 and early_stopping_best_loss <= min_loss_start_training):
            if eval_freq_epochs > 0 and epoch_number_fix_eval % eval_freq_epochs == 0:
                # Eval dev set

                with torch.no_grad():
                    mean_reward, std_reward = evaluate_policy_custom(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated_dev), render=False, device=device,
                                                                     reward_idx=num_labels_reward - 1, logger=logger, max_icl_examples=max_icl_examples,
                                                                     l2_norm_model_output=knn_distance_ip and disable_l2_norm_in_model)

                early_stopping_metric_dev = mean_reward

                logger.info("Dev eval (epoch: %d): %s +- %s", epoch, mean_reward, std_reward)

                if early_stopping_metric_dev > early_stopping_best_result_dev:
                    logger.info("Patience better dev result: %s -> %s", early_stopping_best_result_dev, early_stopping_metric_dev)

                    current_patience = 0
                    early_stopping_best_result_dev = early_stopping_metric_dev

                    if save_model_path:
                        logger.info("Saving best model: %s", save_model_path)

                        torch.save(model.state_dict(), save_model_path)
                elif patience > 0:
                    current_patience += 1

                    logger.info("Exhausting patience... %d / %d", current_patience, patience)

                if patience > 0 and current_patience >= patience:
                    logger.info("Patience is over ...")

                    do_training = False

                    break # we need to force the break to avoid the training of the current epoch

            if generate_new_samples_for_training_set_every_epochs > 0 and epoch_number_fix_eval % generate_new_samples_for_training_set_every_epochs == 0:
                # Generate new samples for the training set using the model trained so far, to have more informative samples in the training set as the training progresses

                logger.info("Generating new samples for the training set with the current model (epoch: %d). Recomputing the label rewards", epoch)
                # BEGIN generate training set
                _training_dataset = do_generate_rollouts(
                    data_to_be_translated_training,
                    1, # we generate 1 rollout per data entry at each epoch after the random rollouts generated at the beginning of the training
                    env_training_dummy,
                    max_icl_examples,
                    data_icl_examples_training,
                    num_labels_reward,
                    logger,
                    n_jobs,
                    best_rewards,
                    model=model,
                    k=k_training,
                    device=device,
                    icl_examples_prepend=current_icl_examples_prepend,
                    initial_rollout_idx=max([t["rollout_id"] for t in training_dataset]) + 1 if len(training_dataset) > 0 else 0,
                    l2_norm_model_output=knn_distance_ip and disable_l2_norm_in_model,
                )

                training_dataset.extend(_training_dataset)

                training_dataset = update_training_dataset(
                    training_dataset,
                    best_rewards,
                    num_labels_reward,
                    epoch + 1,
                    logger
                )

                training_steps_per_epoch = len(training_dataset) // batch_size + (0 if len(training_dataset) % batch_size == 0 else 1)
                training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to patience
                # END generate training set

        logger.info("Epoch #%d", epoch + 1)

        # Training loop
        model.zero_grad()

        for batch_idx, batch in enumerate(make_batches(training_dataset, batch_size, sample=True, replacement=True, replacement_assert_mod_n=gradient_accumulation), 1):
            icl_examples = [item["icl_example"] for item in batch]
            reward_labels = [item["reward_label"] for item in batch]
            states = [item["state"] for item in batch]
            actions = [item["action"] for item in batch]
            steps = [item["step"] for item in batch]

            #for idx, icl_example in enumerate(icl_examples):
            #    if "visit involves flying into Orlando International" in icl_example[0]:
            #        logger.error("debug (%s): %s\n%s\n%s", reward_labels[idx], icl_example,
            #                     torch.from_numpy(states[idx]).reshape(-1, 4), actions[idx])

            assert len(states) == len(reward_labels) == len(steps) == len(batch), f"{len(states)} vs {len(reward_labels)} vs {len(steps)} vs {len(batch)}"

            states = np.array(states)
            states = torch.from_numpy(states).to(device) # input
            actions = np.array(actions)
            actions = torch.from_numpy(actions).to(device) # output

            assert len(states.shape) == 2, states.shape
            assert len(actions.shape) == 2, actions.shape

            # TODO remove
#            model.eval()
#            with torch.no_grad():
#                model_output = model(states, reward_embedding_idxs=reward_labels, step_embedding_idxs=steps).cpu().detach().numpy() # logits
#                _icl_examples = env_training_dummy.get_closest_neighbors_urls(model_output, k=1, get_representations_instead_of_embeddings=True, debug=False)[0]
#
#                for reward_label, icl_example, actual_icl_example, state, action in zip(reward_labels, _icl_examples, icl_examples, states, actions):
#                    logger.error("debug (label: %s): %s | %s\n%s\n%s", reward_label, icl_example, actual_icl_example, state.reshape(-1, 4), action)
#            model.train()
            # TODO remove

            model_output = model(states, reward_embedding_idxs=reward_labels, step_embedding_idxs=steps) # logits

            assert len(model_output.shape) == 2, model_output.shape
            assert model_output.shape[0] == len(batch), model_output.shape
            assert model_output.shape[1] == action_dim, model_output.shape
            assert model_output.shape == actions.shape, f"{model_output.shape} vs {actions.shape}"

            if knn_distance_ip:
                _loss = loss_function(model_output, actions, torch.ones(actions.shape[0]).to(device))
            else:
                _loss = loss_function(model_output, actions)

            loss = _loss / gradient_accumulation
            loss_accumulated_steps += 1

            loss.backward()

            epoch_loss.append(loss.cpu().detach().item() * gradient_accumulation) # we multiply by gradient_accumulation to log the actual loss value before the division for gradient accumulation

            with torch.no_grad():
                cos_similarity = torch.nn.functional.cosine_similarity(model_output, actions, dim=1).cpu().detach().tolist()
                epoch_cos_similarity.extend(cos_similarity)

                assert len(cos_similarity) == len(reward_labels)

                for label, cos_similarity_value in zip(reward_labels, cos_similarity):
                    assert label in epoch_cos_similarity_per_label, label

                    epoch_cos_similarity_per_label[label].append(cos_similarity_value)

            # loss
            if batch_idx % gradient_accumulation == 0 or batch_idx == training_steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                times_optimizer_step += 1
                times_optimizer_step_global += 1
                show_statistics = debug and batch_idx % 100 == 0 or batch_idx == training_steps_per_epoch

                if show_statistics:
                    # Grad
                    _model_grad_sum = sum([p.grad.sum().item() for p in model.parameters() if p.grad is not None])
                    _model_projection_grad_sum = sum([p.grad.sum().item() for p in model.projection.parameters() if model.projection is not None and p.grad is not None])

                    logger.debug("Grad sum (model, projection): %s %s", _model_grad_sum, _model_projection_grad_sum)
                    logger.debug("Optimizer steps: %s (* gradient_accumulation = * %s = %s); Accumulated loss steps: %d", times_optimizer_step, gradient_accumulation, times_optimizer_step * gradient_accumulation, loss_accumulated_steps)
                    logger.debug("Optimizer steps (global): %s (* gradient_accumulation = * %s = %s)", times_optimizer_step_global, gradient_accumulation, times_optimizer_step_global * gradient_accumulation)

                loss_accumulated_steps = 0

                if lr_scheduler_restart_every_steps > 0 and times_optimizer_step_global % lr_scheduler_restart_every_steps == 0:
                    # Before optimizer.step, so in the next iteration, the new learning rate will be applied (but in the current iteration the optimizer will use the current learning rate)
                    logger.info("Restarting LR scheduler at global step: %d", times_optimizer_step_global)

                    #lr_scheduler_args[0] = "0" # disable warmup after the first lr_scheduler_restart_every_steps steps
                    scheduler = utils.get_lr_scheduler_and_optimizer_using_argparse_values(None, lr_scheduler_str, None, lr_scheduler_args, None, learning_rate, training_steps_lr_scheduler, training_steps_per_epoch_lr_scheduler, logger, _optimizer=optimizer)[1]

                optimizer.step()
                scheduler.step()

                model.zero_grad()

                if show_statistics:
                    logger.debug("Current learning rate: %s", optimizer.param_groups[0]["lr"])

            if (batch_idx % (log_steps * gradient_accumulation)) == 0:
                sum_partial_loss = sum(epoch_loss[-1 * log_steps:]) # no: -1 * log_steps * gradient_accumulation!
                sum_loss = sum(epoch_loss)

                logger.info("Batch #%d: %s (last %d steps: %s)", batch_idx, sum_loss, log_steps * gradient_accumulation, sum_partial_loss)

                sys.stdout.flush()
                sys.stderr.flush()

        assert batch_idx == training_steps_per_epoch, f"{batch_idx} vs {training_steps_per_epoch}"

        sum_epoch_loss = sum(epoch_loss)
        mean_epoch_loss = np.mean(epoch_loss)

        logger.info("Epoch loss: %s", sum_epoch_loss)
        # CosineEmbeddingLoss: mean loss range is [0, 2], without negative pairs
        # CosineEmbeddingLoss: cos=1-loss, and range is [-1, 1], where -1 is opposite, and 1 is perfect alignment
        logger.info("Epoch loss (mean): %s%s", mean_epoch_loss, f" (cos = 1 - loss = {1.0 - mean_epoch_loss} )" if knn_distance_ip else '')
        logger.info("Epoch cosine similarity (mean): %s", np.mean(epoch_cos_similarity) if len(epoch_cos_similarity) > 0 else None)
        logger.info("Epoch cosine similarity (mean per label): %s", {label: np.mean(epoch_cos_similarity_per_label[label]) if len(epoch_cos_similarity_per_label[label]) > 0 else None for label in range(num_labels_reward)})

        assert str(sum_epoch_loss) != "nan", "Some values in the input data are NaN"
        assert str(mean_epoch_loss) != "nan", "Some values in the input data are NaN"

        if mean_epoch_loss < early_stopping_best_loss:
            # we use the mean instead of the sum to avoid problems with the growth of the training set
            logger.info("Better loss result: %s -> %s", early_stopping_best_loss, mean_epoch_loss)

            early_stopping_best_loss = mean_epoch_loss
        else:
            logger.info("No improvement in loss (best: %s): %s", early_stopping_best_loss, mean_epoch_loss)

        sys.stdout.flush()
        sys.stderr.flush()

        epoch += 1
        do_training = epoch < epochs or train_until_patience

    with torch.no_grad():
        logger.info("Loading best model for final evaluation: %s", save_model_path)

        model_weights = torch.load(save_model_path)

        model.load_state_dict(model_weights)

        model = model.to(device)

        mean_reward, std_reward = evaluate_policy_custom(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated_dev), render=False, device=device,
                                                         reward_idx=num_labels_reward - 1, logger=logger, max_icl_examples=max_icl_examples,
                                                         l2_norm_model_output=knn_distance_ip and disable_l2_norm_in_model)

        logger.info("Final dev eval: %s +- %s", mean_reward, std_reward)

if __name__ == "__main__":
    main()
