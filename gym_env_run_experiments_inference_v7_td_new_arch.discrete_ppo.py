
import sys
import logging

import utils
from gym_env_v1_eval import MTICLEvalEnv

from gym_env_run_experiments import InverseSqrtWithWarmUpLRSchedule
import numpy as np

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor, NFeaturesExtractor, TransformerExtractor, NFeaturesExtractorWithTimeStepEmbeddings
from stable_baselines3.common.buffers import NStepReplayBuffer, MonteCarloReplayBuffer
import torch
from stable_baselines3.common.policies import ContinuousCritic, ContinuousCriticTower

def main():
    # Evaluate
    best_model_path = sys.argv[1]
    src_lang = sys.argv[2]
    trg_lang = sys.argv[3]
    file_data = sys.argv[4]
    file_data_icl_examples = sys.argv[5]
    parsed_kwargs = utils.parse_args(sys.argv[6:])

    assert best_model_path.endswith(".zip"), best_model_path
    assert utils.file_exists(best_model_path), f"Best model not found: {best_model_path}"

    # read data
    data_to_be_translated = []
    _data_icl_examples = []

    for _file_data, data_to_be_translated in ((file_data, data_to_be_translated),):
        with open(_file_data, "rt") as fd:
            for line in fd:
                data_to_be_translated.append(line.rstrip("\r\n"))

    for _file_data, data_icl_examples in ((file_data_icl_examples, _data_icl_examples),):
        with open(_file_data, "rt") as fd:
            for line in fd:
                data_icl_examples.append(line.rstrip("\r\n"))

    # parse args
    max_data_entries = int(parsed_kwargs.get("max_data_entries", -1)) # load all data (default value)
    max_data_icl_examples_entries = int(parsed_kwargs.get("max_data_icl_examples_entries", -1)) # load all data (default value)
    #max_data_entries = 5 # TODO remove
    #max_data_entries = 10 # TODO remove
    #max_data_icl_examples_entries = 100 # TODO remove
    #max_data_icl_examples_entries = 10 # TODO remove
    device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    state_representation = parsed_kwargs.get("state_representation", "representation_per_token_with_features")
    max_icl_examples = int(parsed_kwargs.get("max_icl_examples", 5))
    parsed_kwargs["max_icl_examples"] = max_icl_examples
    parsed_kwargs["max_data_entries"] = max_data_entries
    parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    parsed_kwargs["device"] = device
    parsed_kwargs["state_representation"] = state_representation
    parsed_kwargs["eval_strategy"] = parsed_kwargs.get("eval_strategy", "api-eval")
    parsed_kwargs["gym_logger_level"] = parsed_kwargs.get("gym_logger_level", gym.logger.DEBUG)
    parsed_kwargs["enable_eos_action"] = parsed_kwargs.get("enable_eos_action", False)
    parsed_kwargs["model_hidden_size_action_src_sentence"] = parsed_kwargs.get("model_hidden_size_action_src_sentence", 1024)
    parsed_kwargs["actions_without_replacement"] = parsed_kwargs.get("actions_without_replacement", False)
    parsed_kwargs["current_icl_examples_prepend"] = bool(int(parsed_kwargs.get("current_icl_examples_prepend", False)))
    parsed_kwargs["model_hidden_size"] = parsed_kwargs.get("model_hidden_size", 1536)
    parsed_kwargs["action_representation"] = "discrete_index"

    if state_representation == "representation_mean_plus_last_75_perc_layer_and_relative_diff":
        parsed_kwargs["model_hidden_size"] *= 2

    data_to_be_translated = data_to_be_translated[:max_data_entries if max_data_entries > 0 else None]
    _data_icl_examples = _data_icl_examples[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]

    process_token_time_step = bool(int(parsed_kwargs.get("process_token_time_step", True)))
    parsed_kwargs["process_token_time_step"] = process_token_time_step
    parsed_kwargs["num_icl_examples"] = len(_data_icl_examples)

    process_token_time_step = False # TODO change when implemented and see how to feed time_steps (replay_buffer_kwargs is not an argument anymore)

    if "_seed" in parsed_kwargs or "seed" in parsed_kwargs:
        seed = parsed_kwargs.pop("_seed", None)

        if seed is None:
            seed = parsed_kwargs.pop("seed")

        seed = int(seed)
    else:
        seed = 42

    utils.set_random_seed(seed)

    # custom
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.rl_experiments"), level=logging.DEBUG)

    logger.info("Seed: %s", seed)

    env_args = [src_lang, trg_lang, file_data, file_data_icl_examples]

    ## load model
    logger.info("Loading model: %s", best_model_path)

    env_eval_dev_class = MTICLEvalEnv
    env_eval_dev = Monitor(env_eval_dev_class(*list(env_args), custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs), filename=None, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={})

    action_dim = env_eval_dev.action_dim
    state_dim_per_token = env_eval_dev.state_dim_per_token
    state_window_length = env_eval_dev.state_window_length
    state_dim_per_token_time_step = env_eval_dev.state_dim_per_token_time_step

    if state_representation == "representation_per_token_with_features":
        n_features = state_dim_per_token * (state_window_length - 1) # -1 due to the action representation which we skip
    elif state_representation in ("representation_last_token_current_and_relative_diff", "representation_mean_plus_last_75_perc_layer_and_relative_diff"):
        n_features = state_dim_per_token * 2 + (state_dim_per_token_time_step if process_token_time_step else 0)
    elif state_representation == "representation_mean_75_perc_layer":
        n_features = state_dim_per_token + (state_dim_per_token_time_step if process_token_time_step else 0)
    else:
        n_features = 0

    #net_arch = [512, 128, 32]
    #net_arch = [512, 256, 128]

    net_arch = {
        "pi": [512, 512],
        "qf": [512, 256, 128]
    } # "pi" is actor and "qf" the critic

    logger.info("net_arch: %s", net_arch)

    if n_features <= 0:
        features_extractor_class = FlattenExtractor
        features_extractor_kwargs = {}
    else:
        features_extractor_class = NFeaturesExtractorWithTimeStepEmbeddings
        n = n_features
        skip_n = action_dim

        if state_representation == "representation_per_token_with_features":
            step_embeddings_dim = state_dim_per_token
            n -= step_embeddings_dim # "- step_embeddings_dim" to add the time_step embedding
            skip_n += state_dim_per_token * 2 # "state_dim_per_token * 2" to remove the time_step information
        elif state_representation in ("representation_last_token_current_and_relative_diff", "representation_mean_plus_last_75_perc_layer_and_relative_diff", "representation_mean_75_perc_layer"):
            step_embeddings_dim = state_dim_per_token_time_step
            #n -= state_dim_per_token_time_step

            if process_token_time_step:
                skip_n += state_dim_per_token_time_step # the model adds the time step information to the features, so we need to skip it
        else:
            raise Exception()

        features_extractor_kwargs = {
            "n": n,
            "skip_n": skip_n,
            "state_dim_per_token": state_dim_per_token,
            "check_zeros": True if state_representation == "representation_per_token_with_features" else False,
            "step_embeddings": max_icl_examples + 1, # add embeddings for each time step (+1 to avoid error in the model forward for computing next_actions, although the result will be discarded)
            "step_embeddings_dim": step_embeddings_dim,
        }

    model_class = PPO
    model = model_class.load(
        best_model_path,
        learning_rate=lambda foo: 100.0, # dummy callable
        lr_schedule=lambda foo: 100.0, # dummy callable
        policy_kwargs={
            "net_arch": dict(net_arch),
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs,
            "layer_norm_input": True,
            "layer_norm_before_activation": True,
            "activation_fn": torch.nn.GELU,
            "avoid_overlapping_action": True,
        },
        device=device,
        #replay_buffer_kwargs={"process_time_steps": process_token_time_step},
    )

    ## dev: evaluate and report result
    logger.info("Evaluating dev")

    #mean_reward, std_reward = evaluate_policy(model, env_eval_dev.unwrapped, n_eval_episodes=len(data_to_be_translated))
    episode_rewards, episode_lengths = evaluate_policy(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated), return_episode_rewards=True,
        predict_kwargs={
            "env_instance": env_eval_dev.unwrapped,
        },)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"Mean reward dev: {mean_reward} +/- {std_reward} (length: {mean_length} +/- {std_length})")

if __name__ == "__main__":
    main()
