
from collections import Counter
import sys
import random
import logging

import utils
from gym_env_v1_eval import MTICLEvalEnv

from gym_env_run_experiments import make_env
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor, NFeaturesExtractor, TransformerExtractor, NFeaturesExtractorWithTimeStepEmbeddings
from stable_baselines3.common.buffers import NStepReplayBuffer, MonteCarloReplayBuffer
import torch
from stable_baselines3.common.policies import ContinuousCritic, ContinuousCriticTower
from stable_baselines3.common.vec_env.vec_normalize import VecNormalizeRangeAndRewardSentenceLevelICL

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

    initial_parsed_kwargs = dict(parsed_kwargs)

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
    #max_data_entries = 100 # TODO remove
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
    parsed_kwargs["eval_strategy_training"] = parsed_kwargs.get("eval_strategy_training", "chrf2")
    parsed_kwargs["eval_strategy_eval"] = parsed_kwargs.get("eval_strategy_eval", "chrf2")
    parsed_kwargs["repeat_translation_candidates"] = parsed_kwargs.get("repeat_translation_candidates", False)
    parsed_kwargs["repeat_translation_candidates_times"] = int(parsed_kwargs.get("repeat_translation_candidates_times", 0))
    parsed_kwargs["gym_logger_level"] = parsed_kwargs.get("gym_logger_level", gym.logger.DEBUG)
    parsed_kwargs["enable_eos_action"] = parsed_kwargs.get("enable_eos_action", False)
    parsed_kwargs["model_hidden_size_action_src_sentence"] = parsed_kwargs.get("model_hidden_size_action_src_sentence", 1024)
    parsed_kwargs["actions_without_replacement"] = parsed_kwargs.get("actions_without_replacement", False)
    parsed_kwargs["current_icl_examples_prepend"] = bool(int(parsed_kwargs.get("current_icl_examples_prepend", False)))
    parsed_kwargs["model_hidden_size"] = parsed_kwargs.get("model_hidden_size", 1536)
    parsed_kwargs["action_representation"] = "discrete_index"
    parsed_kwargs["multi_step_eval"] = bool(int(parsed_kwargs.get("multi_step_eval", 0)))

    if state_representation == "representation_mean_plus_last_75_perc_layer_and_relative_diff":
        parsed_kwargs["model_hidden_size"] *= 2

    data_to_be_translated = data_to_be_translated[:max_data_entries if max_data_entries > 0 else None]
    _data_icl_examples = _data_icl_examples[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]

    process_token_time_step = bool(int(parsed_kwargs.get("process_token_time_step", True)))

    assert state_representation in ("representation_one_hot_representation_time_and_selected_icl_examples", "representation_one_hot_representation_time_and_selected_icl_examples_external_embedding_src_and_last_example", "representation_per_token_with_features_v2", "representation_per_token_with_features_v3"), f"Some values such as icl_mask_duplicates_last_values_from_state expect this configuration: {state_representation}"

    if state_representation in ("representation_one_hot_representation_time_and_selected_icl_examples", "representation_one_hot_representation_time_and_selected_icl_examples_external_embedding_src_and_last_example", "representation_per_token_with_features_v2", "representation_per_token_with_features_v3"):
        process_token_time_step = False

    parsed_kwargs["process_token_time_step"] = process_token_time_step
    parsed_kwargs["num_icl_examples"] = len(_data_icl_examples)

    if "_seed" in parsed_kwargs or "seed" in parsed_kwargs:
        seed = parsed_kwargs.pop("_seed", None)

        if seed is None:
            seed = parsed_kwargs.pop("seed")

        seed = int(seed)
    else:
        seed = 42

    utils.set_random_seed(seed)

    num_envs = 80
    vec_env_class = SubprocVecEnv
    vec_env_kwargs = {"start_method": "forkserver"} if vec_env_class is SubprocVecEnv else {}
    randint_values = (1, 1000)
    env_seeds = [random.randint(*randint_values) for _ in range(num_envs)]

    for rank in range(1, num_envs):
        while env_seeds[rank] in env_seeds[:rank]:
            env_seeds[rank] = random.randint(*randint_values)

    assert len(env_seeds) == len(set(env_seeds)) == num_envs

    # custom
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.rl_experiments"), level=logging.DEBUG)
    linear_bottleneck = int(parsed_kwargs.pop("linear_bottleneck", 0))
    activation_fn_str = parsed_kwargs.pop("activation_fn", "relu")
    use_vec_normalize = bool(int(parsed_kwargs.pop("use_vec_normalize", 1)))
    store_rewards_fn = parsed_kwargs.pop("store_rewards_fn", None)
    available_actions_strategy = parsed_kwargs.get("available_actions_strategy", "bm25_and_sonar_embeddings")
    parsed_kwargs["available_actions_strategy"] = available_actions_strategy
    parsed_kwargs["available_actions_strategy_n"] = int(parsed_kwargs.get("available_actions_strategy_n", 5))
    use_transformer = bool(int(parsed_kwargs.pop("use_transformer", 0)))
    activation_fn_str = "gelu" if use_transformer and activation_fn_str == "relu" else activation_fn_str
    activation_fn = utils.get_activation_cls(activation_fn_str)

    if state_representation == "representation_one_hot_representation_time_and_selected_icl_examples_external_embedding_src_and_last_example":
        parsed_kwargs["apply_l2_normalization_state"] = False

    if state_representation == "representation_per_token_with_features_v2":
        parsed_kwargs["apply_l2_normalization_state"] = False
    elif state_representation == "representation_per_token_with_features_v3":
        parsed_kwargs["apply_l2_normalization_state"] = True

    if use_vec_normalize:
        logger.info("Using VecNormalize for normalizing observations and rewards")
        parsed_kwargs["apply_l2_normalization_state"] = False

    logger.info("Seed: %s", seed)
    logger.info("Seed: %s (env_seeds: %s)", seed, env_seeds)

    #env_args = [src_lang, trg_lang, file_data, file_data_icl_examples]

    assert len(data_to_be_translated) >= num_envs

    file_data_per_env = {n: None for n in range(num_envs)}
    bsz = len(data_to_be_translated) // num_envs
    start = 0
    end = bsz
    n_eval_episodes = [0 for _ in range(num_envs)]

    assert (bsz + 1) * num_envs >= len(data_to_be_translated), f"{bsz} * {num_envs} < {len(data_to_be_translated)}"

    for n in range(num_envs):
        file_data_per_env[n] = list(data_to_be_translated[start:end])
        n_eval_episodes[n] = len(file_data_per_env[n])
        start = end
        end += bsz

        assert n_eval_episodes[n] > 0, n

    if start < len(data_to_be_translated):
        first = True
        idx = 0

        while sum(n_eval_episodes) < len(data_to_be_translated):
            if num_envs > 1 and first and n_eval_episodes[n] < n_eval_episodes[n-1]:
                file_data_per_env[n].extend(data_to_be_translated[start:start+1])
                n_eval_episodes[n] = len(file_data_per_env[n])
                start += 1

                if n_eval_episodes[n] >= n_eval_episodes[n-1]:
                    first = False

            file_data_per_env[idx % num_envs].extend(data_to_be_translated[start:start+1])
            n_eval_episodes[idx % num_envs] = len(file_data_per_env[idx % num_envs])
            idx += 1
            start += 1

    assert sum(n_eval_episodes) == len(data_to_be_translated)

    _all_data = [file_data_per_env[n] for n in range(num_envs)]
    _all_data = sorted([x for xs in _all_data for x in xs])

    assert len(_all_data) == len(data_to_be_translated)
    assert _all_data == sorted(data_to_be_translated)

    logger.info("num_envs: %s, data_to_be_translated: %s, bsz: %s, n_eval_episodes: %s", num_envs, len(data_to_be_translated), bsz, n_eval_episodes)

    parsed_kwargs_removed_elements = {k: initial_parsed_kwargs[k] for k in initial_parsed_kwargs.keys() if k not in parsed_kwargs.keys()}

    logger.info("parsed_kwargs: %s", parsed_kwargs)
    logger.info("parsed_kwargs_removed_elements: %s", parsed_kwargs_removed_elements)

    ## load model
    logger.info("Loading model: %s", best_model_path)

    env_eval_dev_class = MTICLEvalEnv
    env_eval_dev = Monitor(vec_env_class([make_env(rank, env_eval_dev_class, [src_lang, trg_lang, file_data_per_env[rank], file_data_icl_examples], dict({"custom_env_id": f"eval_dev_{str(rank)}", "is_eval_env": True, "_parallel_env": True, **parsed_kwargs}), seed=env_seeds[rank]) for rank in range(num_envs)], **vec_env_kwargs))

    #env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={})
    env_eval_dev.unwrapped.env_method("_init_load_data_and_populate_knn_pool", options={})

    action_dim = env_eval_dev.unwrapped.get_attr("action_dim")[0]
    state_dim_per_token = env_eval_dev.unwrapped.get_attr("state_dim_per_token")[0]
    state_window_length = env_eval_dev.unwrapped.get_attr("state_window_length")[0]
    state_dim_per_token_time_step = env_eval_dev.unwrapped.get_attr("state_dim_per_token_time_step")[0]
    model_hidden_size_embedding = env_eval_dev.unwrapped.get_attr("model_hidden_size_embedding")[0]

    if state_representation == "representation_per_token_with_features":
        n_features = state_dim_per_token * (state_window_length - 1) # -1 due to the action representation which we skip
    elif state_representation in ("representation_last_token_current_and_relative_diff", "representation_mean_plus_last_75_perc_layer_and_relative_diff"):
        n_features = state_dim_per_token * 2 + (state_dim_per_token_time_step if process_token_time_step else 0)
    elif state_representation in ("representation_mean_75_perc_layer", "representation_last_75_perc_layer"):
        n_features = state_dim_per_token + (state_dim_per_token_time_step if process_token_time_step else 0)
    elif state_representation == "representation_one_hot_representation_time_and_selected_icl_examples":
        n_features = state_dim_per_token + (max_icl_examples + 1) + len(_data_icl_examples)
    elif state_representation == "representation_one_hot_representation_time_and_selected_icl_examples_external_embedding_src_and_last_example":
        n_features = model_hidden_size_embedding * 2 + (max_icl_examples + 1) + len(_data_icl_examples)
    elif state_representation in ("representation_per_token_with_features_v2", "representation_per_token_with_features_v3"):
        n_features = state_dim_per_token * (state_window_length - 4) + (max_icl_examples + 1) + len(_data_icl_examples)
    else:
        n_features = 0

    assert best_model_path.endswith("_model.zip"), best_model_path

    vec_normalize_path = best_model_path[:-len("_model.zip")] + "_vecnormalize.pkl"

    if use_vec_normalize:
        assert utils.file_exists(vec_normalize_path), f"VecNormalize file not found: {vec_normalize_path}"
        #assert state_representation in ("representation_one_hot_representation_time_and_selected_icl_examples", "representation_one_hot_representation_time_and_selected_icl_examples_external_embedding_src_and_last_example", "representation_per_token_with_features_v2")

        env_eval_dev = VecNormalizeRangeAndRewardSentenceLevelICL.load(vec_normalize_path, env_eval_dev)

        assert isinstance(env_eval_dev, VecNormalizeRangeAndRewardSentenceLevelICL), f"Expected env_eval_dev to be an instance of VecNormalizeRangeAndRewardSentenceLevelICL, but got {type(env_eval_dev)}"
        assert isinstance(env_eval_dev.unwrapped, vec_env_class), f"Expected env_eval_dev.unwrapped to be an instance of {vec_env_class}, but got {type(env_eval_dev.unwrapped.unwrapped)}"

        env_eval_dev.training = False
    else:
        assert not utils.file_exists(vec_normalize_path), f"VecNormalize file found but use_vec_normalize is False: {vec_normalize_path}"

    #net_arch = [512, 128, 32]
    #net_arch = [512, 256, 128]

    net_arch = {
        #"pi": [1024, 1024],
        #"vf": [256, 256]
        "pi": [256, 256],
        "vf": [256, 256]
    } # "pi" is actor and "vf" the critic

    logger.info("net_arch: %s, linear_bottleneck: %s, activation_fn: %s", net_arch, linear_bottleneck, activation_fn)

    if use_transformer:
        logger.info("Transformer enabled: using transformer+MLP")

        net_arch = {
            "pi": [],
            "vf": [],
            "empty_layers": True, # custom code
        }

    if n_features <= 0:
        features_extractor_class = FlattenExtractor
        features_extractor_kwargs = {}
    else:
        features_extractor_class = NFeaturesExtractorWithTimeStepEmbeddings
        n = n_features
        skip_n = action_dim

        if state_representation == "representation_per_token_with_features":
            step_embeddings = max_icl_examples + 1
            step_embeddings_dim = state_dim_per_token
            n -= step_embeddings_dim # "- step_embeddings_dim" to add the time_step embedding
            skip_n += state_dim_per_token * 2 # "state_dim_per_token * 2" to remove the time_step information
        elif state_representation in ("representation_last_token_current_and_relative_diff", "representation_mean_plus_last_75_perc_layer_and_relative_diff", "representation_mean_75_perc_layer", "representation_last_75_perc_layer"):
            step_embeddings = max_icl_examples + 1
            step_embeddings_dim = state_dim_per_token_time_step
            #n -= state_dim_per_token_time_step

            if process_token_time_step:
                skip_n += state_dim_per_token_time_step # the model adds the time step information to the features, so we need to skip it
        elif state_representation in ("representation_one_hot_representation_time_and_selected_icl_examples", "representation_one_hot_representation_time_and_selected_icl_examples_external_embedding_src_and_last_example", "representation_per_token_with_features_v2", "representation_per_token_with_features_v3"):
            step_embeddings = 0
            step_embeddings_dim = 0
            skip_n += len(_data_icl_examples) # one-hot representation of available actions
        else:
            raise Exception()

        if use_transformer:
            assert state_representation in ("representation_per_token_with_features_v2", "representation_per_token_with_features_v3"), f"Unexpected state_representation: {state_representation}"
            assert state_window_length - 4 > 0, f"Expected state_window_length to be greater than 4 for state representation {state_representation}, but got {state_window_length}"

            #transformer_d_model = 256
            #transformer_d_model = 128
            #transformer_d_model = 64
            transformer_d_model = 32
            #transformer_d_model = 16

            if state_representation == "representation_per_token_with_features_v3":
                transformer_d_model = 256

            dropout_p = 0.0
            transformer_kwargs = {
                "d_model": transformer_d_model,
                #"nhead": 4,
                "nhead": 2,
                "dim_feedforward": transformer_d_model * 4,
                "nlayers": 2,
                #"nlayers": 1,
                "projection_in": state_dim_per_token,
                "projection_out": None, # let the MLP layers after the feature extractor handle the rest of the processing
                "activation": activation_fn_str,
                "bias": True,
                "norm_first": True,
                "initial_layer_norm": True,
                "initial_layer_norm_first": True,
                "embedding_dropout": dropout_p, # it can increse the variance in the training
                "dropout_p": dropout_p, # we can disable dropout setting to 0.0 if needed
                "projection_out_dropout_p": dropout_p,
                "max_seq_len": 8192, # the positional encoding is absolute and using this big value does not affect to the previous positions
                "skip_n_word_embeddings_from_observation": "0:0",
                "expected_seq_len": state_window_length - 4,
                "last_layer_norm": False,
                "last_linear_layer": True,
                "check_zeros": False,
                "remove_first_column_of_zeros": True if state_representation == "representation_per_token_with_features_v2" else False,
                "step_embeddings": max_icl_examples + 1, # add embeddings for each time step (+1 to avoid error in the model forward for computing next_actions, although the result will be discarded)
                "step_embeddings_from_observation": True, # we expect the time step information to be included in the observation, so we can use it for the step embeddings
                "action_embeddings": len(_data_icl_examples),
                "max_actions": max_icl_examples,
                "l2_norm": False,
                "mean_pooling": True,
                "reward_embeddings": 0,
                "init_zeros_last_layer": False,
                "use_first_n_tokens": -1,
                #"initial_layer_norm_first_additional_layer_norm_after_projection": False,
                "initial_layer_norm_first_additional_layer_norm_after_projection": True,
                "str_id": "feature_extractor",
                "disable_step_embeddings": True,
                "mask_tokens_p": 0.0,
            }
            #custom_features_dim = projection_out if projection_out is not None else d_model
            custom_features_dim = transformer_d_model
        else:
            transformer_kwargs = {}
            custom_features_dim = None

        features_extractor_kwargs = {
            "n": n,
            "skip_n": skip_n,
            "state_dim_per_token": state_dim_per_token,
            "check_zeros": True if state_representation == "representation_per_token_with_features" else False,
            "step_embeddings": step_embeddings, # add embeddings for each time step (+1 to avoid error in the model forward for computing next_actions, although the result will be discarded)
            "step_embeddings_dim": step_embeddings_dim,
            "linear_bottleneck": linear_bottleneck,
            "activation_fn": activation_fn,
            "use_transformer": use_transformer,
            "transformer_kwargs": transformer_kwargs,
            "custom_features_dim": custom_features_dim,
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
            "share_features_extractor": False,
            #"layer_norm_input": True,
            "layer_norm_input": True if linear_bottleneck > 0 else False,
            "layer_norm_before_activation": True,
            "activation_fn": activation_fn,
            "avoid_overlapping_action": True,
            "icl_mask_duplicates_last_values_from_state": len(_data_icl_examples),
            "check_general_actions_masking": True,
            "temperature": 2.0 if available_actions_strategy == "none" else 1.0,
        },
        device=device,
        rollout_buffer_kwargs={"process_time_steps": process_token_time_step},
    )

    ## dev: evaluate and report result
    logger.info("Evaluating dev")

    #mean_reward, std_reward = evaluate_policy(model, env_eval_dev.unwrapped, n_eval_episodes=len(data_to_be_translated))
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env_eval_dev.unwrapped,
        n_eval_episodes=[bsz + 1] * num_envs,
        return_episode_rewards=True,
        predict_kwargs={
            "env_instance": env_eval_dev.unwrapped,
        },
    )

    episode_rewards = []
    episode_lengths = []
    episode_available_actions_strategy_statistics = {}
    all_rewards_data = env_eval_dev.unwrapped.get_attr("rewards")
    all_available_actions_strategy_statistics_data = env_eval_dev.unwrapped.get_attr("available_actions_strategy_statistics")
    all_rewards = list([[[r2[0] for r2 in r] for r in rewards_data] for rewards_data in all_rewards_data])
    src_mt_ref_sentences = list([[[r2[1] for r2 in r] for r in rewards_data] for rewards_data in all_rewards_data])

    assert len(all_rewards_data) == num_envs
    assert len(all_rewards) == len(n_eval_episodes) == num_envs
    assert len(src_mt_ref_sentences) == len(n_eval_episodes) == num_envs
    assert len(all_available_actions_strategy_statistics_data) == len(n_eval_episodes) == num_envs

    for i in range(num_envs): # env
        # Remove extra evaluations
        assert isinstance(src_mt_ref_sentences[i], list)
        assert isinstance(all_available_actions_strategy_statistics_data[i], list)

        all_rewards[i] = all_rewards[i][:n_eval_episodes[i]]
        src_mt_ref_sentences[i] = src_mt_ref_sentences[i][:n_eval_episodes[i]]
        all_available_actions_strategy_statistics_data[i] = all_available_actions_strategy_statistics_data[i][:n_eval_episodes[i]]

        assert len(all_rewards[i]) == n_eval_episodes[i]
        assert len(src_mt_ref_sentences[i]) == n_eval_episodes[i]
        assert len(all_available_actions_strategy_statistics_data[i]) == n_eval_episodes[i]

        for j in range(len(all_rewards[i])): # episode
            assert isinstance(all_rewards[i][j], list)
            assert isinstance(all_available_actions_strategy_statistics_data[i][j], list)
            assert all(isinstance(d, list) for d in all_available_actions_strategy_statistics_data[i][j]), f"{i} {j} {all_available_actions_strategy_statistics_data[i][j]}"

            for k in range(len(all_available_actions_strategy_statistics_data[i][j])): # step
                assert isinstance(all_available_actions_strategy_statistics_data[i][j][k], list), f"{i} {j} {k} {all_available_actions_strategy_statistics_data[i][j]}"
                assert all(isinstance(d, tuple) for d in all_available_actions_strategy_statistics_data[i][j][k]), f"{i} {j} {k} {all_available_actions_strategy_statistics_data[i][j]}"

                if k + 1 not in episode_available_actions_strategy_statistics:
                    episode_available_actions_strategy_statistics[k + 1] = []

                only_strategy_and_rank = [(d[0], d[1]) for d in all_available_actions_strategy_statistics_data[i][j][k]]
                episode_available_actions_strategy_statistics[k + 1].extend(only_strategy_and_rank)

            if len(all_rewards[i][j]) == 0:
                #del all_rewards[i][j]

                assert len(src_mt_ref_sentences[i][j]) == 0, f"{i} {j} {src_mt_ref_sentences[i][j]}"
            else:
                episode_lengths.append(len(all_rewards[i][j]))

                all_rewards[i][j] = sum(all_rewards[i][j])

                assert len(src_mt_ref_sentences[i][j]) > 0, f"{i} {j} {src_mt_ref_sentences[i][j]}"

            assert isinstance(src_mt_ref_sentences[i][j], list), f"{i} {j} {src_mt_ref_sentences[i]}"

            src_sentences = [src.split("\t")[0] for src in src_mt_ref_sentences[i][j]]

            assert len(set(src_sentences)) in (0, 1), src_mt_ref_sentences[i][j]

            if len(src_mt_ref_sentences[i][j]) == 0:
                del src_mt_ref_sentences[i][j]

                assert len(all_available_actions_strategy_statistics_data[i][j]) == 0, f"{i} {j} {all_available_actions_strategy_statistics_data[i][j]}"
                assert isinstance(all_rewards[i][j], list), f"{i} {j} {all_rewards[i][j]}"
                assert len(all_rewards[i][j]) == 0, f"{i} {j} {all_rewards[i][j]}"
            else:
                src_mt_ref_sentences[i][j] = src_mt_ref_sentences[i][j][-1] # the last step is the one with the mt

                assert len(all_available_actions_strategy_statistics_data[i][j]) > 0, f"{i} {j} {all_available_actions_strategy_statistics_data[i][j]}"
                assert isinstance(all_rewards[i][j], (int, float)), f"{i} {j} {all_rewards[i][j]}"
                assert isinstance(src_mt_ref_sentences[i][j], str), f"{i} {j} {src_mt_ref_sentences[i][j]}"

        episode_rewards.extend(all_rewards[i])

        assert len(episode_rewards) == len(episode_lengths)

    for episode_step in episode_available_actions_strategy_statistics:
        # count (strategy, rank) for each episode step
        episode_available_actions_strategy_statistics[episode_step] = Counter(episode_available_actions_strategy_statistics[episode_step])

        print(f"episode_available_actions_strategy_statistics: episode step {episode_step}: {episode_available_actions_strategy_statistics[episode_step]}")

    sys.stdout.flush()

    src_mt_ref_sentences = [x for xs in src_mt_ref_sentences for x in xs]
    src_and_ref_sentences = [s.split("\t")[0] + "\t" + s.split("\t")[2] for s in src_mt_ref_sentences]

    assert len(src_mt_ref_sentences) == len(data_to_be_translated), f"{len(src_mt_ref_sentences)} vs {len(data_to_be_translated)}"
    assert len(src_and_ref_sentences) == len(data_to_be_translated), f"{len(src_and_ref_sentences)} vs {len(data_to_be_translated)}"
    assert len(set(src_and_ref_sentences)) == len(set(data_to_be_translated)), f"{len(set(src_and_ref_sentences))} vs {len(set(data_to_be_translated))}"
    assert set(src_and_ref_sentences) == set(data_to_be_translated), f"src_and_ref_sentences not matching data_to_be_translated: {set(src_and_ref_sentences).symmetric_difference(set(data_to_be_translated))}" # symmetric_difference -> elements not shared
    assert len(episode_rewards) == len(src_mt_ref_sentences), f"{len(episode_rewards)} vs {len(src_mt_ref_sentences)}"
    assert len(episode_lengths) == len(src_mt_ref_sentences), f"{len(episode_lengths)} vs {len(src_mt_ref_sentences)}"

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"Mean reward dev: {mean_reward} +/- {std_reward} (length: {mean_length} +/- {std_length})")

    if store_rewards_fn is not None:
        src, mt, ref = zip(*[s.split("\t") for s in src_mt_ref_sentences])

        # sort by source sentence to make it easier to analyze results
        sorted_data = sorted(zip(src, mt, ref, episode_rewards, episode_lengths), key=lambda x: x[0])
        src, mt, ref, episode_rewards, episode_lengths = zip(*sorted_data)

        with open(store_rewards_fn, "wt") as fd:
            for episode_reward, episode_length, s, m, r in zip(episode_rewards, episode_lengths, src, mt, ref):
                fd.write(f"{episode_reward}\t{episode_length}\t{s}\t{m}\t{r}\n")

        print(f"Rewards and lengths stored in (sorted by source sentence): {store_rewards_fn}")

if __name__ == "__main__":
    main()
