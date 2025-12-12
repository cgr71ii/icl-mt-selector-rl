
import sys
import logging

import utils
from gym_env_v1_eval import MTICLEvalEnv

from gym_env_run_experiments import InverseSqrtWithWarmUpLRSchedule
import numpy as np

import gymnasium as gym
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor, NoFlattenExtractor, TransformerExtractor

if __name__ == "__main__":
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
    max_data_entries = parsed_kwargs.get("max_data_entries", -1) # load all data (default value)
    max_data_icl_examples_entries = parsed_kwargs.get("max_data_icl_examples_entries", -1) # load all data (default value)
    #max_data_entries = 5 # TODO remove
    #max_data_icl_examples_entries = 100 # TODO remove
    state_representation = parsed_kwargs.get("state_representation", "representation_per_token_with_features")
    parsed_kwargs["max_icl_examples"] = parsed_kwargs.get("max_icl_examples", 5)
    parsed_kwargs["max_data_entries"] = max_data_entries
    parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    parsed_kwargs["device"] = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    parsed_kwargs["state_representation"] = state_representation
    parsed_kwargs["eval_strategy"] = parsed_kwargs.get("eval_strategy", "api-eval")
    parsed_kwargs["knn_api_retrieve"] = parsed_kwargs.get("knn_api_retrieve", None)
    parsed_kwargs["knn_api_insert"] = parsed_kwargs.get("knn_api_insert", None)
    parsed_kwargs["gym_logger_level"] = parsed_kwargs.get("gym_logger_level", gym.logger.DEBUG)
    parsed_kwargs["knn_always_add_eos_action"] = parsed_kwargs.get("knn_always_add_eos_action", True)
    parsed_kwargs["enable_eos_action"] = parsed_kwargs.get("enable_eos_action", False)
    parsed_kwargs["state_window_length"] = parsed_kwargs.get("state_window_length", 1024 + 1)
    parsed_kwargs["action_representation"] = parsed_kwargs.get("action_representation", "src_embedding:SONAR")
    parsed_kwargs["model_hidden_size_action_src_sentence"] = parsed_kwargs.get("model_hidden_size_action_src_sentence", 1024)

    if "_seed" in parsed_kwargs or "seed" in parsed_kwargs:
        seed = parsed_kwargs.pop("_seed", None)

        if seed is None:
            seed = parsed_kwargs.pop("seed")

        seed = int(seed)
    else:
        seed = 42

    utils.set_random_seed(seed)

    data_to_be_translated = data_to_be_translated[:max_data_entries if max_data_entries > 0 else None]
    _data_icl_examples = _data_icl_examples[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]

    # custom
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.rl_experiments"), level=logging.DEBUG)
    #max_data_icl_examples_entries = 100 # TODO remove
    #max_data_entries = 1 # TODO remove
    #parsed_kwargs["max_data_entries"] = max_data_entries
    #parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    #data_to_be_translated = data_to_be_translated[:max_data_entries if max_data_entries > 0 else None]
    model_hidden_size = parsed_kwargs.get("model_hidden_size", 4096)
    pre_k = parsed_kwargs.pop("k", "0.05")
    pre_k_is_float = '.' in pre_k
    k = max(1, int(float(pre_k) * len(_data_icl_examples)) if pre_k_is_float else int(pre_k))

    logger.info("Seed: %s", seed)
    logger.info("k=%d", k)

    env_args = [src_lang, trg_lang, file_data, file_data_icl_examples]

    ## load model
    logger.info("Loading model: %s", best_model_path)

    env_eval_dev_class = MTICLEvalEnv
    env_eval_dev = Monitor(env_eval_dev_class(*list(env_args), custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs), filename=None, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False})

    action_dim = env_eval_dev.action_dim
    state_dim_per_token = env_eval_dev.state_dim_per_token
    state_window_length = env_eval_dev.state_window_length
    skip_we = action_dim // state_dim_per_token
    #retrieve_embeddings_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, debug=True)[0] # Get only the result, not I or D
    retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=k, get_representations_instead_of_embeddings=False, observations=observations, debug=True)[0]
    net_arch = {
        "pi": [256, 256],
        "qf": [512, 256]
    }
    #actor_transformer_args_and_kwargs = {
    #    "d_model": 512,
    #    "nhead": 4,
    #    "dim_feedforward": 2048,
    #    "nlayers": 3,
    #    "max_seq_len": 16,
    #    "projection_in": model_hidden_size,
    #    "l2_norm": False,
    #    "str_id": "actor",
    #}
    #critic_transformer_args_and_kwargs = {
    #    "d_model": 512,
    #    "nhead": 4,
    #    "dim_feedforward": 2048,
    #    "nlayers": 3,
    #    "max_seq_len": 16,
    #    "projection_in": model_hidden_size,
    #    "str_id": "critic",
    #}
    #actor_use_transformer = True
    #critic_use_transformer = True
    policy_actor_kwargs = {
        #"actor_lr_schedule": lambda foo: actor_learning_rate, # callable
        "actor_lr_schedule": lambda foo: 100.0, # dummy callable
        "actor_layer_norm_input": True,
        "actor_layer_norm_before_activation": True,
        "actor_last_layer_init_uniform_value": 0.001,
        #"actor_dropout": True,
        #"actor_dropout_p": 0.1,
        #"actor_transformer": actor_use_transformer,
        #"actor_transformer_args_and_kwargs": actor_transformer_args_and_kwargs,
        "actor_mlp_l2_norm": True,
    }
    policy_critic_kwargs = {
        #"critic_lr_schedule": lambda foo: critic_learning_rate, # callable
        #"lr_schedule": InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, logger=logger, str_id="critic"), # callable
        "critic_layer_norm_input": True,
        "critic_layer_norm_before_activation": True,
        #"critic_last_layer_init_uniform_value": 0.001,
        #"critic_dropout": True,
        #"critic_dropout_p": 0.1,
        #"critic_transformer": critic_use_transformer,
        #"critic_transformer_args_and_kwargs": critic_transformer_args_and_kwargs,
        #"critic_first_actions_then_features": True if critic_use_transformer else False,
    }
    use_transformer = True
    dropout_p = 0.1

    if not use_transformer:
        features_extractor_class = FlattenExtractor
        #features_extractor_class = NoFlattenExtractor
        features_extractor_kwargs_actor = {}
        features_extractor_kwargs_critic = {}
    else:
        # the feature extractor only processes the state (not the action for the critic)
        features_extractor_class = TransformerExtractor
        transformer_common_kwargs = {
            "d_model": 512,
            "nhead": 4,
            "dim_feedforward": 2048,
            "nlayers": 3,
            "projection_in": state_dim_per_token,
            #"projection_out": action_dim,
            #"projection_out": 256,
            "projection_out": None, # let the MLP layers after the feature extractor handle the rest of the processing
            "activation": "relu",
            "bias": True,
            "norm_first": True,
            "initial_layer_norm": True,
            "initial_layer_norm_first": False,
            "embedding_dropout": 0.0, # it can increse the variance in the training
            "dropout_p": dropout_p, # we can disable dropout setting to 0.0 if needed
            "projection_out_dropout_p": 0.0,
            "max_seq_len": 8192, # the positional encoding is absolute and using this big value does not affect to the previous positions
            "l2_norm": False, # disable to let the model learn how the representation should be
            "skip_n_word_embeddings_from_observation": f"0:{skip_we}" if state_representation == "representation_per_token_with_features" else "0:0", # skip the first word embeddings, which corresponds to the source translation candidate representation for detecting overlap
            #"skip_n_word_embeddings_from_observation": "0:0", # TODO remove
            "expected_seq_len": ((state_window_length - 1) * state_dim_per_token) // state_dim_per_token if state_representation == "representation_per_token_with_features" else None,
            #"expected_seq_len": None, # TODO remove
        }
        features_extractor_kwargs_actor = {
            "str_id": "actor",
            **transformer_common_kwargs
        }
        features_extractor_kwargs_critic = {
            "str_id": "critic",
            **transformer_common_kwargs
        }

    model_class = TD3
    model = model_class.load(
        best_model_path,
        learning_rate=lambda foo: 100.0, # dummy callable
        lr_schedule=lambda foo: 100.0, # dummy callable
        policy_kwargs={
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_dev,
            "callback_retrieve_knn_training": None,
            "k": k,
            #"add_all_knn_to_batch": True, # Faster
            "add_all_knn_to_batch": False,
            "apply_rws_inference": False,
            **policy_actor_kwargs,
            **policy_critic_kwargs,
            #"squash_output": True,
            "squash_output": False,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs_actor": features_extractor_kwargs_actor,
            "features_extractor_kwargs_critic": features_extractor_kwargs_critic,
        },
    )

    ## dev: evaluate and report result
    logger.info("Evaluating dev")

    #mean_reward, std_reward = evaluate_policy(model, env_eval_dev.unwrapped, n_eval_episodes=len(data_to_be_translated))
    episode_rewards, episode_lengths = evaluate_policy(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated), return_episode_rewards=True)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"Mean reward dev: {mean_reward} +/- {std_reward} (length: {mean_length} +/- {std_length})")
