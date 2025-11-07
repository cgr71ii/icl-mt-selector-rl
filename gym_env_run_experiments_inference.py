
import sys
import logging

import utils
from gym_env_v1 import MTICLEnv
from gym_env_v1_eval import MTICLEvalEnv

from gym_env_run_experiments import InverseSqrtWithWarmUpLRSchedule

import gymnasium as gym
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor, NoFlattenExtractor

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

    for _file_data, data_to_be_translated in ((file_data, data_to_be_translated),):
        with open(_file_data, "rt") as fd:
            for line in fd:
                data_to_be_translated.append(line.rstrip("\r\n"))

    # parse args
    parsed_kwargs["device"] = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    parsed_kwargs["max_icl_examples"] = parsed_kwargs.get("max_icl_examples", 4)
    parsed_kwargs["max_data_entries"] = parsed_kwargs.get("max_data_entries", -1)
    parsed_kwargs["max_data_icl_examples_entries"] = parsed_kwargs.get("max_data_icl_examples_entries", -1)
    parsed_kwargs["state_representation"] = parsed_kwargs.get("state_representation", "model_single_representation")
    parsed_kwargs["eval_strategy"] = parsed_kwargs.get("eval_strategy", "comet-22-da")
    parsed_kwargs["dimensionality_reduction_factor_state_and_action"] = int(parsed_kwargs.get("dimensionality_reduction_factor_state_and_action", 256))
    parsed_kwargs["repeat_translation_candidates"] = parsed_kwargs.get("repeat_translation_candidates", False)
    parsed_kwargs["knn_api_retrieve"] = parsed_kwargs.get("knn_api_retrieve", None)
    parsed_kwargs["knn_api_insert"] = parsed_kwargs.get("knn_api_insert", None)
    parsed_kwargs["dimensionality_reduction_type"] = parsed_kwargs.get("dimensionality_reduction_type", "iterative_nonoverlapping_average")
    parsed_kwargs["max_distance_threshold"] = parsed_kwargs.get("max_distance_threshold", "inf")
    parsed_kwargs["gym_logger_level"] = parsed_kwargs.get("gym_logger_level", gym.logger.DEBUG)

    # custom
    k = 20
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.rl_experiments"), level=logging.DEBUG)
    #max_data_icl_examples_entries = 100 # TODO remove
    #max_data_entries = 1 # TODO remove
    #parsed_kwargs["max_data_entries"] = max_data_entries
    #parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    #data_to_be_translated = data_to_be_translated[:max_data_entries if max_data_entries > 0 else None]
    max_distance_threshold = parsed_kwargs["max_distance_threshold"]
    model_hidden_size = parsed_kwargs.get("model_hidden_size", 4096)

    #assert parsed_kwargs["knn_api_retrieve"] is not None
    #assert parsed_kwargs["knn_api_insert"] is not None

    parsed_kwargs_training_dummy = {}
    # TODO use following code?
    #parsed_kwargs_training_dummy["knn_api_retrieve"] = parsed_kwargs["knn_api_retrieve"]
    #parsed_kwargs_training_dummy["knn_api_insert"] = parsed_kwargs["knn_api_insert"]
    #parsed_kwargs_training_dummy["max_distance_threshold"] = parsed_kwargs["max_distance_threshold"]

    #del parsed_kwargs["knn_api_retrieve"]
    #del parsed_kwargs["knn_api_insert"]
    #del parsed_kwargs["max_distance_threshold"]

    env_args = [src_lang, trg_lang, file_data, file_data_icl_examples]

    ## training environment
    #env_class = MTICLEnv
    #env_training_dummy = env_class(*list(env_args), **dict({"custom_env_id": "training_dummy", **parsed_kwargs_training_dummy, **parsed_kwargs}))

    #env_training_dummy._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": True}) # env_training_dummy.get_closest_neighbors_urls() is available

    ## dev: load model
    logger.info("Loading best model (dev): %s", best_model_path)

    env_eval_dev_class = MTICLEvalEnv
    env_eval_dev = Monitor(env_eval_dev_class(*list(env_args), custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs), filename=None, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False})

    #retrieve_embeddings_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, add_saturated_action=False, max_distance_threshold=max_distance_threshold, debug=True)[0] # Get only the result, not I or D
    retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, debug=True)[0]
    net_arch = {
        "pi": [400, 300],
        #"qf": [400, 300]
        "qf": [2000, 2000, 2000, 2000]
    }
    actor_learning_rate = 1e-4
    actor_transformer_args_and_kwargs = {
        "d_model": 512,
        "nhead": 4,
        "dim_feedforward": 2048,
        "nlayers": 3,
        "max_seq_len": 16,
        "projection_in": model_hidden_size,
        "l2_norm": False,
        "str_id": "actor",
    }
    critic_transformer_args_and_kwargs = {
        "d_model": 512,
        "nhead": 4,
        "dim_feedforward": 2048,
        "nlayers": 3,
        "max_seq_len": 16,
        "projection_in": model_hidden_size,
        "str_id": "critic",
    }
    actor_use_transformer = True
    critic_use_transformer = True
    policy_actor_kwargs = {
        #"actor_lr_schedule": lambda foo: actor_learning_rate, # callable
        "actor_lr_schedule": lambda foo: 100.0, # dummy callable
        "actor_layer_norm_input": True,
        "actor_layer_norm_before_activation": True,
        "actor_last_layer_init_uniform_value": 0.001,
        "actor_dropout": True,
        "actor_dropout_p": 0.1,
        "actor_transformer": actor_use_transformer,
        "actor_transformer_args_and_kwargs": actor_transformer_args_and_kwargs,
    }
    policy_critic_kwargs = {
        #"critic_lr_schedule": lambda foo: critic_learning_rate, # callable
        #"lr_schedule": InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, logger=logger, str_id="critic"), # callable
        "critic_layer_norm_input": True,
        "critic_layer_norm_before_activation": True,
        "critic_last_layer_init_uniform_value": 0.001,
        "critic_dropout": True,
        "critic_dropout_p": 0.1,
        "critic_transformer": critic_use_transformer,
        "critic_transformer_args_and_kwargs": critic_transformer_args_and_kwargs,
    }

    assert (actor_use_transformer and critic_use_transformer) or (not actor_use_transformer and not critic_use_transformer), "Supported: both enabled or disabled"

    features_extractor_class = NoFlattenExtractor if actor_use_transformer or critic_use_transformer else FlattenExtractor
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
            "add_all_knn_to_batch": True, # Faster
            "apply_rws_inference": False,
            **policy_actor_kwargs,
            **policy_critic_kwargs,
            "squash_output": True,
            "features_extractor_class": features_extractor_class,
        },
    )

    ## dev: evaluate and report result
    logger.info("Evaluating dev")

    #mean_reward, std_reward = evaluate_policy(model, env_eval_dev.unwrapped, n_eval_episodes=len(data_to_be_translated))
    mean_reward, std_reward = evaluate_policy(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated)) # TODO remove? use .unwrapped?

    print(f"Mean reward dev: {mean_reward} +/- {std_reward}")
