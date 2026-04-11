
import os
import sys
import random
import logging
from datetime import datetime

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

import utils
from gym_env_v1 import MTICLEnv
from gym_env_v1_eval import MTICLEvalEnv
#from gym_env_v1_eval_single_episode import MTICLEvalSingleEpisodeEnv
from gym_env_run_experiments import LinearDecayScheduler, DelayedEvalCallback, LinearWithWarmUpLRSchedule, NormalActionNoiseWithClip
from gym_env_run_experiments import make_env, store_model

import gymnasium as gym
from stable_baselines3 import DDPG, TD3
import stable_baselines3.common.callbacks as sb3_cb
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor, NoFlattenExtractor, TransformerExtractor, NFeaturesExtractor, NFeaturesExtractorWithTimeStepEmbeddings
from stable_baselines3.common.buffers import NStepReplayBuffer, MonteCarloReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic, ContinuousCriticTower
import numpy as np
import torch

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
    min_conf_debug = False
    #min_conf_debug = True
    num_envs = max(1, int(parsed_kwargs.pop("num_envs", 8)))
    device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    max_icl_examples = int(parsed_kwargs.get("max_icl_examples", 5))
    apply_rws_inference = parsed_kwargs.get("apply_rws_inference", False)
    wolpertinger_disable_actor = bool(int(parsed_kwargs.pop("wolpertinger_disable_actor", 0)))
    pre_k = parsed_kwargs.pop("k", "0.15")
    update_to_data_ratio = float(parsed_kwargs.pop("update_to_data_ratio", 1.0)) # UTD (check, for example, "Dropout Q-Functions for Doubly Efficient Reinforcement Learning" paper)
    disable_eval = bool(int(parsed_kwargs.pop("disable_eval", 0)))
    store_model_on_eval = bool(int(parsed_kwargs.pop("store_model_on_eval", 0)))

    if min_conf_debug:
        logger.warning("min_conf_debug is set to True, which overrides some parameters to make the training faster. DEBUG purpose only!")

    if min_conf_debug:
        num_envs = 5 # TODO remove
        #update_to_data_ratio = 0.0 # TODO remove
        disable_eval = False # TODO remove
        store_model_on_eval = False # TODO remove

    if store_model_on_eval:
        logger.info("Model will be stored on each evaluation")

    if wolpertinger_disable_actor and pre_k != "1.0":
        logger.warning("wolpertinger_disable_actor is set to True, and k should be set to 1.0 (instead of %s)", pre_k)

    logger.debug("wolpertinger_disable_actor: %s", wolpertinger_disable_actor)

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
    assert not apply_rws_inference, "Biased learning"

    # set defaults in case they are not provided
    max_data_entries = int(parsed_kwargs.get("max_data_entries", -1)) # load all data (default value)
    max_data_icl_examples_entries = int(parsed_kwargs.get("max_data_icl_examples_entries", -1)) # load all data (default value)

    if min_conf_debug:
        #max_data_entries = 1 # TODO remove
        max_data_entries = 10 # TODO remove
        #max_data_entries = 50 # TODO remove
        max_data_icl_examples_entries = 10 # TODO remove
        pass

    state_representation = parsed_kwargs.get("state_representation", "representation_per_token_with_features")
    parsed_kwargs["device"] = device
    parsed_kwargs["max_icl_examples"] = max_icl_examples
    parsed_kwargs["max_data_entries"] = max_data_entries
    parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    parsed_kwargs["state_representation"] = state_representation
    parsed_kwargs["eval_strategy_training"] = parsed_kwargs.get("eval_strategy_training", "chrf2")
    parsed_kwargs["eval_strategy_eval"] = parsed_kwargs.get("eval_strategy_eval", "chrf2")
    parsed_kwargs["repeat_translation_candidates"] = parsed_kwargs.get("repeat_translation_candidates", True)
    parsed_kwargs["repeat_translation_candidates_times"] = int(parsed_kwargs.get("repeat_translation_candidates_times", 4))
    parsed_kwargs["knn_always_add_eos_action"] = parsed_kwargs.get("knn_always_add_eos_action", True)
    parsed_kwargs["enable_eos_action"] = parsed_kwargs.get("enable_eos_action", False)
    parsed_kwargs["state_window_length"] = int(parsed_kwargs.get("state_window_length", 1024)) + 3
    parsed_kwargs["action_representation"] = parsed_kwargs.get("action_representation", "src_embedding:SONAR")
    parsed_kwargs["model_hidden_size_action_src_sentence"] = parsed_kwargs.get("model_hidden_size_action_src_sentence", 1024)
    parsed_kwargs["actions_without_replacement"] = parsed_kwargs.get("actions_without_replacement", False) # allow/disallow selecting the same ICL example more than once in the same trajectory
    parsed_kwargs["knn_distance_ip"] = bool(int(parsed_kwargs.get("knn_distance_ip", True)))
    parsed_kwargs["current_icl_examples_prepend"] = bool(int(parsed_kwargs.get("current_icl_examples_prepend", False)))
    parsed_kwargs["model_hidden_size"] = parsed_kwargs.get("model_hidden_size", 1536)
    data_to_be_translated_training = data_to_be_translated_training[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_dev = data_to_be_translated_dev[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_test = data_to_be_translated_test[:max_data_entries if max_data_entries > 0 else None]
    data_icl_examples_training = data_icl_examples_training[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_dev = data_icl_examples_dev[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_test = data_icl_examples_test[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    knn_distance_ip = parsed_kwargs["knn_distance_ip"]
    process_token_time_step = bool(int(parsed_kwargs.get("process_token_time_step", True)))
    parsed_kwargs["process_token_time_step"] = process_token_time_step

    if state_representation == "representation_mean_plus_last_75_perc_layer_and_relative_diff":
        parsed_kwargs["model_hidden_size"] *= 2

    # Some values
    pre_k_is_float = '.' in pre_k
    k_training = min(max(1, int(float(pre_k) * len(data_icl_examples_training)) if pre_k_is_float else int(pre_k)), len(data_icl_examples_training))
    k_dev = min(max(1, int(float(pre_k) * len(data_icl_examples_dev)) if pre_k_is_float else int(pre_k)), len(data_icl_examples_dev))
    k_test = min(max(1, int(float(pre_k) * len(data_icl_examples_test)) if pre_k_is_float else int(pre_k)), len(data_icl_examples_test))
    #return_all_neighbors_training = k_training == len(data_icl_examples_training)
    #return_all_neighbors_dev = k_dev == len(data_icl_examples_dev)
    #return_all_neighbors_test = k_test == len(data_icl_examples_test)
    return_all_neighbors_training = return_all_neighbors_dev = return_all_neighbors_test = False # With this option duplicates can't be removed (easily...)
    #return_all_neighbors_training = return_all_neighbors_dev = return_all_neighbors_test = True # TODO DEBUG REMOVE

    assert isinstance(k_training, int) # other parameters use k assuming integer instead of float
    assert isinstance(k_dev, int)
    assert isinstance(k_test, int)

    logger.info("k=(%d, %d, %d)", k_training, k_dev, k_test)

    # Other kwargs
    parsed_kwargs_training = {}
    #parsed_kwargs_training["initial_time_sleep"] = num_envs * 2 # sleep to synchronize all environments
    parsed_kwargs_training_dummy = {}

    logger.info("parsed_kwargs: %s", parsed_kwargs)

    # Other values
    filename_time = datetime.now().strftime("%Y%m%d_%H%M")
    #save_freq = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs) # steps
    save_freq = 1e1000 # disabled
    #eval_freq = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs) # steps (approx. once per epoch)
    eval_freq = 10000 # steps
    save_path = f"./rl_models_{filename_time}/"
    name_prefix = f"rl_{filename_time}"
    #monitor_filename = f"{save_path}{name_prefix}_eval.log"
    monitor_filename = None # pickle serialization doesn't allow to have an opened file descriptor (EvalCallback)
    max_episodes_epochs = 100000 # repeat N times (patience-driven environment, so this value might not be used at all)
    max_episodes = len(data_to_be_translated_training) * max_episodes_epochs
    patience = -1 # early stopping patience (number of evals with no improvement; disabled if < 0)
    enable_eval = not disable_eval

    if min_conf_debug:
        #eval_freq = 500 # TODO remove
        #eval_freq = 1000 # steps
        #eval_freq = 2000 # steps
        #eval_freq = 5000 # steps
        #eval_freq = 40 # TODO remove
        eval_freq = 200 # TODO remove
        patience = 6 #  TODO remove
        #patience = 3 # TODO remove?

    if not enable_eval:
        logger.warning("Evaluation (dev set) disabled: no best model will be available, only last model")
    else:
        assert patience >= 0, "Expected patience to be non-negative, but got %d" % patience

    if patience < 0:
        logger.info("Early stopping disabled (patience < 0)")
    else:
        logger.info("Early stopping enabled (patience: %d evals with no improvement)", patience)

    assert not os.path.exists(save_path), f"Save path already exists: {save_path}"

    os.makedirs(save_path, exist_ok=False)

    logger.info("Save path: %s", save_path)

    # Environment
    env_class = MTICLEnv
    env_eval_dev_class = MTICLEvalEnv
    env_eval_test_class = MTICLEvalEnv
    #vec_env_class = DummyVecEnv # debug
    vec_env_class = SubprocVecEnv
    vec_env_kwargs = {"start_method": "forkserver"} if vec_env_class is SubprocVecEnv else {}
    #batch_size = 256
    batch_size = max(1, int(parsed_kwargs.pop("rl_batch_size", 256)))
    gamma = 1.0
    #gamma = 0.99
    #replay_buffer_size = 1000000
    #replay_buffer_size = 50000 # given 5 ICL examples and training set of 1000 sentences, this allows to store all transitions of 10 epochs (1000 * 5 * 10 = 50000)
    #replay_buffer_size = 10000
    #critic_learning_rate = 1e-3
    #actor_learning_rate = 1e-4
    #critic_learning_rate = 5e-4
    #actor_learning_rate = 5e-4
    #critic_learning_rate = 1e-3
    #actor_learning_rate = 1e-3
    #critic_learning_rate = 1e-4
    #actor_learning_rate = 1e-5
    #critic_learning_rate = 5e-4
    #critic_learning_rate = 5e-5
    critic_learning_rate = 1e-4
    actor_learning_rate = 1e-4
    #max_steps = 1e100 # fake value due to callback StopTrainingOnMaxEpisodes
#    max_steps_training = 10000 # steps while training
    #max_steps_training = 20000 # steps while training
    #max_steps_training = 50000 # steps while training
    #max_steps_training = 100000 # steps while training
    max_steps_training = 200000
    #max_steps_training = 10000000 # steps while training # TODO remove?
    max_steps_training += num_envs + 1 # to be sure that the last model is stored after training, given that eval_freq is adjusted by num_envs
    #init_training_steps = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs)
    init_training_steps = int(max_steps_training * 0.1 + 0.5)

    if min_conf_debug:
        batch_size = 16 # TODO remove
        init_training_steps = 50 # TODO remove
        #init_training_steps = 100 # TODO remove

    max_steps = max_steps_training + init_training_steps
    replay_buffer_size = int(max_steps * 0.5 + 0.5) # small value to avoid remove old transitions and avoid averaging Q-values over "bad" actions. If monte carlo updates are used, and old transitions are updated to the best Q-value found, then this value can be increased
    actor_mlp_l2_norm = bool(int(parsed_kwargs.pop("actor_mlp_l2_norm", 1)))
    add_all_knn_to_batch = max(1, int(parsed_kwargs.pop("add_all_knn_to_batch", 1)))
    use_transformer = bool(int(parsed_kwargs.pop("use_transformer", 0)))
    total_evaluations = max_steps_training // eval_freq

    logger.info("Evaluation frequency (steps, adjusted by number of parallel environments): %d // %d = %d (total evaluations: %d)", eval_freq, num_envs, max(1, eval_freq // num_envs), total_evaluations)
    logger.info("Init. steps collecting rollouts without training: %d", init_training_steps)
    logger.info("Max. steps (no_training, training, total): (%d, %d, %d)", init_training_steps, max_steps_training, max_steps)
    logger.info("actor_mlp_l2_norm: %s", actor_mlp_l2_norm)
    logger.info("Actual batch_size for running the critic with the observations and the kNN elements: %d * min(%d, k) = %d", batch_size, add_all_knn_to_batch, batch_size * add_all_knn_to_batch)
    logger.info("Replay buffer size: %d", replay_buffer_size)

    eval_freq = max(1, eval_freq // num_envs)

    if num_envs > 1:
        logger.info("Be aware that the environment will be executed %d time steps, but %d // %d = %d per environment (%d // %d = %d init. training steps) instance due to the number of parallel envinronments (%d training steps, where %d different batches will be used for training from the replay buffer)",
                    max_steps, max_steps, num_envs, max_steps // num_envs,
                    init_training_steps, num_envs, init_training_steps // num_envs,
                    (max_steps - init_training_steps) // num_envs, max_steps - init_training_steps)

    #env = env_class(src_lang, trg_lang, file_data_training, file_data_icl_examples_training, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)
    env_args = [src_lang_training, trg_lang_training, file_data_training, file_data_icl_examples_training]
    env_kwargs = {"gym_logger_level": gym.logger.DEBUG, **parsed_kwargs}
    env_training_dummy = env_class(*list(env_args), **dict({"custom_env_id": "training_dummy", **env_kwargs, **parsed_kwargs_training_dummy}))

    env_training_dummy._init_load_data_and_populate_knn_pool(options={}) # env_training_dummy.get_closest_neighbors_urls() is available

    parsed_kwargs_training["initial_sample_list_actions"] = [(k, env_training_dummy.str2representation[k]) for k in env_training_dummy.str2representation_valid_actions_k] # initial random action sampling
    env = vec_env_class([make_env(rank, env_class, list(env_args), dict({"custom_env_id": str(rank), **env_kwargs, **parsed_kwargs_training}), seed=env_seeds[rank]) for rank in range(num_envs)], **vec_env_kwargs)
    env_eval_dev = Monitor(env_eval_dev_class(src_lang_dev, trg_lang_dev, file_data_dev, file_data_icl_examples_dev, gym_logger_level=gym.logger.INFO, custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs), filename=monitor_filename, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={}) # env_eval_dev.get_closest_neighbors_urls() is available

    retrieve_embeddings_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=k_training, get_representations_instead_of_embeddings=False, observations=observations, debug=False, return_all_neighbors=return_all_neighbors_training)[0] # Get only the result, not I or D
    retrieve_embeddings_training_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=k_training, get_representations_instead_of_embeddings=False, observations=observations, debug=False, return_all_neighbors=return_all_neighbors_training)[0] # Get only the result, not I or D
    retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=k_dev, get_representations_instead_of_embeddings=False, observations=observations, debug=False, return_all_neighbors=return_all_neighbors_dev)[0]
    n_actions = env.unwrapped.action_space.shape[-1]
    #action_noise_sigma = 0.05
    action_noise_sigma = 0.0215
    action_noise_clip = 2.5 * action_noise_sigma
    #action_noise_sigma = 0.5
    #action_noise_clip = 1.0
    action_noise = NormalActionNoiseWithClip(mean=np.zeros(n_actions), sigma=action_noise_sigma * np.ones(n_actions), clip_value=action_noise_clip)
    #action_noise = None # TODO remove
    action_dim = env_training_dummy.action_dim
    state_dim_per_token = env_training_dummy.state_dim_per_token
    state_window_length = env_training_dummy.state_window_length
    state_dim_per_token_time_step = env_training_dummy.state_dim_per_token_time_step
    skip_we = action_dim // state_dim_per_token
    callbacks = []
    #model_class = DDPG
    model_class = TD3
    td3_args = {
        "policy_delay": 2,
        #"target_policy_noise": 1e-3, # small values for better generalization and robustness
        #"target_noise_clip": 2e-3, # small values for better generalization and robustness
        "target_policy_noise": 0.0, # TODO remove
        "target_noise_clip": 0.0, # TODO remove
    } if model_class is TD3 else {}
    #use_transformer = True
    #dropout_p = 0.1 # TODO enable?
    #dropout_p = 0.0
    actor_dropout_p = 0.0
    #critic_dropout_p = 0.1
    #critic_dropout_p = 0.01 # DroQ paper if value greater than 0
    critic_dropout_p = 0.0 # TODO enable?
    #exploration_rate_steps_percentage = 0.5
    exploration_rate_steps_percentage = 0.1
    exploration_rate_steps = int((max_steps - init_training_steps) / num_envs * exploration_rate_steps_percentage + 0.5)
    #exploration_rate_steps = 50000
    #exploration_rate_steps = 50 # TODO remove
    exploration_rate_steps_percentage_of_training = exploration_rate_steps / ((max_steps - init_training_steps) / num_envs)
    exploration_rate_initial = 1.0
    #exploration_rate_initial = 0.1 # TODO remove
    exploration_rate_last = 0.05
    exploration_rate = LinearDecayScheduler(exploration_rate_initial, exploration_rate_last, exploration_rate_steps, logger, "epsilon-greedy exploration")
    #exploration_rate = 0.1
    #wolpertinger_target_policy_actor_noise = 0.1
    #wolpertinger_target_policy_actor_noise = 0.02
    wolpertinger_target_policy_actor_noise = 0.0212
    wolpertinger_target_actor_noise_clip = 2.5 * wolpertinger_target_policy_actor_noise
    share_features_extractor = False
    #share_features_extractor = True
    #train_freq_steps = 1
    train_freq_steps = max_icl_examples
    #gradient_steps = -1 if num_envs > 1 else 1
    #gradient_steps = num_envs # "do as many gradient steps as steps done in the environment during the rollout", also recommended for off-policy algorithms with num_envs > 1
    gradient_steps = max(int(update_to_data_ratio * train_freq_steps * num_envs), 1) # times data is sampled from the replay buffer and then used for training

    if state_representation == "representation_per_token_with_features":
        n_features = state_dim_per_token * (state_window_length - 1) # -1 due to the action representation which we skip
    elif state_representation in ("representation_last_token_current_and_relative_diff", "representation_mean_plus_last_75_perc_layer_and_relative_diff"):
        n_features = state_dim_per_token * 2 + (state_dim_per_token_time_step if process_token_time_step else 0)
    elif state_representation == "representation_mean_75_perc_layer":
        n_features = state_dim_per_token + (state_dim_per_token_time_step if process_token_time_step else 0)
    else:
        n_features = 0

    if use_transformer and not wolpertinger_disable_actor:
        assert not share_features_extractor, "share_features_extractor is not supported when using transformers (different configuration for actor and critic)"

    if wolpertinger_disable_actor and share_features_extractor:
        share_features_extractor = False

        logger.info("Disabling share_features_extractor because wolpertinger_disable_actor is set to True")

    logger.info("Exploration rate steps: %d (%s %% of the total training steps): from %s to %s", exploration_rate_steps, exploration_rate_steps_percentage_of_training * 100, exploration_rate_initial, exploration_rate_last)
    logger.info("Gradient steps (UTD: %s; train_freq_steps: %s): %d", update_to_data_ratio, train_freq_steps, gradient_steps)

    min_actor_learning_rate = actor_learning_rate / 10
    min_critic_learning_rate = critic_learning_rate / 10
    transformer_d_model = 128
    #step_size = 500
    #step_size = max(int(step_size / (train_freq_steps * num_envs) + 0.5), 1) # We divide by (train_freq_steps * num_envs) to apply a similar number of times the learning rate regardless the number of environments and how much the training is delayed
    #warmup_steps = step_size * 2

    #logger.info("Step size and warmup steps: %d %d", step_size, warmup_steps)

    if not use_transformer:
        net_arch = {
            #"pi": [256, 256],
            "pi": [512, 512],
            #"qf": [512, 256]
            #"qf": [512, 512]
            #"qf": [1024, 512]
            #"qf": [400, 300]
            #"qf": [512, 256, 128]
            #"qf": [512, 128, 32]
            "qf": [512, 256, 128]
        } # "pi" is actor and "qf" the critic (ignored if transformer is used)

        logger.info("Transformer disabled: using MLP: %s", net_arch)

        warmup_steps = 0
        #actor_learning_rate = 1e-3
        #critic_learning_rate = 1e-3
        actor_learning_rate = 1e-4
        critic_learning_rate = 1e-4
        min_actor_learning_rate = actor_learning_rate
        min_critic_learning_rate = critic_learning_rate
    else:
        net_arch = {
            "pi": [512, 512],
            "qf": [128]
        } # "pi" is actor and "qf" the critic (ignored if transformer is used)

        logger.info("Transformer enabled: using transformer+MLP: ... + %s", net_arch)

        #warmup_steps = 200
        warmup_steps = 1000
        #warmup_steps = 2000
        #warmup_steps = 5000
        actor_learning_rate = 1e-5
        critic_learning_rate = 1e-4
        min_actor_learning_rate = actor_learning_rate
        min_critic_learning_rate = critic_learning_rate

    logger.info("Warmup steps: %d", warmup_steps)

    #if patience >= 0:
    #    actor_lr_schedule = InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=actor_learning_rate, logger=logger, str_id="actor") # callable
    #    critic_lr_schedule = InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, logger=logger, str_id="critic") # callable
    #else:
    #    total_steps = (max_steps - init_training_steps) // num_envs
    #    actor_lr_schedule = LinearWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=actor_learning_rate, total_steps=total_steps, logger=logger, min_lr=min_actor_learning_rate, str_id="actor")
    #    critic_lr_schedule = LinearWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, total_steps=total_steps, logger=logger, min_lr=min_critic_learning_rate, str_id="critic")

    clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
    min_actor_learning_rate = actor_learning_rate / 10
    min_critic_learning_rate = critic_learning_rate / 10
    #actor_lr_schedule = CyclicWithWarmUpLRSchedule(base_lr=min_actor_learning_rate, max_lr=actor_learning_rate, step_size=step_size, scale_fn=clr_fn, scale_mode="cycle", warmup_steps=warmup_steps, logger=logger, str_id="actor")
    #critic_lr_schedule = CyclicWithWarmUpLRSchedule(base_lr=min_critic_learning_rate, max_lr=critic_learning_rate, step_size=step_size, scale_fn=clr_fn, scale_mode="cycle", warmup_steps=warmup_steps, logger=logger, str_id="critic")
    total_steps = max(int((max_steps - init_training_steps) / (train_freq_steps * num_envs) + 0.5), 1)
    actor_lr_schedule = LinearWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=actor_learning_rate, total_steps=total_steps, logger=logger, min_lr=min_actor_learning_rate, str_id="actor")
    critic_lr_schedule = LinearWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, total_steps=total_steps, logger=logger, min_lr=min_critic_learning_rate, str_id="critic")
    policy_actor_kwargs = {
        #"squash_output": True,
        "squash_output": not actor_mlp_l2_norm, # actor l2-norm
        #"actor_lr_schedule": lambda foo: actor_learning_rate, # callable
        "actor_lr_schedule": actor_lr_schedule,
        "actor_layer_norm_input": True,
        "actor_layer_norm_before_activation": True,
        "actor_last_layer_init_uniform_value": 0.001,
        #"actor_dropout": True,
        "actor_dropout_p": actor_dropout_p,
        "actor_mlp_l2_norm": actor_mlp_l2_norm,
    }
    policy_critic_kwargs = {
        #"critic_lr_schedule": lambda foo: critic_learning_rate, # callable
        #"lr_schedule": critic_lr_schedule,
        "critic_layer_norm_input": True,
        "critic_layer_norm_before_activation": True,
        #"critic_last_layer_init_uniform_value": 0.001,
        "critic_dropout_p": critic_dropout_p, # DroQ paper
        #"critic_first_actions_then_features": True if use_transformer else False,
        "critic_class": ContinuousCriticTower,
    }
    n_critics = 2 # TD3 uses 2 critics to mitigate overestimation bias
    #n_critics = 1 # TODO remove

    if not use_transformer:
        if n_features <= 0:
            features_extractor_class = FlattenExtractor
            #features_extractor_class = NoFlattenExtractor
            features_extractor_kwargs_actor = {}
            features_extractor_kwargs_critic = {}
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

            features_extractor_kwargs_actor = {
                "n": n,
                "skip_n": skip_n,
                "state_dim_per_token": state_dim_per_token,
                "check_zeros": True if state_representation == "representation_per_token_with_features" else False,
                "step_embeddings": max_icl_examples + 1, # add embeddings for each time step (+1 to avoid error in the model forward for computing next_actions, although the result will be discarded)
                "step_embeddings_dim": step_embeddings_dim,
            }
            features_extractor_kwargs_critic = dict(features_extractor_kwargs_actor)
    else:
        assert state_representation == "representation_per_token_with_features", "Some values should be revised for the transformer such as 'projection_in'"

        # the feature extractor only processes the state (not the action for the critic)
        features_extractor_class = TransformerExtractor
        transformer_common_kwargs = {
            #"d_model": 512,
            #"d_model": 128,
            "d_model": transformer_d_model,
            "nhead": 4,
            #"dim_feedforward": 2048,
            #"dim_feedforward": 512,
            "dim_feedforward": transformer_d_model * 4,
            #"nlayers": 3,
            "nlayers": 2,
            "projection_in": state_dim_per_token,
            #"projection_out": action_dim,
            #"projection_out": 512,
            "projection_out": None, # let the MLP layers after the feature extractor handle the rest of the processing
            #"projection_out": 64,
            #"activation": "relu",
            "activation": "gelu",
            "bias": True,
            "norm_first": True,
            "initial_layer_norm": True,
            #"initial_layer_norm_first": False,
            "initial_layer_norm_first": True,
            "embedding_dropout": critic_dropout_p, # it can increse the variance in the training
            "dropout_p": critic_dropout_p, # we can disable dropout setting to 0.0 if needed
            "projection_out_dropout_p": critic_dropout_p,
            "max_seq_len": 8192, # the positional encoding is absolute and using this big value does not affect to the previous positions
            #"l2_norm": False,
            #"skip_n_word_embeddings_from_observation": f"0:{skip_we}" if state_representation == "representation_per_token_with_features" else "0:0", # skip the first word embeddings, which corresponds to the source translation candidate representation for detecting overlap
            "skip_n_word_embeddings_from_observation": f"0:{skip_we + 2}" if state_representation == "representation_per_token_with_features" else "0:0", # skip the first word embeddings, which corresponds to the source translation candidate representation for detecting overlap (+2 to ignore the custom position tokens)
            #"skip_n_word_embeddings_from_observation": "0:0", # TODO remove
            #"expected_seq_len": ((state_window_length - 1) * state_dim_per_token) // state_dim_per_token if state_representation == "representation_per_token_with_features" else None,
            "expected_seq_len": ((state_window_length - 1) * state_dim_per_token) // state_dim_per_token - 2 if state_representation == "representation_per_token_with_features" else None,
            #"expected_seq_len": None, # TODO remove
            "last_layer_norm": False,
            "last_linear_layer": True,
            "remove_first_column_of_zeros": True if state_representation == "representation_per_token_with_features" else False,
            "step_embeddings": max_icl_examples + 1, # add embeddings for each time step (+1 to avoid error in the model forward for computing next_actions, although the result will be discarded)
            "check_zeros": True if state_representation == "representation_per_token_with_features" else False,
        }

        features_extractor_kwargs_actor = {
            "str_id": "actor",
            "l2_norm": True if knn_distance_ip else False,
            "mean_pooling": True,
            **transformer_common_kwargs
        }
        features_extractor_kwargs_critic = {
            "str_id": "critic",
            "l2_norm": False,
            #"mean_pooling": False,
            "mean_pooling": True, # TODO remove?
            **transformer_common_kwargs
        }

    #critic_action_linear_layer_projection = 256
    #critic_features_linear_layer_projection = 256
    #critic_action_linear_layer_projection = 512 # TODO enable? with Tower architecture should not be necessary
    #critic_features_linear_layer_projection = 512 # TODO enable? with Tower architecture should not be necessary
    #critic_action_layer_norm_input = True # TODO enable? with Tower architecture should not be necessary
    #critic_features_layer_norm_input = True # TODO enable? with Tower architecture should not be necessary
    critic_action_linear_layer_projection = 0
    critic_features_linear_layer_projection = 0
    critic_action_layer_norm_input = False
    critic_features_layer_norm_input = False
    model = model_class(
        "WolpertingerPolicy",
        env,
        verbose=1,
        seed=seed,
        batch_size=batch_size,
        learning_starts=init_training_steps,
        #learning_rate=critic_learning_rate,
        learning_rate=critic_lr_schedule,
        policy_kwargs={
            "n_critics": n_critics,
            # Optimizer
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {
                "eps": 1e-5, # smaller value than default for Adam optimizer (found in common/policies.py), but I do not understand why
                "weight_decay": 1e-2,
            },
            # Other
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_training,
            "callback_retrieve_knn_training": retrieve_embeddings_training_training,
            "k": k_training,
            #"add_all_knn_to_batch": True, # Faster
            "add_all_knn_to_batch": add_all_knn_to_batch,
            "apply_rws_inference": apply_rws_inference,
            "exploration_rate": exploration_rate,
            "share_features_extractor": share_features_extractor,
            **policy_actor_kwargs,
            **policy_critic_kwargs,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs_actor": features_extractor_kwargs_actor,
            "features_extractor_kwargs_critic": features_extractor_kwargs_critic,
            "wolpertinger_disable_actor": wolpertinger_disable_actor,
            "critic_action_layer_norm_input": critic_action_layer_norm_input,
            "critic_features_layer_norm_input": critic_features_layer_norm_input,
            "critic_action_layer_dropout": 0.0,
            "critic_action_linear_layer_projection": critic_action_linear_layer_projection,
            "critic_features_linear_layer_projection": critic_features_linear_layer_projection,
            "activation_fn": torch.nn.GELU,
        },
        gamma=gamma,
        device=device,
        gradient_steps=gradient_steps,
        #train_freq=(1, "step"),
        #train_freq=(1, "episode"), # Not supported "episode" and num_envs > 1
        train_freq=(train_freq_steps, "step"), # sparse reward environment: we only have reward != 0 at the end of the episode
        buffer_size=replay_buffer_size,
        #replay_buffer_kwargs={} if not use_transformer else {"process_time_steps": True},
        #replay_buffer_class=NStepReplayBuffer,
        #replay_buffer_class=MonteCarloReplayBuffer,
        #replay_buffer_kwargs={"process_time_steps": True, "n_steps": max_icl_examples, "gamma": gamma},
        #replay_buffer_kwargs={"process_time_steps": True, "episodes_length": max_icl_examples, "gamma": gamma, "update_old_q_values_with_max_found": True, "epsilon": 1e-7},
        #replay_buffer_kwargs={"process_time_steps": True},
        replay_buffer_kwargs={"process_time_steps": process_token_time_step},
        action_noise=action_noise, # actor noise before kNN -> it should improve exploration of different actions
        lambda_penalty=0.0, # the results on the dev set are better, and the actor representations seem to stabilize over time
        #max_grad_norm=0.5,
        max_grad_norm=1.0,
        #invert_grad=True,
        wolpertinger_add_noise_after_knn=True, # TD3 noise -> it should improve generalization
        wolpertinger_target_policy_actor_noise=wolpertinger_target_policy_actor_noise, # noise before kNN for the target policy -> it should improve exploration of different actions
        wolpertinger_target_actor_noise_clip=wolpertinger_target_actor_noise_clip,
        #wolpertinger_target_policy_actor_noise=0.0, # TODO remove
        #wolpertinger_target_actor_noise_clip=0.0, # TODO remove
        n_steps=1,
        **td3_args,
    )

    #assert init_training_episodes < max_episodes

    # Add callbacks
    stop_train_callback = sb3_cb.StopTrainingOnNoModelImprovement(max_no_improvement_evals=patience, min_evals=0, verbose=1) if patience >= 0 else None # early stopping
    # EvalCallback
    ## it returns the average "sum of undiscounted rewards" per episode (https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html)
    ## it does not evaluate the model performance when training finishes (we evaluate below)
    custom_callback_on_eval = None if not store_model_on_eval else lambda n_calls, eval_freq: store_model(save_path, name_prefix, f"eval-{n_calls // eval_freq}", model, logger)
    callbacks.append(DelayedEvalCallback(
        env_eval_dev,
        learning_starts=init_training_steps,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=eval_freq,
        #n_eval_episodes=1,
        n_eval_episodes=len(data_to_be_translated_dev),
        callback_after_eval=stop_train_callback,
        deterministic=True,
        render=False,
        verbose=1,
        predict_kwargs={
            "knn_callback": retrieve_embeddings_dev,
            "env_instance": env_eval_dev.unwrapped,
        },
        disable_eval=disable_eval,
        custom_callback_on_eval=custom_callback_on_eval,
        custom_logger=logger,
    ))
    callbacks.append(sb3_cb.CheckpointCallback( # store training data in order to resume training later
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=name_prefix,
        save_replay_buffer=False, # too much disk...
        save_vecnormalize=True,
        verbose=2,
    ))
    callbacks.append(sb3_cb.StopTrainingOnMaxEpisodes(
        max_episodes=max_episodes,
        verbose=1,
    ))

    callback = sb3_cb.CallbackList(callbacks)

    # Train
    model.learn(max_steps, log_interval=1, callback=callback)

    # Store last version
    model_path = store_model(save_path, name_prefix, "last-step", model, logger)

    # Evaluate
    ## do not evaluate best_model as the result is already in the log (unless eval is disabled and best model is unknown)
    assert utils.file_exists(model_path), f"Model not found: {model_path}"

    ## dev: load model
    #logger.info("Loading %s model (dev): %s", "best" if patience >= 0 else "last-step", best_model_path)
    logger.info("Loading last-step model (dev): %s", model_path)

    policy_actor_kwargs["actor_lr_schedule"] = lambda foo: 100.0 # dummy callable
    model = model_class.load(
        model_path,
        learning_rate=lambda foo: 100.0, # dummy callable
        lr_schedule=lambda foo: 100.0, # dummy callable
        policy_kwargs={
            "n_critics": n_critics,
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_dev,
            "callback_retrieve_knn_training": None,
            "k": k_dev,
            #"add_all_knn_to_batch": True, # Faster
            "add_all_knn_to_batch": add_all_knn_to_batch,
            "apply_rws_inference": False,
            **policy_actor_kwargs,
            **policy_critic_kwargs,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs_actor": features_extractor_kwargs_actor,
            "features_extractor_kwargs_critic": features_extractor_kwargs_critic,
            "wolpertinger_disable_actor": wolpertinger_disable_actor,
            "critic_action_layer_norm_input": critic_action_layer_norm_input,
            "critic_features_layer_norm_input": critic_features_layer_norm_input,
            "critic_action_layer_dropout": 0.0,
            "critic_action_linear_layer_projection": critic_action_linear_layer_projection,
            "critic_features_linear_layer_projection": critic_features_linear_layer_projection,
            "activation_fn": torch.nn.GELU,
        },
    )

    ## dev: evaluate and report result
    logger.info("Evaluating dev")

    #mean_reward, std_reward = evaluate_policy(model, env_eval_dev.unwrapped, n_eval_episodes=len(data_to_be_translated_dev))
    mean_reward, std_reward = evaluate_policy(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated_dev),
        predict_kwargs={
            "env_instance": env_eval_dev.unwrapped,
        },)

    print(f"Mean reward dev: {mean_reward} +/- {std_reward}")

    ## test: load model
    logger.info("Loading last-step model (test): %s", model_path)

    env_eval_test = env_eval_test_class(src_lang_test, trg_lang_test, file_data_test, file_data_icl_examples_test, gym_logger_level=gym.logger.INFO, custom_env_id="eval_test", is_eval_env=True, **parsed_kwargs)

    env_eval_test._init_load_data_and_populate_knn_pool(options={})

    retrieve_embeddings_test = lambda proto_action, _k, observations: env_eval_test.get_closest_neighbors_urls(proto_action, k=k_test, get_representations_instead_of_embeddings=False, observations=observations, debug=False, return_all_neighbors=return_all_neighbors_test)[0]
    policy_actor_kwargs["actor_lr_schedule"] = lambda foo: 100.0 # dummy callable
    model = model_class.load(
        model_path,
        learning_rate=lambda foo: 100.0, # dummy callable
        lr_schedule=lambda foo: 100.0, # dummy callable
        policy_kwargs={
            "n_critics": n_critics,
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_test,
            "callback_retrieve_knn_training": None,
            "k": k_test,
            #"add_all_knn_to_batch": True, # Faster
            "add_all_knn_to_batch": add_all_knn_to_batch,
            "apply_rws_inference": False,
            **policy_actor_kwargs,
            **policy_critic_kwargs,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs_actor": features_extractor_kwargs_actor,
            "features_extractor_kwargs_critic": features_extractor_kwargs_critic,
            "wolpertinger_disable_actor": wolpertinger_disable_actor,
            "critic_action_layer_norm_input": critic_action_layer_norm_input,
            "critic_features_layer_norm_input": critic_features_layer_norm_input,
            "critic_action_layer_dropout": 0.0,
            "critic_action_linear_layer_projection": critic_action_linear_layer_projection,
            "critic_features_linear_layer_projection": critic_features_linear_layer_projection,
            "activation_fn": torch.nn.GELU,
        },
    )

    ## test: evaluate and report result
    logger.info("Evaluating test")

    mean_reward, std_reward = evaluate_policy(model, env_eval_test, n_eval_episodes=len(data_to_be_translated_test),
        predict_kwargs={
            "env_instance": env_eval_test,
        },)

    print(f"Mean reward test: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    main()
