
import os
import sys
import logging
from datetime import datetime

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

import utils
from gym_env_v1 import MTICLEnv
from gym_env_v1_eval import MTICLEvalEnv
#from gym_env_v1_eval_single_episode import MTICLEvalSingleEpisodeEnv

import gymnasium as gym
from stable_baselines3 import DDPG, TD3
import stable_baselines3.common.callbacks as sb3_cb
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import numpy as np

class SelectActionNoiseFromList(ActionNoise):

    def __init__(self, noises, p=None):
        assert isinstance(noises, list)

        for n in noises:
            assert isinstance(n, ActionNoise), f"Expected all elements in noises to be ActionNoise instances, found {type(n)}"

        self.noises = noises
        self.p = p

        super().__init__()

    def __call__(self):
        idx = np.random.choice(range(len(self.noises)), size=1, p=self.p)[0]
        noise = self.noises[idx]()

        assert isinstance(noise, np.ndarray), f"Expected noise to be a numpy array, found {type(noise)}"

        return noise

    def __repr__(self) -> str:
        return f"SelectActionNoiseFromList(noises={self.noises}, p={self.p})"

class VectorFromPoolActionNoise(ActionNoise):

    def __init__(self, pool):
        self.pool = pool

        assert len(self.pool) > 0, "Pool must not be empty"

        super().__init__()

    def __call__(self):
        idx = np.random.choice(range(len(self.pool)), size=1)[0]
        noise = self.pool[idx]

        assert isinstance(noise, np.ndarray), f"Expected noise to be a numpy array, found {type(noise)}"

        return noise

    def __repr__(self) -> str:
        return f"VectorFromPoolActionNoise(len(pool)={len(self.pool)})"

def make_env(rank, env_cls, env_args, env_kwargs, seed=0):
    def _init():
        sys.stderr.flush()
        env = env_cls(*env_args, **{"_seed": seed + rank, **env_kwargs})

        #env.reset(seed=seed + rank, options={"soft_reset_after_hard_reset": False})

        return env

    return _init

if __name__ == "__main__":
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

    for _file_data, data_to_be_translated in ((file_data_training, data_to_be_translated_training), (file_data_dev, data_to_be_translated_dev), (file_data_test, data_to_be_translated_test)):
        with open(_file_data, "rt") as fd:
            for line in fd:
                data_to_be_translated.append(line.rstrip("\r\n"))

    # default values
    num_envs = parsed_kwargs.get("num_envs", 6)
    device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    max_icl_examples = parsed_kwargs.get("max_icl_examples", 4)
    state_representation = parsed_kwargs.get("state_representation", "sentence_and_actions")
    eval_strategy = parsed_kwargs.get("eval_strategy", "comet-22-da")
    dimensionality_reduction_factor_state_and_action = parsed_kwargs.get("dimensionality_reduction_factor_state_and_action", 1)

    # set defaults in case they are not provided
    max_data_entries = parsed_kwargs.get("max_data_entries", -1) # load all data (default value)
    max_data_icl_examples_entries = parsed_kwargs.get("max_data_icl_examples_entries", -1) # load all data (default value)
    parsed_kwargs["device"] = device
    parsed_kwargs["max_icl_examples"] = max_icl_examples
    parsed_kwargs["max_data_entries"] = max_data_entries
    parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    parsed_kwargs["state_representation"] = state_representation
    parsed_kwargs["eval_strategy"] = eval_strategy
    parsed_kwargs["dimensionality_reduction_factor_state_and_action"] = dimensionality_reduction_factor_state_and_action
    data_to_be_translated_training = data_to_be_translated_training[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_dev = data_to_be_translated_dev[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_test = data_to_be_translated_test[:max_data_entries if max_data_entries > 0 else None]

    # Some values
    #k = 0.01
    #k = 100
    k = 10

    # Other kwargs
    parsed_kwargs_training = {}
    parsed_kwargs_training["initial_time_sleep"] = num_envs * 2 # sleep to synchronize all environments
    parsed_kwargs_training["prob_add_saturated_action"] = 1.0
    parsed_kwargs_training["add_saturated_action_k"] = k
    parsed_kwargs_training_dummy = {}
    #parsed_kwargs_training_dummy["add_n_random_saturated_actions"] = 100000

    # Other values
    filename_time = datetime.now().strftime("%Y%m%d_%H%M")
    save_freq = len(data_to_be_translated_training) * max_icl_examples // num_envs # steps (save model approx. once per epoch)
    eval_freq = len(data_to_be_translated_training) // 2 * max_icl_examples // num_envs # steps
    save_path = "./rl_models12/"
    name_prefix = f"rl_model_{filename_time}"
    #monitor_filename = f"{save_path}{name_prefix}_eval.log"
    monitor_filename = None # pickle serialization doesn't allow to have an opened file descriptor (EvalCallback)
    max_episodes_epochs = 10 # repeat N times
    max_episodes = len(data_to_be_translated_training) * max_episodes_epochs

    # Environment
    env_class = MTICLEnv
    #env_eval_class = MTICLEvalSingleEpisodeEnv
    env_eval_dev_class = MTICLEvalEnv
    env_eval_test_class = MTICLEvalEnv
    #vec_env_class = DummyVecEnv
    vec_env_class = SubprocVecEnv
    vec_env_kwargs = {"start_method": "forkserver"} if vec_env_class is SubprocVecEnv else {}
    batch_size = 256
    net_arch = {"pi": [400, 300], "qf": [400, 300]} # paper architecture
    gamma = 1.0
    #gamma = 0.99
    replay_buffer_size = 1000000
    #replay_buffer_size = 100000
    seed = 42
    critic_learning_rate = 1e-3
    actor_learning_rate = 1e-4
    init_training_episodes = 1000
    #max_steps = max_episodes * max_icl_examples
    max_steps = 1e100 # fake value due to callback StopTrainingOnMaxEpisodes
    init_training_steps = init_training_episodes * max_icl_examples
    #env = env_class(src_lang, trg_lang, file_data_training, file_data_icl_examples_training, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)
    env_args = [src_lang_training, trg_lang_training, file_data_training, file_data_icl_examples_training]
    env_kwargs = {"gym_logger_level": gym.logger.DEBUG, **parsed_kwargs}
    env_training_dummy = env_class(*list(env_args), **dict({"custom_env_id": "training_dummy", **env_kwargs, **parsed_kwargs_training_dummy})) # WARN: each vectorized environments receives a copy of this environment

    env_training_dummy._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": True}) # env_training_dummy.get_closest_neighbors_urls() is available

    parsed_kwargs_training["knn_callback"] = env_training_dummy.get_closest_neighbors_urls # avoid non-stationary behaviour due to different fake actions among the different environments
    parsed_kwargs_training["knn_callback_data_icl_examples"] = env_training_dummy.data_icl_examples
    env = vec_env_class([make_env(rank, env_class, list(env_args), dict({"custom_env_id": str(rank), **env_kwargs, **parsed_kwargs_training}), seed=42) for rank in range(num_envs)], **vec_env_kwargs)
    env_eval_dev = Monitor(env_eval_dev_class(src_lang_dev, trg_lang_dev, file_data_dev, file_data_icl_examples_dev, gym_logger_level=gym.logger.INFO, custom_env_id="eval_dev", **parsed_kwargs), filename=monitor_filename, override_existing=True)
    env_eval_test = env_eval_test_class(src_lang_test, trg_lang_test, file_data_test, file_data_icl_examples_test, gym_logger_level=gym.logger.INFO, custom_env_id="eval_test", **parsed_kwargs)

    #env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False}) # env_eval_dev.get_closest_neighbors_urls() is available
    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False})
    env_eval_test._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False})

    retrieve_embeddings_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False)[0] # Get only the result, not I or D
    retrieve_embeddings_training_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, debug=True)[0] # Get only the result, not I or D
    retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False)[0]
    retrieve_embeddings_test = lambda proto_action, _k, observations: env_eval_test.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False)[0]
    n_actions = env.unwrapped.action_space.shape[-1]
    normal_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    #pool_action_noise = VectorFromPoolActionNoise(np.array(env_training_dummy.random_vectors, copy=True)) # force bad actions with small probability

    #del env_training_dummy.random_vectors

    #action_noise = SelectActionNoiseFromList([normal_action_noise, pool_action_noise], p=[0.95, 0.05])
    action_noise = normal_action_noise
    callbacks = []
    #model_class = DDPG
    model_class = TD3
    #model = DDPG(
    model = model_class(
        "WolpertingerPolicy",
        env,
        verbose=1,
        seed=seed,
        batch_size=batch_size,
        learning_starts=init_training_steps,
        learning_rate=critic_learning_rate,
        policy_kwargs={
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_training,
            "callback_retrieve_knn_training": retrieve_embeddings_training_training,
            "k": k,
            "add_all_knn_to_batch": True, # Faster
            #"add_all_knn_to_batch": False, # Better avoid due to removal of overlapping actions
            "apply_rws_inference": False,
            "actor_lr_schedule": lambda foo: actor_learning_rate, # callable
            "actor_layer_norm_input": True,
            "actor_layer_norm_before_activation": True,
            "actor_last_layer_init_uniform_value": 0.001,
        },
        gamma=gamma,
        device=device,
        gradient_steps=-1 if num_envs > 1 else 1, # "do as many gradient steps as steps done in the environment during the rollout", also recommended for off-policy algorithms with num_envs > 1
        #gradient_steps=1,
        #train_freq=(1, "step"),
        #train_freq=(batch_size, "step"),
        #train_freq=(1, "episode"), # config in the wolpertinger policy paper
        train_freq=1,
        buffer_size=replay_buffer_size,
        action_noise=action_noise,
        lambda_penalty=1e-3,
        max_grad_norm=1.0,
        invert_grad=True,
    )

    #assert init_training_episodes < max_episodes

    # Add callbacks
    stop_train_callback = sb3_cb.StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1) # early stopping
    # EvalCallback
    ## it returns the average "sum of undiscounted rewards" per episode (https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html)
    ## it does not evaluate the model performance when training finishes (we evaluate below)
    callbacks.append(sb3_cb.EvalCallback(
        env_eval_dev,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=eval_freq,
        #n_eval_episodes=1,
        n_eval_episodes=len(data_to_be_translated_dev),
        callback_after_eval=stop_train_callback,
        deterministic=True,
        render=False,
        verbose=1,
    ))
    callbacks.append(sb3_cb.CheckpointCallback( # store training data in order to resume training later
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=name_prefix,
        save_replay_buffer=True,
        verbose=1,
    ))
    callbacks.append(sb3_cb.StopTrainingOnMaxEpisodes(
        max_episodes=max_episodes,
        verbose=1,
    ))

    callback = sb3_cb.CallbackList(callbacks)

    # Train
    model.learn(max_steps, log_interval=1, callback=callback)

    # Evaluate
    best_model_path = os.path.join(save_path, "best_model.zip")

    assert utils.file_exists(best_model_path), f"Best model not found: {best_model_path}"

    ## dev: load model
    logger.info("Loading best model (dev): %s", best_model_path)

    model = model_class.load(
        best_model_path,
        policy_kwargs={
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_dev,
            "callback_retrieve_knn_training": None,
            "k": k,
            "add_all_knn_to_batch": True, # Faster
            "apply_rws_inference": False,
            "actor_lr_schedule": lambda foo: actor_learning_rate, # callable
            "actor_layer_norm_input": True,
            "actor_layer_norm_before_activation": True,
            "actor_last_layer_init_uniform_value": None,
        },
    )

    ## dev: evaluate and report result
    logger.info("Evaluating dev")

    mean_reward, std_reward = evaluate_policy(model, env_eval_dev.unwrapped, n_eval_episodes=len(data_to_be_translated_dev))

    print(f"Mean reward dev: {mean_reward} +/- {std_reward}")

    ## test: load model
    logger.info("Loading best model (test): %s", best_model_path)

    model = model_class.load(
        best_model_path,
        policy_kwargs={
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_test,
            "callback_retrieve_knn_training": None,
            "k": k,
            "add_all_knn_to_batch": True, # Faster
            "apply_rws_inference": False,
            "actor_lr_schedule": lambda foo: actor_learning_rate, # callable
            "actor_layer_norm_input": True,
            "actor_layer_norm_before_activation": True,
            "actor_last_layer_init_uniform_value": None,
        },
    )

    ## test: evaluate and report result
    logger.info("Evaluating test")

    mean_reward, std_reward = evaluate_policy(model, env_eval_test, n_eval_episodes=len(data_to_be_translated_test))

    print(f"Mean reward test: {mean_reward} +/- {std_reward}")
