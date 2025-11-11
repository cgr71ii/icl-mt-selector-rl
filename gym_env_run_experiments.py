
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
from stable_baselines3.common.torch_layers import FlattenExtractor, NoFlattenExtractor
import numpy as np
import torch

# TODO implement a replay buffer that samples from two different bins (given a probability p): actions with >= reward and < reward

class InverseSqrtWithWarmUpLRSchedule:
    def __init__(self, warmup_steps, initial_lr, logger, str_id="none"):
        assert warmup_steps >= 0, warmup_steps

        self.warmup_steps = warmup_steps + 1
        self.initial_lr = initial_lr
        self.old_current_progress_remaining = np.inf
        self.step = 1
        self.logger = logger
        self.str_id = str_id

        self.logger.debug("[%s] First LR and warmup steps: %f (initial: %f) %d", self.str_id, self.get_lr(), self.initial_lr, self.warmup_steps)

    def __call__(self, _current_progress_remaining, _update_learning_rate=False):
        lr = self.get_lr()

        if not _update_learning_rate:
            assert np.isclose(_current_progress_remaining, 1.0), _current_progress_remaining
            assert self.step == 1, self.step

            return lr
        else:
            assert _current_progress_remaining <= self.old_current_progress_remaining, f"Expected _current_progress_remaining to be non-increasing, but got {self.old_current_progress_remaining} -> {_current_progress_remaining}"

            self.old_current_progress_remaining = _current_progress_remaining

        self.logger.debug("[%s] New LR (step %d): %f", self.str_id, self.step, lr)

        self.step += 1

        return lr

    def get_lr(self):
        if self.step < self.warmup_steps:
            lr = self.initial_lr * float(self.step) / float(self.warmup_steps)
        else:
            lr = self.initial_lr * (self.warmup_steps ** 0.5) * (self.step ** -0.5)

        return lr

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

class DelayedEvalCallback(sb3_cb.EvalCallback):
    def __init__(self, *args, learning_starts=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_starts = learning_starts

    def _on_step(self) -> bool:
        # Only start evaluating after learning_starts
        if self.num_timesteps < self.learning_starts:
            self.n_calls = 0 # so it starts from this point when this conditions does not hold

            return True

        return super()._on_step()

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
    num_envs = parsed_kwargs.get("num_envs", 4)
    device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    max_icl_examples = parsed_kwargs.get("max_icl_examples", 4)
    #max_icl_examples = 1 # TODO remove
    #state_representation = parsed_kwargs.get("state_representation", "sentence_and_actions")
    state_representation = parsed_kwargs.get("state_representation", "model_single_representation")
    eval_strategy = parsed_kwargs.get("eval_strategy", "comet-22-da")
    dimensionality_reduction_factor_state_and_action = parsed_kwargs.get("dimensionality_reduction_factor_state_and_action", 256)
    repeat_translation_candidates = parsed_kwargs.get("repeat_translation_candidates", False)
    dimensionality_reduction_type = parsed_kwargs.get("dimensionality_reduction_type", "iterative_nonoverlapping_average")
    model_hidden_size = parsed_kwargs.get("model_hidden_size", 4096)
    #dimensionality_reduction_type = "fixed_orthogonal_projection" # TODO remove
    knn_always_add_eos_action = parsed_kwargs.get("knn_always_add_eos_action", True) # TODO remove?
    apply_rws_inference = parsed_kwargs.get("apply_rws_inference", False)

    # set defaults in case they are not provided
    max_data_entries = parsed_kwargs.get("max_data_entries", -1) # load all data (default value)
    max_data_icl_examples_entries = parsed_kwargs.get("max_data_icl_examples_entries", -1) # load all data (default value)
    #max_data_icl_examples_entries = 100 # TODO remove
    #max_data_entries = 1 # TODO remove
    parsed_kwargs["device"] = device
    parsed_kwargs["max_icl_examples"] = max_icl_examples
    parsed_kwargs["max_data_entries"] = max_data_entries
    parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    parsed_kwargs["state_representation"] = state_representation
    parsed_kwargs["eval_strategy"] = eval_strategy
    parsed_kwargs["dimensionality_reduction_factor_state_and_action"] = int(dimensionality_reduction_factor_state_and_action)
    parsed_kwargs["repeat_translation_candidates"] = repeat_translation_candidates
    parsed_kwargs["knn_api_retrieve"] = parsed_kwargs.get("knn_api_retrieve", None)
    parsed_kwargs["knn_api_insert"] = parsed_kwargs.get("knn_api_insert", None)
    parsed_kwargs["dimensionality_reduction_type"] = dimensionality_reduction_type
    data_to_be_translated_training = data_to_be_translated_training[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_dev = data_to_be_translated_dev[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_test = data_to_be_translated_test[:max_data_entries if max_data_entries > 0 else None]
    parsed_kwargs["knn_always_add_eos_action"] = knn_always_add_eos_action

    #assert parsed_kwargs["knn_api_retrieve"] is not None
    #assert parsed_kwargs["knn_api_insert"] is not None

    # Some values
    #k = 0.01
    #k = 100
    k = 20
    #k = 101 # TODO remove

    assert isinstance(k, int) # other parameters use k assuming integer instead of float

    logger.info("k=%s", k)

    # Other kwargs
    parsed_kwargs_training = {}
    #parsed_kwargs_training["initial_time_sleep"] = num_envs * 2 # sleep to synchronize all environments
    parsed_kwargs_training["knn_api_retrieve"] = parsed_kwargs["knn_api_retrieve"]
    parsed_kwargs_training["knn_api_insert"] = parsed_kwargs["knn_api_insert"]
    parsed_kwargs_training_dummy = {}
    parsed_kwargs_training_dummy["knn_api_retrieve"] = parsed_kwargs["knn_api_retrieve"]
    parsed_kwargs_training_dummy["knn_api_insert"] = parsed_kwargs["knn_api_insert"]

    del parsed_kwargs["knn_api_retrieve"]
    del parsed_kwargs["knn_api_insert"]

    # Other values
    filename_time = datetime.now().strftime("%Y%m%d_%H%M")
    save_freq = len(data_to_be_translated_training) * max_icl_examples // num_envs # steps (save model approx. once per epoch)
    eval_freq = len(data_to_be_translated_training) // 2 * max_icl_examples // num_envs # steps
    save_freq = 1e1000 # TODO remove
    eval_freq = 50 # TODO remove
    #eval_freq = 10 # TODO remove
    save_path = f"./rl_models_{filename_time}/"
    name_prefix = f"rl_model_{filename_time}"
    #monitor_filename = f"{save_path}{name_prefix}_eval.log"
    monitor_filename = None # pickle serialization doesn't allow to have an opened file descriptor (EvalCallback)
    max_episodes_epochs = 10 # repeat N times
    max_episodes_epochs = 10000 # TODO remove
    max_episodes = len(data_to_be_translated_training) * max_episodes_epochs

    logger.info("Save path: %s", save_path)

    # Environment
    env_class = MTICLEnv
    #env_eval_class = MTICLEvalSingleEpisodeEnv
    env_eval_dev_class = MTICLEvalEnv
    env_eval_test_class = MTICLEvalEnv
    #vec_env_class = DummyVecEnv
    vec_env_class = SubprocVecEnv
    vec_env_kwargs = {"start_method": "forkserver"} if vec_env_class is SubprocVecEnv else {}
    batch_size = 256
#    batch_size = 512
    net_arch = {
        "pi": [400, 300],
#        "pi": [1024, 512, 1024],
        "qf": [400, 300]
    } # paper architecture ("pi" is actor and "qf" the critic)
    gamma = 1.0
    #gamma = 0.99
    replay_buffer_size = 1000000
    #replay_buffer_size = 100000
    seed = 42
    #critic_learning_rate = 1e-3
    #actor_learning_rate = 1e-4
    critic_learning_rate = 1e-4
    actor_learning_rate = 1e-4
    #init_training_episodes = 1000
    #init_training_episodes = 500 # TODO remove
    init_training_episodes = 100 # TODO remove
    #max_steps = max_episodes * max_icl_examples
    max_steps = 1e100 # fake value due to callback StopTrainingOnMaxEpisodes
    init_training_steps = init_training_episodes * max_icl_examples
    #env = env_class(src_lang, trg_lang, file_data_training, file_data_icl_examples_training, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)
    env_args = [src_lang_training, trg_lang_training, file_data_training, file_data_icl_examples_training]
    env_kwargs = {"gym_logger_level": gym.logger.DEBUG, **parsed_kwargs}
    env_training_dummy = env_class(*list(env_args), **dict({"custom_env_id": "training_dummy", **env_kwargs, **parsed_kwargs_training_dummy})) # WARN: each vectorized environments receives a copy of this environment

    env_training_dummy._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": True}) # env_training_dummy.get_closest_neighbors_urls() is available

    parsed_kwargs_training["initial_sample_list_actions"] = [env_training_dummy.str2representation[k] for k in env_training_dummy.str2representation_valid_actions_k] # initial random action sampling
    env = vec_env_class([make_env(rank, env_class, list(env_args), dict({"custom_env_id": str(rank), **env_kwargs, **parsed_kwargs_training}), seed=42) for rank in range(num_envs)], **vec_env_kwargs)
    env_eval_dev = Monitor(env_eval_dev_class(src_lang_dev, trg_lang_dev, file_data_dev, file_data_icl_examples_dev, gym_logger_level=gym.logger.INFO, custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs), filename=monitor_filename, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False}) # env_eval_dev.get_closest_neighbors_urls() is available

    retrieve_embeddings_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, observations=observations)[0] # Get only the result, not I or D
    retrieve_embeddings_training_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, observations=observations, debug=True)[0] # Get only the result, not I or D
    retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, observations=observations)[0]
    n_actions = env.unwrapped.action_space.shape[-1]
    normal_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    #normal_action_noise = None
    #pool_action_noise = VectorFromPoolActionNoise(np.array(env_training_dummy.random_vectors, copy=True)) # force bad actions with small probability

    #del env_training_dummy.random_vectors

    #action_noise = SelectActionNoiseFromList([normal_action_noise, pool_action_noise], p=[0.95, 0.05])
    action_noise = normal_action_noise

    #assert action_noise is None, "Action noise should not be used for Wolpertinger policy due to the discretization of the action space (target action noise during TD3 algorithm is different)"
    # I modified the code to add the action_noise before kNN

    callbacks = []
    #model_class = DDPG
    model_class = TD3
    #model = DDPG(
    td3_args = {
        "policy_delay": 2,
        #"target_policy_noise": 0.2, # default value in TD3 -> it diverges due to too much noise for our setting of discrete actions
        #"target_noise_clip": 0.5, # default value in TD3
        #"target_policy_noise": 0.0, # TODO remove noise? target policy noise may not make sense with Wolpertinger policy (but then generalization may be worse to different data...)
        #"target_policy_noise": 0.1,
        #"target_noise_clip": 0.25,
        #"target_policy_noise": 0.01,
        #"target_noise_clip": 0.1,
        #"target_noise_clip": 0.0, # disable temporarily for debug purposes TODO enable with small values for better generalization and robustness
        "target_policy_noise": 1e-3,
        "target_noise_clip": 2e-3,
    } if model_class is TD3 else {}
    actor_transformer_args_and_kwargs = {
        "d_model": 512,
        "nhead": 4,
        "dim_feedforward": 2048,
        "nlayers": 3,
        "max_seq_len": 16,
        "projection_in": model_hidden_size,
        #"l2_norm": True, # disable and let l2 norm penalty handle it to improve stability through regularization
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
    warmup_steps = 200
    policy_actor_kwargs = {
        #"actor_lr_schedule": lambda foo: actor_learning_rate, # callable
        "actor_lr_schedule": InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=actor_learning_rate, logger=logger, str_id="actor"), # callable
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
    model = model_class(
        "WolpertingerPolicy",
        env,
        verbose=1,
        seed=seed,
        batch_size=batch_size,
        learning_starts=init_training_steps,
        #learning_rate=critic_learning_rate,
        learning_rate=InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, logger=logger, str_id="critic"), # callable
        policy_kwargs={
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
            "k": k,
            "add_all_knn_to_batch": True, # Faster
            #"add_all_knn_to_batch": False, # Better avoid due to removal of overlapping actions
            "apply_rws_inference": apply_rws_inference,
            **policy_actor_kwargs,
            **policy_critic_kwargs,
            "squash_output": True,
            "features_extractor_class": features_extractor_class,
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
        #lambda_penalty=1e-1,
        #lambda_penalty=1e-2,
        #lambda_penalty=0.0, # TODO does the actor saturate the representation when disabled? it seems so, but the results are better somehow
        max_grad_norm=1.0,
        #invert_grad=True,
        wolpertinger_target_policy_actor_noise=0.1,
        wolpertinger_target_actor_noise_clip=0.25,
        n_steps=max_icl_examples, # Monte-carlo TD3/DDPG instead of 1-step TD. Better for handling differences in the length of episodes for the early-stopping action (for 1-step TD, the episodes of length 1 due to early stopping are selected most times)
        **td3_args,
    )

    #assert init_training_episodes < max_episodes

    # Add callbacks
    stop_train_callback = sb3_cb.StopTrainingOnNoModelImprovement(max_no_improvement_evals=6, min_evals=0, verbose=1) # early stopping
    # EvalCallback
    ## it returns the average "sum of undiscounted rewards" per episode (https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html)
    ## it does not evaluate the model performance when training finishes (we evaluate below)
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
        }
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

    policy_actor_kwargs["actor_lr_schedule"] = lambda foo: 100.0 # dummy callable
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

    #mean_reward, std_reward = evaluate_policy(model, env_eval_dev.unwrapped, n_eval_episodes=len(data_to_be_translated_dev))
    mean_reward, std_reward = evaluate_policy(model, env_eval_dev, n_eval_episodes=len(data_to_be_translated_dev))

    print(f"Mean reward dev: {mean_reward} +/- {std_reward}")

    ## test: load model
    logger.info("Loading best model (test): %s", best_model_path)

    env_eval_test = env_eval_test_class(src_lang_test, trg_lang_test, file_data_test, file_data_icl_examples_test, gym_logger_level=gym.logger.INFO, custom_env_id="eval_test", is_eval_env=True, **parsed_kwargs)

    env_eval_test._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False})

    retrieve_embeddings_test = lambda proto_action, _k, observations: env_eval_test.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False, observations=observations)[0]
    policy_actor_kwargs["actor_lr_schedule"] = lambda foo: 100.0 # dummy callable
    model = model_class.load(
        best_model_path,
        learning_rate=lambda foo: 100.0, # dummy callable
        lr_schedule=lambda foo: 100.0, # dummy callable
        policy_kwargs={
            "net_arch": dict(net_arch),
            "callback_retrieve_knn": retrieve_embeddings_test,
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

    ## test: evaluate and report result
    logger.info("Evaluating test")

    mean_reward, std_reward = evaluate_policy(model, env_eval_test, n_eval_episodes=len(data_to_be_translated_test))

    print(f"Mean reward test: {mean_reward} +/- {std_reward}")
