
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

import gymnasium as gym
from stable_baselines3 import DDPG, TD3
import stable_baselines3.common.callbacks as sb3_cb
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor, NoFlattenExtractor, TransformerExtractor
import numpy as np
import torch

class NormalActionNoiseWithClip(NormalActionNoise):
    def __init__(self, *args, clip_value=np.inf, **kwargs):
        super().__init__(*args, **kwargs)

        self.clip_value = clip_value

    def __call__(self, *args, **kwargs):
        noise = super().__call__(*args, **kwargs)

        assert isinstance(noise, np.ndarray), f"Expected noise to be a numpy array, found {type(noise)}"

        return np.clip(noise, a_min=-self.clip_value, a_max=self.clip_value)

class InverseSqrtWithWarmUpLRSchedule:
    def __init__(self, warmup_steps, initial_lr, logger, str_id="none"):
        assert warmup_steps >= 0, warmup_steps

        self.warmup_steps = warmup_steps + 1
        self.initial_lr = initial_lr
        self.old_current_progress_remaining = np.inf
        self.step = 1
        self.logger = logger
        self.str_id = str_id

        self.logger.debug("[%s] [%s] First LR and warmup steps: %f (initial: %f) %d", self.__class__.__name__, self.str_id, self.get_lr(), self.initial_lr, self.warmup_steps)

    def __call__(self, _current_progress_remaining, _update_learning_rate=False):
        lr = self.get_lr()

        if not _update_learning_rate:
            assert np.isclose(_current_progress_remaining, 1.0), _current_progress_remaining
            assert self.step == 1, self.step

            return lr
        else:
            assert _current_progress_remaining <= self.old_current_progress_remaining, f"Expected _current_progress_remaining to be non-increasing, but got {self.old_current_progress_remaining} -> {_current_progress_remaining}"

            self.old_current_progress_remaining = _current_progress_remaining

        self.logger.debug("[%s] [%s] New LR (step %d): %f", self.__class__.__name__, self.str_id, self.step, lr)

        self.step += 1

        return lr

    def get_lr(self):
        if self.step < self.warmup_steps:
            lr = self.initial_lr * float(self.step) / float(self.warmup_steps)
        else:
            lr = self.initial_lr * (self.warmup_steps ** 0.5) * (self.step ** -0.5)

        return lr

class LinearWithWarmUpLRSchedule:
    def __init__(self, warmup_steps, initial_lr, total_steps, logger, min_lr=0.0, str_id="none"):
        assert warmup_steps >= 0, warmup_steps

        self.warmup_steps = warmup_steps + 1
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.old_current_progress_remaining = np.inf
        self.step = 1
        self.logger = logger
        self.str_id = str_id
        self.min_lr = min_lr
        self.m, self.n = np.polyfit([self.warmup_steps, self.total_steps], [self.initial_lr, 0.0], 1)

        assert self.total_steps > self.warmup_steps
        assert self.min_lr >= 0.0, self.min_lr

        self.logger.debug("[%s] [%s] First LR and warmup steps: %f (initial: %f) %d", self.__class__.__name__, self.str_id, self.get_lr(), self.initial_lr, self.warmup_steps)
        self.logger.debug("[%s] [%s] m and n values: %f %f", self.__class__.__name__, self.str_id, self.m, self.n)

    def __call__(self, _current_progress_remaining, _update_learning_rate=False):
        lr = self.get_lr()

        if not _update_learning_rate:
            assert np.isclose(_current_progress_remaining, 1.0), _current_progress_remaining
            assert self.step == 1, self.step

            return lr
        else:
            assert _current_progress_remaining <= self.old_current_progress_remaining, f"Expected _current_progress_remaining to be non-increasing, but got {self.old_current_progress_remaining} -> {_current_progress_remaining}"

            self.old_current_progress_remaining = _current_progress_remaining

        self.logger.debug("[%s] [%s] New LR (step %d): %f", self.__class__.__name__, self.str_id, self.step, lr)

        self.step += 1

        return lr

    def get_lr(self):
        if self.step < self.warmup_steps:
            lr = self.initial_lr * float(self.step) / float(self.warmup_steps)
        else:
            lr = self.m * self.step + self.n
            lr = max(lr, self.min_lr)

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

class LinearDecayScheduler:

    def __init__(self, start_val, end_val, total_steps, logger, str_id="none"):
        assert start_val >= end_val
        assert total_steps > 0, total_steps

        self.start_val = start_val
        self.end_val = end_val
        self.total_steps = total_steps
        self.step = 0
        self.logger = logger
        self.str_id = str_id

        self.logger.debug("[%s] Linear decay: start and end values, and total steps: %f-%f %d", self.str_id, self.start_val, self.end_val, self.total_steps)

    def __call__(self):
        if self.step - 1 >= self.total_steps:
            return self.end_val

        # Linear interpolation
        progress = min(1.0, self.step / self.total_steps)
        new_value = self.start_val - progress * (self.start_val - self.end_val)

        if self.step == self.total_steps:
            assert np.isclose(progress, 1.0), progress
            assert np.isclose(new_value, self.end_val), f"{new_value} vs {self.end_val}"

        self.logger.debug("[%s] Linear decay: new value (step %d): %f%s", self.str_id, self.step + 1, new_value, " (and the rest of steps...)" if self.step == self.total_steps else '')

        self.step += 1

        return new_value

def make_env(rank, env_cls, env_args, env_kwargs, seed=None, seed_add_rank=False):
    def _init():
        sys.stderr.flush()
        env = env_cls(*env_args, **{"_seed": seed + (rank if seed_add_rank else 0) if seed is not None else rank, **env_kwargs})

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
    num_envs = max(1, parsed_kwargs.pop("num_envs", 6))
    device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
    max_icl_examples = int(parsed_kwargs.get("max_icl_examples", 5))
    #max_icl_examples = 1 # TODO remove
    model_hidden_size = parsed_kwargs.get("model_hidden_size", 4096)
    apply_rws_inference = parsed_kwargs.get("apply_rws_inference", False)
    wolpertinger_disable_actor = bool(int(parsed_kwargs.pop("wolpertinger_disable_actor", 0)))
    pre_k = parsed_kwargs.pop("k", "0.15")

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
    #max_data_entries = 5 # TODO remove
    #max_data_icl_examples_entries = 8 # TODO remove
    state_representation = parsed_kwargs.get("state_representation", "representation_per_token_with_features")
    parsed_kwargs["device"] = device
    parsed_kwargs["max_icl_examples"] = max_icl_examples
    parsed_kwargs["max_data_entries"] = max_data_entries
    parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
    parsed_kwargs["state_representation"] = state_representation
    parsed_kwargs["eval_strategy"] = parsed_kwargs.get("eval_strategy", "chrf2")
    parsed_kwargs["repeat_translation_candidates"] = parsed_kwargs.get("repeat_translation_candidates", True)
    parsed_kwargs["repeat_translation_candidates_times"] = parsed_kwargs.get("repeat_translation_candidates_times", 3)
    parsed_kwargs["knn_api_retrieve"] = parsed_kwargs.get("knn_api_retrieve", None)
    parsed_kwargs["knn_api_insert"] = parsed_kwargs.get("knn_api_insert", None)
    parsed_kwargs["knn_always_add_eos_action"] = parsed_kwargs.get("knn_always_add_eos_action", True)
    parsed_kwargs["enable_eos_action"] = parsed_kwargs.get("enable_eos_action", False)
    parsed_kwargs["state_window_length"] = int(parsed_kwargs.get("state_window_length", 1024)) + 3
    parsed_kwargs["action_representation"] = parsed_kwargs.get("action_representation", "src_embedding:SONAR")
    parsed_kwargs["model_hidden_size_action_src_sentence"] = parsed_kwargs.get("model_hidden_size_action_src_sentence", 1024)
    data_to_be_translated_training = data_to_be_translated_training[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_dev = data_to_be_translated_dev[:max_data_entries if max_data_entries > 0 else None]
    data_to_be_translated_test = data_to_be_translated_test[:max_data_entries if max_data_entries > 0 else None]
    data_icl_examples_training = data_icl_examples_training[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_dev = data_icl_examples_dev[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
    data_icl_examples_test = data_icl_examples_test[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]

    #assert parsed_kwargs["knn_api_retrieve"] is not None
    #assert parsed_kwargs["knn_api_insert"] is not None

    # Some values
    pre_k_is_float = '.' in pre_k
    k_training = max(1, int(float(pre_k) * len(data_icl_examples_training)) if pre_k_is_float else int(pre_k))
    k_dev = max(1, int(float(pre_k) * len(data_icl_examples_dev)) if pre_k_is_float else int(pre_k))
    k_test = max(1, int(float(pre_k) * len(data_icl_examples_test)) if pre_k_is_float else int(pre_k))

    assert isinstance(k_training, int) # other parameters use k assuming integer instead of float
    assert isinstance(k_dev, int)
    assert isinstance(k_test, int)

    logger.info("k=(%d, %d, %d)", k_training, k_dev, k_test)

    # Other kwargs
    parsed_kwargs_training = {}
    #parsed_kwargs_training["initial_time_sleep"] = num_envs * 2 # sleep to synchronize all environments
    parsed_kwargs_training["knn_api_retrieve"] = parsed_kwargs["knn_api_retrieve"]
    parsed_kwargs_training["knn_api_insert"] = parsed_kwargs["knn_api_insert"]
    parsed_kwargs_training_dummy = {}
    parsed_kwargs_training_dummy["knn_api_retrieve"] = parsed_kwargs["knn_api_retrieve"]
    parsed_kwargs_training_dummy["knn_api_insert"] = parsed_kwargs["knn_api_insert"]

    logger.info("parsed_kwargs: %s", parsed_kwargs)

    del parsed_kwargs["knn_api_retrieve"]
    del parsed_kwargs["knn_api_insert"]

    # Other values
    filename_time = datetime.now().strftime("%Y%m%d_%H%M")
    #save_freq = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs) # steps
    save_freq = 1e1000 # disabled
    #eval_freq = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs) # steps (approx. once per epoch)
    #eval_freq = 500 # steps
    #eval_freq = 1000 # steps
    eval_freq = 2000 # steps
    #eval_freq = 5 # TODO remove
    save_path = f"./rl_models_{filename_time}/"
    name_prefix = f"rl_{filename_time}"
    #monitor_filename = f"{save_path}{name_prefix}_eval.log"
    monitor_filename = None # pickle serialization doesn't allow to have an opened file descriptor (EvalCallback)
    max_episodes_epochs = 10000 # repeat N times (patience-driven environment, so this value might not be used at all)
    max_episodes = len(data_to_be_translated_training) * max_episodes_epochs
    #patience = 6 # early stopping patience (number of evals with no improvement)
    #patience = 20 # TODO remove
    patience = -1 # disabled
    enable_eval = True

    if not enable_eval:
        logger.info("Evaluation (dev set) disabled: no best model will be available, only last model")

    if patience < 0:
        logger.info("Early stopping disabled (patience < 0)")
    else:
        logger.info("Early stopping enabled (patience: %d evals with no improvement)", patience)

    logger.info("Save path: %s", save_path)

    eval_freq = max(1, eval_freq // num_envs)

    # Environment
    env_class = MTICLEnv
    env_eval_dev_class = MTICLEvalEnv
    env_eval_test_class = MTICLEvalEnv
    #vec_env_class = DummyVecEnv # debug
    vec_env_class = SubprocVecEnv
    vec_env_kwargs = {"start_method": "forkserver"} if vec_env_class is SubprocVecEnv else {}
    #batch_size = 256
    batch_size = max(1, int(parsed_kwargs.pop("rl_batch_size", 256)))
    #batch_size = 16 # TODO remove
    net_arch = {
        #"pi": [256, 256],
        "pi": [512, 512],
        #"qf": [512, 256]
        "qf": [512, 512]
    } # "pi" is actor and "qf" the critic (ignored if transformer is used)
    gamma = 1.0
    #gamma = 0.99
    replay_buffer_size = 1000000
    #replay_buffer_size = 10000
    #critic_learning_rate = 1e-3
    #actor_learning_rate = 1e-4
    #critic_learning_rate = 5e-4
    #actor_learning_rate = 5e-4
    #critic_learning_rate = 1e-3
    #actor_learning_rate = 1e-3
    #critic_learning_rate = 1e-4
    #actor_learning_rate = 1e-5
    critic_learning_rate = 5e-4
    actor_learning_rate = 1e-4
    #max_steps = 1e100 # fake value due to callback StopTrainingOnMaxEpisodes
#    max_steps_training = 10000 # steps while training
    max_steps_training = 20000 # steps while training
    #init_training_steps = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs)
    init_training_steps = 5000
#    init_training_steps = 1000
#    init_training_steps = 2000
    #init_training_steps = 0 # TODO remove
    #init_training_steps = 5 # TODO remove
    #init_training_steps = 200 # TODO remove
    max_steps = max_steps_training + init_training_steps
    actor_mlp_l2_norm = bool(int(parsed_kwargs.pop("actor_mlp_l2_norm", 1)))
    add_all_knn_to_batch = max(1, int(parsed_kwargs.pop("add_all_knn_to_batch", 1)))

    logger.info("Evaluation frequency (steps, adjusted by number of parallel environments): %d // %d = %d (total times evaluation: %d)", eval_freq, num_envs, max(1, eval_freq // num_envs), ((max_steps - init_training_steps) / num_envs) // max(1, eval_freq / num_envs))
    logger.info("Init. steps collecting rollouts without training: %d", init_training_steps)
    logger.info("Max. steps (no_training, training, total): (%d, %d, %d)", init_training_steps, max_steps_training, max_steps)
    logger.info("actor_mlp_l2_norm: %s", actor_mlp_l2_norm)
    logger.info("Actual batch_size for running the critic with the observations and the kNN elements: %d * min(%d, k) = %d", batch_size, add_all_knn_to_batch, batch_size * add_all_knn_to_batch)

    if num_envs > 1:
        logger.info("Be aware that the environment will be executed %d time steps, but %d // %d = %d per environment (%d // %d = %d init. training steps) instance due to the number of parallel envinronments (%d training steps, where %d different batches will be used for training from the replay buffer)",
                    max_steps, max_steps, num_envs, max_steps // num_envs,
                    init_training_steps, num_envs, init_training_steps // num_envs,
                    (max_steps - init_training_steps) // num_envs, max_steps - init_training_steps)

    #env = env_class(src_lang, trg_lang, file_data_training, file_data_icl_examples_training, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)
    env_args = [src_lang_training, trg_lang_training, file_data_training, file_data_icl_examples_training]
    env_kwargs = {"gym_logger_level": gym.logger.DEBUG, **parsed_kwargs}
    env_training_dummy = env_class(*list(env_args), **dict({"custom_env_id": "training_dummy", **env_kwargs, **parsed_kwargs_training_dummy})) # WARN: each vectorized environments receives a copy of this environment

    env_training_dummy._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False}) # env_training_dummy.get_closest_neighbors_urls() is available

    parsed_kwargs_training["initial_sample_list_actions"] = [(k, env_training_dummy.str2representation[k]) for k in env_training_dummy.str2representation_valid_actions_k] # initial random action sampling
    env = vec_env_class([make_env(rank, env_class, list(env_args), dict({"custom_env_id": str(rank), **env_kwargs, **parsed_kwargs_training}), seed=env_seeds[rank]) for rank in range(num_envs)], **vec_env_kwargs)
    env_eval_dev = Monitor(env_eval_dev_class(src_lang_dev, trg_lang_dev, file_data_dev, file_data_icl_examples_dev, gym_logger_level=gym.logger.INFO, custom_env_id="eval_dev", is_eval_env=True, **parsed_kwargs), filename=monitor_filename, override_existing=True)

    env_eval_dev.unwrapped._init_load_data_and_populate_knn_pool(options={"shuffle_all_data": False}) # env_eval_dev.get_closest_neighbors_urls() is available

    retrieve_embeddings_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=k_training, get_representations_instead_of_embeddings=False, observations=observations, debug=False)[0] # Get only the result, not I or D
    retrieve_embeddings_training_training = lambda proto_action, _k, observations: env_training_dummy.get_closest_neighbors_urls(proto_action, k=k_training, get_representations_instead_of_embeddings=False, observations=observations, debug=False)[0] # Get only the result, not I or D
    retrieve_embeddings_dev = lambda proto_action, _k, observations: env_eval_dev.unwrapped.get_closest_neighbors_urls(proto_action, k=k_dev, get_representations_instead_of_embeddings=False, observations=observations, debug=True)[0]
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
    #actor_transformer_args_and_kwargs = {
    #    "d_model": 512,
    #    "nhead": 4,
    #    "dim_feedforward": 2048,
    #    "nlayers": 3,
    #    "max_seq_len": max_seq_len,
    #    "projection_in": state_dim_per_token,
    #    #"l2_norm": True, # disable to let the model learn how the representation should be
    #    "l2_norm": False,
    #    "str_id": "actor",
    #    "skip_n_word_embeddings_from_observation": f"0:{skip_we}" if state_representation == "representation_per_token_with_features" else "0:0", # skip the first word embedding, which corresponds to the source translation candidate representation
    #    #"expected_seq_len": ((state_window_length - 1) * state_dim_per_token + action_dim) // state_dim_per_token if state_representation == "representation_per_token_with_features" else None,
    #    "expected_seq_len": ((state_window_length - 1) * state_dim_per_token) // state_dim_per_token if state_representation == "representation_per_token_with_features" else None, # no action, due to skip_n_word_embeddings_from_observation
    #}
    #critic_transformer_args_and_kwargs = {
    #    "d_model": 512,
    #    "nhead": 4,
    #    "dim_feedforward": 2048,
    #    "nlayers": 3,
    #    "max_seq_len": max_seq_len + 2, # +2 for the action (source and target, in case they are different)
    #    "projection_in": state_dim_per_token,
    #    "str_id": "critic",
    #    "skip_n_word_embeddings_from_observation": f"{skip_we}:{skip_we * 2}" if state_representation == "representation_per_token_with_features" else "0:0", # "{skip_we}:{skip_we * 2}" instead of "0:{skip_we}" due to critic_first_actions_then_features == True
    #    #"expected_seq_len": ((state_window_length - 1) * state_dim_per_token + action_dim + action_dim) // state_dim_per_token if state_representation == "representation_per_token_with_features" else None,
    #    "expected_seq_len": ((state_window_length - 1) * state_dim_per_token + action_dim) // state_dim_per_token if state_representation == "representation_per_token_with_features" else None, # one action only, due to skip_n_word_embeddings_from_observation
    #}
    use_transformer = True
    dropout_p = 0.1
    #warmup_steps = 200
    warmup_steps = 1000
    #warmup_steps = 10 # TODO remove
    exploration_rate_steps_percentage = 0.5
    exploration_rate_steps = int((max_steps - init_training_steps) / num_envs * exploration_rate_steps_percentage)
    exploration_rate_initial = 1.0
    exploration_rate_last = 0.01
    #exploration_rate = LinearDecayScheduler(exploration_rate_initial, exploration_rate_last, exploration_rate_steps, logger, "epsilon-greedy exploration")
    exploration_rate = 0.1
    #wolpertinger_target_policy_actor_noise = 0.1
    #wolpertinger_target_policy_actor_noise = 0.02
    wolpertinger_target_policy_actor_noise = 0.0212
    wolpertinger_target_actor_noise_clip = 2.5 * wolpertinger_target_policy_actor_noise
    share_features_extractor = False
    #share_features_extractor = True

    if wolpertinger_disable_actor and share_features_extractor:
        share_features_extractor = False

        logger.info("Disabling share_features_extractor because wolpertinger_disable_actor is set to True")

    logger.info("Exploration rate steps: %d (%s %% of the total training steps): from %s to %s", exploration_rate_steps, exploration_rate_steps_percentage * 100, exploration_rate_initial, exploration_rate_last)

    if patience >= 0:
        actor_lr_schedule = InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=actor_learning_rate, logger=logger, str_id="actor") # callable
        critic_lr_schedule = InverseSqrtWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, logger=logger, str_id="critic") # callable
    else:
        total_steps = (max_steps - init_training_steps) // num_envs
        actor_lr_schedule = LinearWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=actor_learning_rate, total_steps=total_steps, logger=logger, str_id="actor")
        critic_lr_schedule = LinearWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, total_steps=total_steps, logger=logger, str_id="critic")

    policy_actor_kwargs = {
        #"squash_output": True,
        "squash_output": not actor_mlp_l2_norm, # actor l2-norm
        #"actor_lr_schedule": lambda foo: actor_learning_rate, # callable
        "actor_lr_schedule": actor_lr_schedule,
        "actor_layer_norm_input": True,
        "actor_layer_norm_before_activation": True,
        "actor_last_layer_init_uniform_value": 0.001,
        #"actor_dropout": True,
        #"actor_dropout_p": 0.1,
        #"actor_transformer": use_transformer,
        #"actor_transformer_args_and_kwargs": actor_transformer_args_and_kwargs,
        "actor_mlp_l2_norm": actor_mlp_l2_norm,
    }
    policy_critic_kwargs = {
        #"critic_lr_schedule": lambda foo: critic_learning_rate, # callable
        #"lr_schedule": critic_lr_schedule,
        "critic_layer_norm_input": True,
        "critic_layer_norm_before_activation": True,
        #"critic_last_layer_init_uniform_value": 0.001,
        #"critic_dropout": True,
        #"critic_dropout_p": dropout_p,
        #"critic_transformer": use_transformer,
        #"critic_transformer_args_and_kwargs": critic_transformer_args_and_kwargs,
        #"critic_first_actions_then_features": True if use_transformer else False,
    }

    if not use_transformer:
        features_extractor_class = FlattenExtractor
        #features_extractor_class = NoFlattenExtractor
        features_extractor_kwargs_actor = {}
        features_extractor_kwargs_critic = {}
    else:
        # the feature extractor only processes the state (not the action for the critic)
        features_extractor_class = TransformerExtractor
        transformer_common_kwargs = {
            #"d_model": 512,
            "d_model": 128,
            "nhead": 4,
            #"dim_feedforward": 2048,
            "dim_feedforward": 512,
            #"nlayers": 3,
            "nlayers": 2,
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
        },
        gamma=gamma,
        device=device,
        gradient_steps=-1 if num_envs > 1 else 1, # "do as many gradient steps as steps done in the environment during the rollout", also recommended for off-policy algorithms with num_envs > 1
        train_freq=(1, "step"),
        buffer_size=replay_buffer_size,
        action_noise=action_noise, # actor noise before kNN -> it should improve exploration of different actions
        lambda_penalty=0.0, # the results on the dev set are better, and the actor representations seem to stabilize over time
        max_grad_norm=1.0,
        #invert_grad=True,
        wolpertinger_add_noise_after_knn=True, # TD3 noise -> it should improve generalization
        wolpertinger_target_policy_actor_noise=wolpertinger_target_policy_actor_noise, # noise before kNN for the target policy -> it should improve exploration of different actions
        wolpertinger_target_actor_noise_clip=wolpertinger_target_actor_noise_clip,
        #wolpertinger_target_policy_actor_noise=0.0, # TODO remove
        #wolpertinger_target_actor_noise_clip=0.0, # TODO remove
        n_steps=max_icl_examples, # Monte-carlo TD3/DDPG instead of 1-step TD. Better for handling differences in the length of episodes for the early-stopping action (for 1-step TD, the episodes of length 1 due to early stopping are selected most times)
        **td3_args,
    )

    #assert init_training_episodes < max_episodes

    # Add callbacks
    stop_train_callback = sb3_cb.StopTrainingOnNoModelImprovement(max_no_improvement_evals=patience, min_evals=0, verbose=1) if patience >= 0 else None # early stopping
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

    # Store last version (adapted from CheckpointCallback)
    last_model_path = f"{name_prefix}_last-step_model.zip"
    model.save(os.path.join(save_path, last_model_path))
    #model.save_replay_buffer(os.path.join(save_path, f"{name_prefix}_last-step_replay-buffer.pkl")) # too much disk...
    if model.get_vec_normalize_env() is not None:
        model.get_vec_normalize_env().save(os.path.join(save_path, f"{name_prefix}_last-step_vecnormalize.pkl"))

    # Evaluate
    best_model_path = os.path.join(save_path, last_model_path) # do not evaluate best_model as the result is already in the log

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

    retrieve_embeddings_test = lambda proto_action, _k, observations: env_eval_test.get_closest_neighbors_urls(proto_action, k=k_test, get_representations_instead_of_embeddings=False, observations=observations, debug=True)[0]
    policy_actor_kwargs["actor_lr_schedule"] = lambda foo: 100.0 # dummy callable
    model = model_class.load(
        best_model_path,
        learning_rate=lambda foo: 100.0, # dummy callable
        lr_schedule=lambda foo: 100.0, # dummy callable
        policy_kwargs={
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
        },
    )

    ## test: evaluate and report result
    logger.info("Evaluating test")

    mean_reward, std_reward = evaluate_policy(model, env_eval_test, n_eval_episodes=len(data_to_be_translated_test))

    print(f"Mean reward test: {mean_reward} +/- {std_reward}")
