
import os
import gc
import sys
import random
import logging
from datetime import datetime
import copy

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

import utils
from gym_env_v1 import MTICLEnv
from gym_env_v1_eval import MTICLEvalEnv
#from gym_env_v1_eval_single_episode import MTICLEvalSingleEpisodeEnv
from gym_env_run_experiments import LinearDecayScheduler, DelayedEvalCallback, LinearWithWarmUpLRSchedule
from gym_env_run_experiments import make_env, store_model

import gymnasium as gym
from stable_baselines3 import PPO
import stable_baselines3.common.callbacks as sb3_cb
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor, NoFlattenExtractor, TransformerExtractor, NFeaturesExtractor, NFeaturesExtractorWithTimeStepEmbeddings
from stable_baselines3.common.buffers import NStepReplayBuffer, MonteCarloReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic, ContinuousCriticTower
from stable_baselines3.common.vec_env.vec_normalize import VecNormalizeRangeAndRewardSentenceLevelICL
import numpy as np
import torch
import optuna

def create_callback_episode_rewards(n_eval_episodes, optuna_trial=None):
    step = 0

    def callback_compute_episode_rewards_and_lengths(l, g):
        nonlocal step

        env = l["env"]
        n_envs = l["n_envs"]
        episode_rewards = l["episode_rewards"]
        episode_lengths = l["episode_lengths"]
        episode_rewards = []
        episode_lengths = []
        all_rewards_data = env.unwrapped.get_attr("rewards")
        #gym.logger.error("wasd8686: %s", all_rewards)
        all_rewards = list([[[r2[0] for r2 in r] for r in rewards_data] for rewards_data in all_rewards_data])

        assert len(all_rewards) == len(n_eval_episodes)

        for i in range(n_envs):
            # Remove extra evaluations
            all_rewards[i] = all_rewards[i][:n_eval_episodes[i]]

            assert len(all_rewards[i]) == n_eval_episodes[i]

            for j in range(len(all_rewards[i])):
                assert isinstance(all_rewards[i][j], list)

                episode_lengths.append(len(all_rewards[i][j]))

                all_rewards[i][j] = sum(all_rewards[i][j])

            episode_rewards.extend(all_rewards[i])

            assert len(episode_rewards) == len(episode_lengths)

        if optuna_trial is not None:
            optuna_trial.report(np.mean(episode_rewards), step=step)

            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        step += 1

        return episode_rewards, episode_lengths

    return callback_compute_episode_rewards_and_lengths

def get_callback_after_eval(n_envs, data_to_be_translated, n_eval_episodes):

    def inner_callback_after_eval(env):
        all_rewards_data = env.get_attr("rewards") # reset rewards
        all_source_sentences_and_refs_data = list([[[r2[1] for r2 in r] for r in rewards_data] for rewards_data in all_rewards_data])

        assert len(all_rewards_data) == n_envs

        for n1 in range(n_envs):
            assert isinstance(all_source_sentences_and_refs_data[n1], list)
            all_source_sentences_and_refs_data[n1] = all_source_sentences_and_refs_data[n1][:n_eval_episodes[n1]]

            assert len(all_source_sentences_and_refs_data[n1]) == n_eval_episodes[n1]

            for n2 in range(len(all_source_sentences_and_refs_data[n1])):
                assert isinstance(all_source_sentences_and_refs_data[n1][n2], list), f"{n1} {n2} {all_source_sentences_and_refs_data[n1]}"
                assert len(set(all_source_sentences_and_refs_data[n1][n2])) in (0, 1), all_source_sentences_and_refs_data[n1][n2]

                if len(set(all_source_sentences_and_refs_data[n1][n2])) == 0:
                    del all_source_sentences_and_refs_data[n1][n2]
                else:
                    all_source_sentences_and_refs_data[n1][n2] = all_source_sentences_and_refs_data[n1][n2][0]

        all_source_sentences_and_refs_data = [x for xs in all_source_sentences_and_refs_data for x in xs]

        assert len(all_source_sentences_and_refs_data) == len(data_to_be_translated), f"{len(all_source_sentences_and_refs_data)} vs {len(data_to_be_translated)}"
        assert set(all_source_sentences_and_refs_data) == set(data_to_be_translated)

        env.env_method("reset_fake", increase_reset_times=False)
        env.set_attr("rewards", []) # reset rewards

    return inner_callback_after_eval

def _custom_callback_on_eval(store_model_on_eval, save_path, name_prefix, model, eval_env, logger):
    def f(n_calls, eval_freq):
        if store_model_on_eval:
            store_model(save_path, name_prefix, f"eval-{n_calls // eval_freq}", model, logger)

        if model.get_vec_normalize_env() is not None:
            logger.info("Loading VecNormalize statistics before evaluation onto the eval environment")

            assert isinstance(eval_env, VecNormalizeRangeAndRewardSentenceLevelICL), f"Expected eval_env to be an instance of VecNormalizeRangeAndRewardSentenceLevelICL, but got {type(eval_env)}"

            # Code adapted from sync_envs_normalization
            eval_env.obs_rms = copy.deepcopy(model.get_vec_normalize_env().obs_rms)
            eval_env.ret_rms = copy.deepcopy(model.get_vec_normalize_env().ret_rms)
            eval_env.training = False

    return f

def main(*main_args, **main_kwargs):
    try:
        logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.rl_experiments"), level=logging.DEBUG)

        if len(main_args) == 0:
            logger.info("Provided args (main params): %s %s", main_args, main_kwargs)
        else:
            logger.info("Provided args (sys.argv): %s", sys.argv)

        logger.info("In this script we assume that data (both translation sentences and ICL examples) and kNN elements are shared among all training environments and evaluation environment")

        # args
        src_lang =                           (sys.argv if len(main_args) == 0 else main_args)[1 - (0 if len(main_args) == 0 else 1)].split(':')
        trg_lang =                           (sys.argv if len(main_args) == 0 else main_args)[2 - (0 if len(main_args) == 0 else 1)].split(':')
        file_data =                          (sys.argv if len(main_args) == 0 else main_args)[3 - (0 if len(main_args) == 0 else 1)].split(':')
        file_data_icl_examples =             (sys.argv if len(main_args) == 0 else main_args)[4 - (0 if len(main_args) == 0 else 1)]
        parsed_kwargs = utils.parse_args(sys.argv[5:]) if len(main_args) == 0 else main_kwargs

        assert len(file_data) in (1, 2), f"Expected 1 or 2 file paths for training and dev sets, but got {len(file_data)}"

        initial_parsed_kwargs = dict(parsed_kwargs)

        # parse args
        src_lang_training, src_lang_dev = src_lang if len(src_lang) == 2 else (src_lang[0],) * 2
        trg_lang_training, trg_lang_dev = trg_lang if len(trg_lang) == 2 else (trg_lang[0],) * 2
        file_data_training, file_data_dev = file_data if len(file_data) == 2 else (file_data[0],) * 2

        # read data
        data_to_be_translated_training, data_to_be_translated_dev = [], []
        data_icl_examples = []

        for _file_data, data_to_be_translated in ((file_data_training, data_to_be_translated_training), (file_data_dev, data_to_be_translated_dev)):
            with open(_file_data, "rt") as fd:
                for line in fd:
                    data_to_be_translated.append(line.rstrip("\r\n"))

        with open(file_data_icl_examples, "rt") as fd:
            for line in fd:
                data_icl_examples.append(line.rstrip("\r\n"))

        # default values
        min_conf_debug = False
        #min_conf_debug = True
        num_envs = max(1, int(parsed_kwargs.pop("num_envs", 8)))
        device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
        max_icl_examples = int(parsed_kwargs.get("max_icl_examples", 5))
        disable_eval = bool(int(parsed_kwargs.pop("disable_eval", 0)))
        store_model_on_eval = bool(int(parsed_kwargs.pop("store_model_on_eval", 0)))
        n_steps = int(parsed_kwargs.pop("n_steps", 100))
        ent_coef = float(parsed_kwargs.pop("ent_coef", 0.03755))
        optuna_trial = parsed_kwargs.pop("optuna_trial", None)
        skip_last_eval = parsed_kwargs.pop("skip_last_eval", False) # it will return a reward of 0
        use_vec_normalize = bool(int(parsed_kwargs.pop("use_vec_normalize", 0)))
        subtract_reward_mean = bool(int(parsed_kwargs.pop("subtract_reward_mean", 1)))

        if min_conf_debug:
            logger.warning("min_conf_debug is set to True, which overrides some parameters to make the training faster. DEBUG purpose only!")

        if min_conf_debug:
            num_envs = 10 # TODO remove
            disable_eval = False # TODO remove
            store_model_on_eval = False # TODO remove
            n_steps = 5

        logger.info("n_steps: %d; the model will be trained with %d data points (num_envs * n_steps)", n_steps, num_envs * n_steps)

        if store_model_on_eval:
            logger.info("Model will be stored on each evaluation")

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
        #max_data_entries_dev = 50
        #max_data_entries_dev = 100
        max_data_entries_dev = -1

        if min_conf_debug:
            #max_data_entries = 1 # TODO remove
            max_data_entries = 10 # TODO remove
            #max_data_entries = 50 # TODO remove
            max_data_icl_examples_entries = 10 # TODO remove
            max_data_entries_dev = max_data_entries

        state_representation = parsed_kwargs.get("state_representation", "representation_per_token_with_features")
        parsed_kwargs["device"] = device
        parsed_kwargs["max_icl_examples"] = max_icl_examples
        parsed_kwargs["max_data_entries"] = max_data_entries
        parsed_kwargs["max_data_icl_examples_entries"] = max_data_icl_examples_entries
        parsed_kwargs["state_representation"] = state_representation
        parsed_kwargs["eval_strategy_training"] = parsed_kwargs.get("eval_strategy_training", "chrf2")
        parsed_kwargs["eval_strategy_eval"] = parsed_kwargs.get("eval_strategy_eval", "chrf2")
        parsed_kwargs["repeat_translation_candidates"] = parsed_kwargs.get("repeat_translation_candidates", False)
        parsed_kwargs["repeat_translation_candidates_times"] = int(parsed_kwargs.get("repeat_translation_candidates_times", 0))
        parsed_kwargs["enable_eos_action"] = parsed_kwargs.get("enable_eos_action", False)
        parsed_kwargs["actions_without_replacement"] = parsed_kwargs.get("actions_without_replacement", False) # allow/disallow selecting the same ICL example more than once in the same trajectory
        parsed_kwargs["current_icl_examples_prepend"] = bool(int(parsed_kwargs.get("current_icl_examples_prepend", False)))
        parsed_kwargs["model_hidden_size"] = parsed_kwargs.get("model_hidden_size", 1536)
        data_to_be_translated_training = data_to_be_translated_training[:max_data_entries if max_data_entries > 0 else None]
        data_to_be_translated_dev = data_to_be_translated_dev[:max_data_entries_dev if max_data_entries_dev > 0 else None]
        data_icl_examples = data_icl_examples[:max_data_icl_examples_entries if max_data_icl_examples_entries > 0 else None]
        process_token_time_step = bool(int(parsed_kwargs.get("process_token_time_step", True)))

        if state_representation == "representation_one_hot_representation_time_and_selected_icl_examples":
            process_token_time_step = False

        parsed_kwargs["process_token_time_step"] = process_token_time_step
        parsed_kwargs["num_icl_examples"] = len(data_icl_examples)
        parsed_kwargs["action_representation"] = "discrete_index"

        if use_vec_normalize:
            logger.info("Using VecNormalize for normalizing observations and rewards")
            parsed_kwargs["apply_l2_normalization_state"] = False

        # parallel eval dev

        assert len(data_to_be_translated_dev) >= num_envs

        file_data_per_env = {n: None for n in range(num_envs)}
        bsz = len(data_to_be_translated_dev) // num_envs
        start = 0
        end = bsz
        n_eval_episodes = [0 for _ in range(num_envs)]

        assert (bsz + 1) * num_envs >= len(data_to_be_translated_dev), f"{bsz} * {num_envs} < {len(data_to_be_translated_dev)}"

        for n in range(num_envs):
            file_data_per_env[n] = list(data_to_be_translated_dev[start:end])
            n_eval_episodes[n] = len(file_data_per_env[n])
            start = end
            end += bsz

            assert n_eval_episodes[n] > 0, n

        if start < len(data_to_be_translated_dev):
            first = True
            idx = 0

            while sum(n_eval_episodes) < len(data_to_be_translated_dev):
                if num_envs > 1 and first and n_eval_episodes[n] < n_eval_episodes[n-1]:
                    file_data_per_env[n].extend(data_to_be_translated_dev[start:start+1])
                    n_eval_episodes[n] = len(file_data_per_env[n])
                    start += 1

                    if n_eval_episodes[n] >= n_eval_episodes[n-1]:
                        first = False

                file_data_per_env[idx % num_envs].extend(data_to_be_translated_dev[start:start+1])
                n_eval_episodes[idx % num_envs] = len(file_data_per_env[idx % num_envs])
                idx += 1
                start += 1

        assert sum(n_eval_episodes) == len(data_to_be_translated_dev)

        _all_data = [file_data_per_env[n] for n in range(num_envs)]
        _all_data = sorted([x for xs in _all_data for x in xs])

        assert len(_all_data) == len(data_to_be_translated_dev)
        assert _all_data == sorted(data_to_be_translated_dev)

        logger.info("num_envs: %s, data_to_be_translated_dev: %s, bsz: %s, n_eval_episodes: %s", num_envs, len(data_to_be_translated_dev), bsz, n_eval_episodes)

        # Other values
        filename_time = datetime.now().strftime("%Y%m%d_%H%M")
        #save_freq = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs) # steps
        save_freq = 1e1000 # disabled
        #eval_freq = max(100, len(data_to_be_translated_training) * max_icl_examples // num_envs) # steps (approx. once per epoch)
        #eval_freq = 10000 # steps
        eval_freq = int(parsed_kwargs.pop("eval_freq", 20000)) # steps
        save_path = f"./all_rl_models/rl_models_{filename_time}/"
        name_prefix = f"rl_{filename_time}"
        #monitor_filename = f"{save_path}{name_prefix}_eval.log"
        monitor_filename = None # pickle serialization doesn't allow to have an opened file descriptor (EvalCallback)
        max_episodes_epochs = 100000 # repeat N times (patience-driven environment, so this value might not be used at all)
        max_episodes = len(data_to_be_translated_training) * max_episodes_epochs
        patience = -1 # early stopping patience (number of evals with no improvement; disabled if < 0)
        enable_eval = not disable_eval
        patience = int(parsed_kwargs.pop("patience", 1000000))

        if min_conf_debug:
            #eval_freq = 500 # TODO remove
            #eval_freq = 1000 # steps
            #eval_freq = 2000 # steps
            #eval_freq = 5000 # steps
            eval_freq = 50 # TODO remove
            #eval_freq = 200 # TODO remove
            #patience = 6 #  TODO remove
            #patience = 3 # TODO remove?
            #patience = 100

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
        #vec_env_class = DummyVecEnv # debug
        vec_env_class = SubprocVecEnv
        vec_env_kwargs = {"start_method": "forkserver"} if vec_env_class is SubprocVecEnv else {}
        #batch_size = 256
        batch_size = max(1, int(parsed_kwargs.pop("rl_batch_size", 400)))
        net_arch = parsed_kwargs.pop("net_arch", None)
        gae_lambda = float(parsed_kwargs.pop("gae_lambda", 0.95))
        clip_range = float(parsed_kwargs.pop("clip_range", 0.1))
        gamma = 1.0
        #gamma = 0.99
        #max_steps = 1e100 # fake value due to callback StopTrainingOnMaxEpisodes
    #    max_steps = 10000 # steps while training
        #max_steps = 20000 # steps while training
        #max_steps = 50000 # steps while training
        #max_steps = 100000 # steps while training
        #max_steps = 200000
        max_steps = int(parsed_kwargs.pop("max_steps", 1000000))
        #max_steps = 10000000 # steps while training # TODO remove?
        max_steps += num_envs + 1 # to be sure that the last model is stored after training, given that eval_freq is adjusted by num_envs
        linear_bottleneck = int(parsed_kwargs.pop("linear_bottleneck", 512))
        activation_fn = utils.get_activation_cls(parsed_kwargs.pop("activation_fn", "tanh"))
        redirect_output_filename = parsed_kwargs.pop("redirect_output_filename", None)
        n_epochs = int(parsed_kwargs.pop("n_epochs", 11))
        vf_coef = float(parsed_kwargs.pop("vf_coef", 0.187))
        target_kl = parsed_kwargs.pop("target_kl", None)
        critic_learning_rate = float(parsed_kwargs.pop("learning_rate", 1e-4))
        actor_learning_rate = critic_learning_rate

        if min_conf_debug:
            batch_size = 25 # TODO remove
            max_steps = 200

        assert (n_steps * num_envs) % batch_size == 0, f"Expected n_steps * num_envs to be divisible by batch_size, but got n_steps={n_steps}, num_envs={num_envs}, batch_size={batch_size}"

        total_evaluations = max_steps // eval_freq

        logger.info("Evaluation frequency (steps, adjusted by number of parallel environments): %d // %d = %d (total evaluations: %d)", eval_freq, num_envs, max(1, eval_freq // num_envs), total_evaluations)

        eval_freq = max(1, eval_freq // num_envs)

        if num_envs > 1:
            logger.info("Be aware that the environment will be executed %d time steps, but %d // %d = %d per environment instance due to the number of parallel envinronments", max_steps, max_steps, num_envs, max_steps // num_envs)

        parsed_kwargs_removed_elements = {k: initial_parsed_kwargs[k] for k in initial_parsed_kwargs.keys() if k not in parsed_kwargs.keys()}

        logger.info("parsed_kwargs: %s", parsed_kwargs)
        logger.info("parsed_kwargs_removed_elements: %s", parsed_kwargs_removed_elements)

        env_args = [src_lang_training, trg_lang_training, file_data_training, file_data_icl_examples]
        env_kwargs = {"gym_logger_level": gym.logger.DEBUG, **parsed_kwargs}
        env = vec_env_class([make_env(rank, env_class, list(env_args), dict({"custom_env_id": str(rank), **env_kwargs}), seed=env_seeds[rank], redirect_output_filename=redirect_output_filename) for rank in range(num_envs)], **vec_env_kwargs)
        parsed_kwargs["max_data_entries"] = max_data_entries_dev
        env_eval_dev = Monitor(vec_env_class([make_env(rank, env_eval_dev_class, [src_lang, trg_lang, file_data_per_env[rank], file_data_icl_examples], dict({"gym_logger_level": gym.logger.INFO, "custom_env_id": f"eval_dev_{str(rank)}", "is_eval_env": True, "_parallel_env": True, **parsed_kwargs}), seed=env_seeds[rank], redirect_output_filename=redirect_output_filename) for rank in range(num_envs)], **vec_env_kwargs), filename=monitor_filename, override_existing=True)
        parsed_kwargs["max_data_entries"] = max_data_entries
        env_eval_dev_unwrapped = env_eval_dev.unwrapped

        env_eval_dev_unwrapped.env_method("_init_load_data_and_populate_knn_pool", options={}) # env_eval_dev.get_closest_neighbors_urls() is available

        action_dim = env_eval_dev_unwrapped.get_attr("action_dim")[0]
        state_dim_per_token = env_eval_dev_unwrapped.get_attr("state_dim_per_token")[0]
        state_window_length = env_eval_dev_unwrapped.get_attr("state_window_length")[0]
        state_dim_per_token_time_step = env_eval_dev_unwrapped.get_attr("state_dim_per_token_time_step")[0]
        callbacks = []

        if state_representation == "representation_per_token_with_features":
            n_features = state_dim_per_token * (state_window_length - 1) # -1 due to the action representation which we skip
        elif state_representation in ("representation_last_token_current_and_relative_diff", "representation_mean_plus_last_75_perc_layer_and_relative_diff"):
            n_features = state_dim_per_token * 2 + (state_dim_per_token_time_step if process_token_time_step else 0)
        elif state_representation in ("representation_mean_75_perc_layer", "representation_last_75_perc_layer"):
            n_features = state_dim_per_token + (state_dim_per_token_time_step if process_token_time_step else 0)
        elif state_representation == "representation_one_hot_representation_time_and_selected_icl_examples":
            n_features = state_dim_per_token + (max_icl_examples + 1) + len(data_icl_examples)
        else:
            n_features = 0

        if use_vec_normalize:
            assert state_representation == "representation_one_hot_representation_time_and_selected_icl_examples"

            normalize_kwargs = {"gamma": gamma, "epsilon": 1e-8, "norm_obs": True, "norm_reward": True, "clip_obs": 10.0, "clip_reward": 10.0}
            normalize_kwargs["subtract_reward_mean"] = subtract_reward_mean
            normalize_kwargs["start_idx"] = 1 + (max_icl_examples + 1) # at the beginning: discrete action (avoid duplicates) and time step representation
            normalize_kwargs["offset"] = state_dim_per_token
            env = VecNormalizeRangeAndRewardSentenceLevelICL(env, training=True, **normalize_kwargs)
            # eval env should not be normalized. Training statistics should be used for normalizing the eval env
            ## However, we initialize the eval env with training statistics, but later are updated!
            env_eval_dev = VecNormalizeRangeAndRewardSentenceLevelICL(env_eval_dev, training=False, **normalize_kwargs)
            env_eval_dev_unwrapped = env_eval_dev.unwrapped

            assert isinstance(env_eval_dev, VecNormalizeRangeAndRewardSentenceLevelICL), f"Expected env_eval_dev to be an instance of VecNormalizeRangeAndRewardSentenceLevelICL, but got {type(env_eval_dev)}"
            assert isinstance(env_eval_dev.unwrapped, vec_env_class), f"Expected env_eval_dev.unwrapped to be an instance of {vec_env_class}, but got {type(env_eval_dev.unwrapped.unwrapped)}"

        #net_arch = [512, 128, 32]
        #net_arch = [512, 256, 128]
        #net_arch = [1024, 512, 256]
        if net_arch is None:
            net_arch = {
                #"pi": [512, 256],
                #"vf": [512, 256]
                #"pi": [1024, 512],
                #"vf": [1024, 512]
                "pi": [1024, 1024],
                "vf": [256, 256]
            } # "pi" is actor and "vf" the critic

        logger.info("net_arch: %s, linear_bottleneck: %s, activation_fn: %s", net_arch, linear_bottleneck, activation_fn)

        warmup_steps = 0
        #actor_learning_rate = 1e-3
        #critic_learning_rate = 1e-3
        min_actor_learning_rate = actor_learning_rate
        min_critic_learning_rate = critic_learning_rate

        logger.info("Warmup steps: %d", warmup_steps)

        #min_critic_learning_rate = critic_learning_rate / 10
        min_critic_learning_rate = 0.0
        total_steps = max(int(max_steps / num_envs + 0.5), 1)
        critic_lr_schedule = LinearWithWarmUpLRSchedule(warmup_steps=warmup_steps, initial_lr=critic_learning_rate, total_steps=total_steps, logger=logger, min_lr_polyfit=min_critic_learning_rate, str_id="critic")

        if n_features <= 0:
            features_extractor_class = FlattenExtractor
            #features_extractor_class = NoFlattenExtractor
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
            elif state_representation == "representation_one_hot_representation_time_and_selected_icl_examples":
                step_embeddings = 0
                step_embeddings_dim = 0
            else:
                raise Exception()

            features_extractor_kwargs = {
                "n": n,
                "skip_n": skip_n,
                "state_dim_per_token": state_dim_per_token,
                "check_zeros": True if state_representation == "representation_per_token_with_features" else False,
                "step_embeddings": step_embeddings, # add embeddings for each time step (+1 to avoid error in the model forward for computing next_actions, although the result will be discarded)
                "step_embeddings_dim": step_embeddings_dim,
                "linear_bottleneck": linear_bottleneck,
                "activation_fn": activation_fn,
            }

        avoid_overlapping_action = True
        #layer_norm_input = True
        layer_norm_input = True if linear_bottleneck > 0 else False
        layer_norm_before_activation = True

        model_class = PPO
        model = model_class(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            batch_size=batch_size,
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
                "layer_norm_input": layer_norm_input,
                "layer_norm_before_activation": layer_norm_before_activation,
                "features_extractor_class": features_extractor_class,
                "features_extractor_kwargs": features_extractor_kwargs,
                "share_features_extractor": True,
                "activation_fn": activation_fn,
                "avoid_overlapping_action": avoid_overlapping_action, # It assumes that the first element in the observation is the representation of the source sentence being translated
            },
            gamma=gamma,
            device=device,
            rollout_buffer_kwargs={"process_time_steps": process_token_time_step},
            max_grad_norm=0.5,
            #max_grad_norm=1.0,
            n_steps=n_steps,
            #n_epochs=4,
            n_epochs=n_epochs,
            #ent_coef=0.02,
            #ent_coef=0.0,
            #ent_coef=0.01,
            #ent_coef=0.05,
            ent_coef=ent_coef,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            target_kl=target_kl,
        )

        #assert init_training_episodes < max_episodes

        # Add callbacks
        stop_train_callback = sb3_cb.StopTrainingOnNoModelImprovement(max_no_improvement_evals=patience, min_evals=0, verbose=1) if patience >= 0 else None # early stopping
        # EvalCallback
        ## it returns the average "sum of undiscounted rewards" per episode (https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html)
        ## it does not evaluate the model performance when training finishes (we evaluate below)
        custom_callback_on_eval = _custom_callback_on_eval(store_model_on_eval, save_path, name_prefix, model, env_eval_dev, logger)
        callbacks.append(DelayedEvalCallback(
            env_eval_dev_unwrapped,
            learning_starts=0,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            #n_eval_episodes=1,
            #n_eval_episodes=len(data_to_be_translated_dev),
            n_eval_episodes=[bsz + 1] * num_envs,
            callback_after_eval=stop_train_callback,
            deterministic=True,
            render=False,
            verbose=1,
            predict_kwargs={
                "env_instance": env_eval_dev_unwrapped,
            },
            disable_eval=disable_eval,
            custom_callback_on_eval=custom_callback_on_eval,
            custom_logger=logger,
            evaluate_policy_kwargs={"callback_compute_episode_rewards_and_lengths": create_callback_episode_rewards(n_eval_episodes, optuna_trial=optuna_trial)},
            inner_callback_after_eval=get_callback_after_eval(num_envs, data_to_be_translated_dev, n_eval_episodes),
            skip_vec_normalize_sync=use_vec_normalize, # if use_vec_normalize is True, we skip syncing the VecNormalize statistics between the training and eval envs after each evaluation, as we load the training statistics before each evaluation in the custom_callback_on_eval
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
        if not skip_last_eval:
            ## do not evaluate best_model as the result is already in the log (unless eval is disabled and best model is unknown)
            ## load best model
            model_path = model_path if disable_eval else os.path.join(save_path, "best_model.zip")
            assert utils.file_exists(model_path), f"Model not found: {model_path}"

            ## dev: load model
            logger.info("Loading %s model (dev): %s", "best" if patience >= 0 else "last-step", model_path)
            #logger.info("Loading last-step model (dev): %s", model_path)

            model = model_class.load(
                model_path,
                learning_rate=lambda foo: 100.0, # dummy callable
                lr_schedule=lambda foo: 100.0, # dummy callable
                policy_kwargs={
                    "net_arch": dict(net_arch),
                    "layer_norm_input": layer_norm_input,
                    "layer_norm_before_activation": layer_norm_before_activation,
                    "features_extractor_class": features_extractor_class,
                    "features_extractor_kwargs": features_extractor_kwargs,
                    "share_features_extractor": True,
                    "activation_fn": activation_fn,
                    "avoid_overlapping_action": avoid_overlapping_action,
                },
            )

            ## dev: evaluate and report result
            logger.info("Evaluating dev")

            #mean_reward, std_reward = evaluate_policy(model, env_eval_dev_unwrapped, n_eval_episodes=len(data_to_be_translated_dev))
            mean_reward, std_reward = evaluate_policy(model, env_eval_dev_unwrapped, n_eval_episodes=len(data_to_be_translated_dev),
                predict_kwargs={
                    "env_instance": env_eval_dev_unwrapped,
                },)
        else:
            mean_reward = 0.0
            std_reward = 0.0

        print(f"Mean reward dev: {mean_reward} +/- {std_reward}")
    except optuna.TrialPruned as e:
        raise e
    finally:
        if "env_eval_dev" in locals():
            env_eval_dev.close()

        if "env" in locals():
            env.close()

        if "model" in locals():
            if hasattr(model, "env") and model.env is not None:
                model.env.close()

                del model.env

            del model

        if "callback" in locals():
            del callback

        if "callbacks" in locals():
            del callbacks

        if "env_eval_dev" in locals():
            del env_eval_dev

        if "env" in locals():
            del env

        if "logger" in locals():
            handlers = logger.handlers[:]

            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)

        gc.collect()

    if "mean_reward" not in locals():
        raise Exception("mean_reward not computed")

    return mean_reward * 100.0

if __name__ == "__main__":
    main()
