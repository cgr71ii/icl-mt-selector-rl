
import sys
import json
import time
import pickle
import base64
import random
import datetime
import collections

import utils

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import torch
import numpy as np
import requests
import faiss
from sacrebleu.metrics import CHRF

# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
# https://gymnasium.farama.org/api/env/

# Gym register:
#gym.envs.registration.register(
#     id="mt_env/MTICLEnv-v0",
#     entry_point="mt_env.envs:MTICLEnv",
#     max_episode_steps=None,
#     nondeterministic=True,
#     order_enforce=True,
#     autoreset=False,
#)

class MTICLEnv(gym.Env):
    """Custom Environment for selecting ICL examples for MT using LLMs that follows gym interface."""

    metadata = {"render_modes": ["human"],}

    def logger_wrapper(self, callback, _str, *args, **kwargs):
        d = str(datetime.datetime.now())

        if len(_str) > 0:
            #if hasattr(self, "reset_times") and self.reset_times > 0:
            _str = f"[{d}] [id: {self.custom_env_id}] [{self.episode}:{self.time_step} -> {self.time_step_global}] {_str}"

        callback(_str, *args, **kwargs)

    def __init__(self, src_lang, trg_lang, file_data, file_data_icl_examples, **kwargs):
        super().__init__()

        gym.logger.set_level(utils.dict_or_default(kwargs, "gym_logger_level", gym.logger.INFO))

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.file_data = file_data # format: source<tab>reference
        self.file_data_icl_examples = file_data_icl_examples # format: source<tab>reference
        self.custom_env_id = utils.dict_or_default(kwargs, "custom_env_id", "none")
        self._seed = utils.dict_or_default(kwargs, "_seed", None)

        assert utils.file_exists(self.file_data), self.file_data
        assert utils.file_exists(self.file_data_icl_examples), self.file_data_icl_examples

        self.reset_times = 0
        self.episode = 0
        self.time_step = 0
        self.time_step_global = 0

        # kNN
        self.data_icl_examples = []
        self.knn_callback = utils.dict_or_default(kwargs, "knn_callback", self.get_closest_neighbors_urls)
        self.knn_callback_data_icl_examples = utils.dict_or_default(kwargs, "knn_callback_data_icl_examples", self.data_icl_examples)
        self.is_self_knn_callback = self.knn_callback.__func__ is self.__class__.get_closest_neighbors_urls and self.knn_callback.__self__ is self
        self.knn_api_retrieve = utils.dict_or_default(kwargs, "knn_api_retrieve", None)
        self.knn_api_insert = utils.dict_or_default(kwargs, "knn_api_insert", None)

        assert isinstance(self.knn_callback_data_icl_examples, list), type(self.knn_callback_data_icl_examples)

        #if self.knn_callback is not self.get_closest_neighbors_urls: # each time we call self, a dynamic object is created to wrap the instance, so id() changes each time
        if not self.is_self_knn_callback:
            assert "knn_callback_data_icl_examples" in kwargs
            assert self.knn_callback_data_icl_examples is not self.data_icl_examples, "If knn_callback is not get_closest_neighbors_urls, knn_callback_data_icl_examples must not be the same as self.data_icl_examples"

            self.knn_callback_data_icl_examples = list(self.knn_callback_data_icl_examples) # copy -> new obj

            del kwargs["knn_callback_data_icl_examples"] # self.logger_wrapper message TOO long...

            self.logger_wrapper(gym.logger.debug, "Different kNN callback provided")
        else:
            assert "knn_callback_data_icl_examples" not in kwargs
            assert self.knn_callback_data_icl_examples is self.data_icl_examples, "If knn_callback is get_closest_neighbors_urls, knn_callback_data_icl_examples must be the same as data_icl_examples"

        if self.knn_api_retrieve is not None or self.knn_api_insert is not None:
            assert self.knn_api_retrieve is not None
            assert self.knn_api_insert is not None
            assert self.is_self_knn_callback, "Two different kNN options selected: callback and API"

            self.logger_wrapper(gym.logger.info, "Embeddings retrieved from API kNN (%s) and saturated vectors added to a different kNN (%s)", self.knn_api_retrieve, self.knn_api_insert)

        self.logger_wrapper(gym.logger.info, "Provided arguments: %s", kwargs)

        self.state_window_length = utils.dict_or_default(kwargs, "state_window_length", 4)
        self.state_window_type = utils.dict_or_default(kwargs, "state_window_type", "concatenate")
        self.max_icl_examples = utils.dict_or_default(kwargs, "max_icl_examples", 4)
        self.max_data_entries = utils.dict_or_default(kwargs, "max_data_entries", -1)
        self.max_data_icl_examples_entries = utils.dict_or_default(kwargs, "max_data_icl_examples_entries", -1)
        self.state_representation = utils.dict_or_default(kwargs, "state_representation", "sentence_and_icl_examples")

        assert self.state_representation in ("sentence_and_icl_examples", "sentence_and_actions"), f"Unexpected state representation: {self.state_representation}"

        if self.state_window_type == "concatenate":
            if self.state_representation == "sentence_and_icl_examples" and self.state_window_length > 1:
                self.logger_wrapper(gym.logger.warn, "State window type is 'concatenate' and state window length is greater than 1: %d > 1. Modifying value to 1", self.state_window_length)

                self.state_window_length = 1
            elif self.state_representation == "sentence_and_actions" and self.state_window_length != self.max_icl_examples + 1:
                self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d != self.max_icl_examples + 1 = %d. Modifying value to the latter", self.state_window_length, self.max_icl_examples + 1)

                self.state_window_length = self.max_icl_examples + 1
            elif self.state_window_length < self.max_icl_examples:
                self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d < self.max_icl_examples = %d", self.state_window_length, self.max_icl_examples)

        # API URLs
        self.translate_model_api = utils.dict_or_default(kwargs, "translate_model_api", "http://127.0.0.1:8000/translate")
        self.embedding_single_token_model_api = utils.dict_or_default(kwargs, "embedding_single_token_model_api", "http://127.0.0.1:8000/get_embedding_from_model_embedding_matrix")
        self.embedding_pooling_model_api = utils.dict_or_default(kwargs, "embedding_pooling_model_api", "http://127.0.0.1:8000/get_embedding_pooling")
        self.eval_model_api = utils.dict_or_default(kwargs, "eval_model_api", "http://127.0.0.1:8000/evaluate_comet_22")

        assert isinstance(self.translate_model_api, str), f"translate_model_api: {type(self.translate_model_api)}: {self.translate_model_api}"
        assert isinstance(self.embedding_single_token_model_api, str), f"embedding_single_token_model_api: {type(self.embedding_single_token_model_api)}: {self.embedding_single_token_model_api}"
        assert isinstance(self.embedding_pooling_model_api, str), f"embedding_pooling_model_api: {type(self.embedding_pooling_model_api)}: {self.embedding_pooling_model_api}"
        assert isinstance(self.eval_model_api, str), f"eval_model_api: {type(self.eval_model_api)}: {self.eval_model_api}"

        self.logger_wrapper(gym.logger.debug, "translate_model_api: %s", self.translate_model_api)
        self.logger_wrapper(gym.logger.debug, "embedding_single_token_model_api: %s", self.embedding_single_token_model_api)
        self.logger_wrapper(gym.logger.debug, "embedding_pooling_model_api: %s", self.embedding_pooling_model_api)
        self.logger_wrapper(gym.logger.debug, "eval_model_api: %s", self.eval_model_api)

        ## Other API parameters
        self.embedding_pooling_model_method = utils.dict_or_default(kwargs, "embedding_pooling_model_method", "mean")
        self.embedding_pooling_model_layer = utils.dict_or_default(kwargs, "embedding_pooling_model_layer", -1)

        # Model conf
        self.batch_size = utils.dict_or_default(kwargs, "batch_size", 16)
        self.device = torch.device(utils.dict_or_default(kwargs, "device", "cuda"))
        self.model_hidden_size = utils.dict_or_default(kwargs, "model_hidden_size", 4096) # (former self.max_transformer_output_length) https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json#L9
        self.state_dim = self.model_hidden_size
        self.action_dim = self.model_hidden_size
        self.eos_token_str = utils.dict_or_default(kwargs, "eos_token_str", "</s>")

        if self.state_window_type == "concatenate":
            self.state_window_type_callback = lambda l: np.concatenate(l, axis=0, dtype=np.float32)
            self.state_dim *= self.state_window_length
        elif self.state_window_type == "average":
            self.state_window_type_callback = lambda l: np.mean(l, axis=0, dtype=np.float32)
        elif self.state_window_type == "maxpooling":
            self.state_window_type_callback = lambda l: np.float32(np.max(l, axis=0))
        else:
            raise Exception(f"Given window type is not valid: {self.state_window_type} (valid: {self.valid_state_window_type})")

        # Other
        self.data_already_loaded = False
        self.translation_candidates_exploration_rate = utils.dict_or_default(kwargs, "translation_candidates_exploration_rate", 1.0) # UCB c
        self.translation_candidates_reward_mean_exponential_decay_alpha = utils.dict_or_default(kwargs, "translation_candidates_reward_mean_exponential_decay_alpha", 0.1) # alpha for exponential decay
        self.repeat_translation_candidates = utils.dict_or_default(kwargs, "repeat_translation_candidates", True)
        self.apply_l2_normalization = utils.dict_or_default(kwargs, "apply_l2_normalization", True)
        self.eval_strategy = utils.dict_or_default(kwargs, "eval_strategy", "chrf2")
        self.dimensionality_reduction_factor_state_and_action = utils.dict_or_default(kwargs, "dimensionality_reduction_factor_state_and_action", 1)
        self.add_n_random_saturated_actions = utils.dict_or_default(kwargs, "add_n_random_saturated_actions", 0)
        self.initial_time_sleep = utils.dict_or_default(kwargs, "initial_time_sleep", 5)
        self.prob_add_saturated_action = utils.dict_or_default(kwargs, "prob_add_saturated_action", 0.0)
        self.add_saturated_action_k = utils.dict_or_default(kwargs, "add_saturated_action_k", 1)
        self.add_saturated_action_storage = set()

        assert self.eval_strategy in ("comet-22-da", "chrf2"), self.eval_strategy
        assert self.model_hidden_size % self.dimensionality_reduction_factor_state_and_action == 0, f"Model hidden size {self.model_hidden_size} must be divisible by the dimensionality reduction factor {self.dimensionality_reduction_factor_state_and_action}"
        assert self.state_dim % self.dimensionality_reduction_factor_state_and_action == 0, f"State dimension {self.state_dim} must be divisible by the dimensionality reduction factor {self.dimensionality_reduction_factor_state_and_action}"
        assert self.action_dim % self.dimensionality_reduction_factor_state_and_action == 0, f"Action dimension {self.action_dim} must be divisible by the dimensionality reduction factor {self.dimensionality_reduction_factor_state_and_action}"

        if self.prob_add_saturated_action > 0.0:
            assert self.prob_add_saturated_action <= 1.0, self.prob_add_saturated_action
            #assert self.is_self_knn_callback
            assert self.add_saturated_action_k > 0, self.add_saturated_action_k

            if not self.is_self_knn_callback:
                self.logger_wrapper(gym.logger.warning, "If the environment is vectorized with SubprocVecEnv, using self.prob_add_saturated_action > 0.0 will NOT have any effect")

        if self.dimensionality_reduction_factor_state_and_action > 1:
            self.logger_wrapper(gym.logger.debug, "Dimensionality reduction factor: %d", self.dimensionality_reduction_factor_state_and_action)

        if self.add_n_random_saturated_actions > 0:
            self.logger_wrapper(gym.logger.info, "Be aware that %d random saturated vectors (random values of -1 and 1) are being added: this helps to prevent the actor saturation if self.action_dim is small enough", self.add_n_random_saturated_actions)

            if not self.is_self_knn_callback:
                self.logger_wrapper(gym.logger.warning, "If the environment is vectorized with SubprocVecEnv, using self.add_n_random_saturated_actions > 0 will NOT have any effect")

        self.state_dim = self.state_dim // self.dimensionality_reduction_factor_state_and_action
        self.action_dim = self.action_dim // self.dimensionality_reduction_factor_state_and_action

        # Need to be defined here
        self.saturated_action_embedding = np.ones((self.action_dim,), dtype=np.float32) # not using np.zeros as it is the initialized observation
        self.saturated_action_embedding_name = f"saturated_vector_env_{self.custom_env_id}_special"
        self.saturated_action_embedding_state = np.ones((self.state_dim,), dtype=np.float32)

        if self.apply_l2_normalization:
            self.saturated_action_embedding_state = utils.l2_normalize(self.saturated_action_embedding_state)

        # Env configuration
        self.logger_wrapper(gym.logger.debug, "State and action embedding size: %d %d", self.state_dim, self.action_dim)
        self.logger_wrapper(gym.logger.info, "Model hidden size and EoS token (you may need to specify the correct values according to your LLM): %d %s", self.model_hidden_size, self.eos_token_str)

        # Define action and observation space (embeddings)
        #self.action_space = gym.spaces.Box(-sys.float_info.max, sys.float_info.max, shape=(self.model_hidden_size,))
        #self.observation_space = gym.spaces.Box(-sys.float_info.max, sys.float_info.max, shape=(self.model_hidden_size,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(self.action_dim,)) # input/output model is expected to have tanh in order to be in [-1, 1]
        self.observation_space = gym.spaces.Box(-1., 1., shape=(self.state_dim,)) # input/output model is expected to have tanh in order to be in [-1, 1]

    def preprocess_action(self, action):
        assert isinstance(action, np.ndarray), type(action)
        assert action.shape == (self.action_dim,), f"Expected action shape {(self.action_dim,)}, got {action.shape}"

        action_url, action_url_distance, action_url_idx = self.knn_callback(np.expand_dims(action, axis=0), k=1, add_saturated_action=self.prob_add_saturated_action > 0.0)
        valid_idx = 0

        assert len(action_url) == 1, len(action_url)
        assert len(action_url[0]) == 1, len(action_url[0])

        if self.src_data_overlap_src_icl_examples > 0:
            assert action_url_distance.shape == (1, 2), action_url_distance.shape
            assert action_url_idx.shape == (1, 2), action_url_idx.shape

            if (action_url_idx[0][0] in (-2, -3)) ^ (action_url_idx[0][1] in (-2, -3)):
                valid_idx = 1 if action_url_idx[0][0] in (-2, -3) else 0
        else:
            assert action_url_distance.shape == (1, 1), action_url_distance.shape
            assert action_url_idx.shape == (1, 1), action_url_idx.shape

        action_url = action_url[0][0]
        action_url_distance = action_url_distance[0][valid_idx]
        action_url_idx = np.array([[action_url_idx[0,valid_idx]]])

        assert action_url_idx.shape == (1, 1), action_url_idx.shape

        if self.is_self_knn_callback and self.knn_api_insert is None and self.knn_api_retrieve is None:
            assert action_url == self.icl_example_representation[action_url_idx[0][0]], f"{action_url} vs {self.icl_example_representation[action_url_idx[0][0]]}"
            assert action_url_idx[0][0] == self.icl_example_representation_icl2idx[action_url], f"{action_url_idx[0][0]} vs {self.icl_example_representation_icl2idx[action_url]}"

        return action_url, action_url_idx, action_url_distance

    def step(self, action):
        """
            Action: can be either an embedding (closest existant representation will be obtained) or tab-separated source and target sentences
        """
        if self.reset_times == 0:
            raise Exception("Calling reset() before step() is mandatory")

        terminated, truncated = self.is_done()
        info = {}

        if terminated or truncated:
            observation = self.state_window_type_callback(self.current_state_window)
            reward = 0.0

            assert observation.shape == (self.state_dim,), f"{observation.shape} vs {(self.state_dim,)}"

            return observation, reward, terminated, truncated, info

        self.time_step += 1
        self.time_step_global += 1

        assert isinstance(action, np.ndarray), type(action)
        assert action.shape == (self.action_dim,), f"Expected action shape {(self.action_dim,)}, got {action.shape}"

        # Preprocess action and apply step
        action_url, action_url_idx, action_url_distance = self.preprocess_action(action)
        terminated, truncated, reward, translation = self.apply_step(action_url)
        #representation = self.str2representation[action_url] # former self.get_url_representation(action_url, apply_model=True).squeeze(0)

        assert translation is None or isinstance(translation, list), type(translation)

        if isinstance(translation, list):
            assert len(translation) == 1, len(translation)

            translation = translation[0]

        #assert representation.shape == (self.action_dim,), representation.shape

        #self.last_representation_str.append(representation)
        #self.last_representation_emb.append(action_url)

        #self.rewards.append(reward)
        self.logger_wrapper(gym.logger.info, "Action in time step #%d (reward: %s; distance: %s): %s",
                            self.time_step, reward, action_url_distance, action_url)

        #previous_observation = self.state_window_type_callback(self.current_state_window) # former: before adding the new observation, code which have been removed
        ## ...
        observation = self.state_window_type_callback(self.current_state_window)

        assert observation.shape == (self.state_dim,), f"{observation.shape} vs {(self.state_dim,)}"

        #self.new_observation(observation, list(self.last_representation_str), list(self.last_representation_emb), children_urls, previous_observation)

        _terminated, _truncated = self.is_done()

        assert _terminated == terminated, f"Expected terminated: {_terminated}, got: {terminated}"
        assert _truncated == truncated, f"Expected truncated: {_truncated}, got: {truncated}"

        if terminated or truncated:
            #average_reward = -np.inf if len(self.rewards) == 0 else (sum(self.rewards) / len(self.rewards))

            #self.logger_wrapper(gym.logger.info, "Average reward for %d steps: %s (sum: %s)", self.time_step, average_reward, sum(self.rewards))

            src_sentence, reference = self.data[self.translation_candidate]
            reward_sum = sum(self.translation_candidates_reward_mean_episode)
            reward_steps = len(self.translation_candidates_reward_mean_episode)
            reward_mean = (reward_sum / reward_steps) if reward_steps > 0 else -100.0

            self.logger_wrapper(gym.logger.info, "Result episode:\n    src: %s\n    ref: %s\n     mt: %s", src_sentence, reference, translation)
            self.logger_wrapper(gym.logger.info, "All episodes statistics: {'sum': %s, 'mean': %s, 'last_episode_reward': %s, 'last_episode_steps': %s}", reward_sum, reward_mean, reward, self.time_step)

        sys.stdout.flush()
        sys.stderr.flush()

        return observation, reward, terminated, truncated, info

    def _init_translation_candidate_variables(self):
        # Variables for selecting the translation sentences for the episodes
        self.translation_candidate = -1 # dummy value in order to skip repetition of current ICL example the first time
        self.translation_candidates_selected_episode = np.array([0] * len(self.data)) # N_episode
        self.translation_candidates_selected_consecutive_episode = np.array([0] * len(self.data)) # N'_episode
        self.translation_candidates_reward_mean_exponential_decay_episode = np.array([0.0] * len(self.data)) # Q_episode

        # Other
        self.translation_candidates_reward_mean_episode = []

    def _init_load_data_and_populate_knn_pool(self, options=None):
        assert self.reset_times == 0, f"Expected reset_times to be 0, got {self.reset_times}"
        assert isinstance(options, dict) or isinstance(options, type(None)), f"Options must be a dictionary or None, got {type(options)}: {options}"

        options = {} if options is None else options

        if self.data_already_loaded:
            self.logger_wrapper(gym.logger.warn, "Data already loaded: skipping. This is expected if you manually call _init_load_data_and_populate_knn_pool()")
            return

        self.data_already_loaded = True

        self.logger_wrapper(gym.logger.info, "Init data loading and populate kNN pool")

        self.data = []
        #self.data_icl_examples = []
        self.src_data_overlap_src_icl_examples = 0
        #self.observation_hash_dict = {} # We do not remove this data in _soft_reset because the replay buffer is not reseted after an episode ends
        self.str2representation = {} # Needed for knn search and apply_step (sentences, icl_examples and EOS token)

        # Load data
        self.load_data()

        assert len(self.data) > 0, "Data must not be empty"
        assert len(self.data_icl_examples) > 0, "ICL examples must not be empty"
        assert self.knn_callback_data_icl_examples is self.data_icl_examples or sorted(self.knn_callback_data_icl_examples) == sorted(self.data_icl_examples), "We assume that both instances share the same pool"

        l = gym.logger.warn if self.src_data_overlap_src_icl_examples > 0 else gym.logger.info
        pr = self.src_data_overlap_src_icl_examples * 100 / len(self.data_icl_examples)

        self.logger_wrapper(l, "Source data overlaps with ICL examples source sentences: %d out of %d (%.2f%%)", self.src_data_overlap_src_icl_examples, len(self.data_icl_examples), pr)

        if self.src_data_overlap_src_icl_examples > 0:
            self.logger_wrapper(gym.logger.info, "Given that ICL examples source sentences overlap with the source sentences of the data set, self.get_closest_neighbors_urls will add an "
                                                 "additional embedding for the ICL example source sentences to the embeddings. This is done to remove the ICL example source sentence from "
                                                 "the knn search results, so that the model does not cheat in the translation. If the the translation is not among the retrieved entries in "
                                                 "the knn search, the less similar entry will be removed")

        # Shuffle data in order to avoid model memorization
        if options.get("shuffle_all_data", True):
            self.logger_wrapper(gym.logger.info, "Data shuffled")

            random.shuffle(self.data)
            random.shuffle(self.data_icl_examples)
        else:
            self.logger_wrapper(gym.logger.info, "Data NOT shuffled")

        # Insert all ICL examples in the embeddings index
        ## This should be placed in self._soft_reset if the index changes after each episode (e.g., ICL examples are removed during the episode)
        self.embeddings_index = faiss.IndexFlatL2(self.action_dim)
        #self.embeddings_index = faiss.IndexFlatIP(self.action_dim) # conflict with self.add_n_random_saturated_actions > 0
        self.icl_example_representation = {} # former self.active_urls_representation # idx (insertion order) to icl example
        self.icl_example_representation_icl2idx = {} # former self.active_urls_representation_url2idx

        ## ICL examples
        self.logger_wrapper(gym.logger.info, "Obtaining representations for %d ICL examples", len(self.data_icl_examples))

        representations_str = [f"{src_icl}\t{trg_icl}" for src_icl, trg_icl in self.data_icl_examples]
        representations_emb = self.get_icl_example_representation(self.data_icl_examples)

        assert len(representations_str) == len(representations_emb), f"Expected {len(representations_str)} representations, got {len(representations_emb)}"

        self.insert_embeddings(representations_str, representations_emb)

        if len(representations_str) != len(set(representations_str)):
            self.logger_wrapper(gym.logger.warn, "Duplicate ICL example representations found: %d", len(representations_str) - len(set(representations_str)))

        for _str, emb in zip(representations_str, representations_emb):
            assert emb.shape[0] == self.action_dim, f"Expected embedding shape {self.action_dim}, got {emb.shape[0]} for {_str}"

            self.str2representation[_str] = emb

        time.sleep(self.initial_time_sleep) # wait so ICL examples and EoS token are not mixed due to parallel envs (i.e., num_envs > 1) and service-streamer, raising an error

        ## EoS token (early stopping action)
        self.logger_wrapper(gym.logger.info, "Obtaining representations for EoS token")

        representations_str = [self.eos_token_str]
        representations_emb = self.get_token_representation(representations_str)

        assert len(representations_str) == len(representations_emb), f"Expected {len(representations_str)} representations, got {len(representations_emb)}"

        self.insert_embeddings(representations_str, representations_emb)

        for _str, emb in zip(representations_str, representations_emb):
            assert emb.shape[0] == self.action_dim, f"Expected token shape {self.action_dim}, got {emb.shape[0]} for {_str}"

            self.str2representation[_str] = emb

        time.sleep(self.initial_time_sleep)

        ## Source sentences (do not insert, but add the representation to self.str2representation)
        self.logger_wrapper(gym.logger.info, "Obtaining representations for %d sentences", len(self.data))

        representations_str = [d[0] for d in self.data]
        representations_emb = self.get_translations(representations_str, only_representation=True)

        assert len(representations_str) == len(representations_emb), f"Expected {len(representations_str)} representations, got {len(representations_emb)}"

        for src_sentence, observation in zip(representations_str, representations_emb):
            assert observation.shape[0] == self.model_hidden_size // self.dimensionality_reduction_factor_state_and_action, f"Expected source sentence shape {self.model_hidden_size}, got {observation.shape[0]} for {src_sentence}"

            self.str2representation[src_sentence] = observation

        time.sleep(self.initial_time_sleep)

        ## Random saturated vectors
        self.logger_wrapper(gym.logger.info, "Generating %d random saturated vectors", self.add_n_random_saturated_actions)

        self.random_vectors = [] # stored for external use! they might be deleted eventually from an external actor
        random_vectors_str = []

        for i in range(self.add_n_random_saturated_actions):
            random_vector = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            random_vector[random_vector < 0] = -1.0
            random_vector[random_vector >= 0] = 1.0

            self.random_vectors.append(random_vector)
            random_vectors_str.append(f"random_saturated_{i}")

            self.str2representation[random_vectors_str[-1]] = random_vector

        if len(self.random_vectors) > 0:
            self.random_vectors = np.array(self.random_vectors, dtype=np.float32)

            assert self.random_vectors.shape == (self.add_n_random_saturated_actions, self.action_dim), f"Expected shape {(self.add_n_random_saturated_actions, self.action_dim)}, got {self.random_vectors.shape}"

            self.insert_embeddings(random_vectors_str, self.random_vectors, check_l2_norm=False)

        time.sleep(self.initial_time_sleep)

        ## Special token for detecting unexpected actions
        self.logger_wrapper(gym.logger.info, "Inserting representation for unexpected actions")

        assert len(self.saturated_action_embedding.shape) == 1

        str_key = ','.join(map(str, self.saturated_action_embedding.astype(np.int64).tolist()))

        self.add_saturated_action_storage.add(str_key)

        representations_str = [self.saturated_action_embedding_name]
        representations_emb = np.expand_dims(self.saturated_action_embedding, axis=0).astype(np.float32)

        assert len(representations_str) == len(representations_emb), f"Expected {len(representations_str)} representations, got {len(representations_emb)}"

        self.insert_embeddings(representations_str, representations_emb, check_l2_norm=False)

        for _str, emb in zip(representations_str, representations_emb):
            assert emb.shape[0] == self.action_dim, f"Expected representation shape {self.action_dim}, got {emb.shape[0]} for {_str}"

            self.str2representation[_str] = emb

        self.logger_wrapper(gym.logger.info, "Data loaded and kNN populated")

    def _hard_reset(self, seed=None, options=None):
        super().reset(seed=seed)

        assert isinstance(options, dict) or isinstance(options, type(None)), f"Options must be a dictionary or None, got {type(options)}: {options}"

        options = {} if options is None else options

        self.logger_wrapper(gym.logger.info, "Env reset (hard): episode %d", self.episode)

        self.time_step_global = 0

        self._init_load_data_and_populate_knn_pool(options=options)
        self._init_translation_candidate_variables()

        # soft reset
        if options.get("soft_reset_after_hard_reset", True):
            observation, info = self._soft_reset(seed, {**(options if isinstance(options, dict) else {}), **{"reset_from_hard_reset": True}})
        else:
            self.logger_wrapper(gym.logger.info, "No soft reset after hard reset: returning fake observation")

            info = {}
            _current_state_window = collections.deque(maxlen=self.state_window_length)

            for _ in range(self.state_window_length):
                _current_state_window.append(np.zeros(self.model_hidden_size // self.dimensionality_reduction_factor_state_and_action))

            observation = self.state_window_type_callback(_current_state_window)

        return observation, info

    def _soft_reset(self, seed=None, options=None):
        # Difference with self._hard_reset: we keep all the results from the models to avoid computing them again
        assert isinstance(options, dict) or isinstance(options, type(None)), f"Options must be a dictionary or None, got {type(options)}: {options}"

        options = {} if options is None else options
        is_hard_reset = utils.dict_or_default(options, "reset_from_hard_reset", False)
        is_soft_reset = not is_hard_reset

        if is_soft_reset:
            super().reset(seed=seed)

            self.logger_wrapper(gym.logger.info, "Env reset (soft): episode %d", self.episode)

        info = {}
        self.time_step = 0
        #self.current_translations = 0 # former self.current_downloaded_urls
        self.current_icl_examples = []
        self.current_state_window = collections.deque(maxlen=self.state_window_length)
        #self.rewards = []
        self.early_stopping = False
        self.early_stopping_saturation = False
        #self.last_representation_str = [] # former self.last_downloaded_url_representation_url
        #self.last_representation_emb = [] # former self.last_downloaded_url_representation
        self.current_datetime = datetime.datetime.now()

        for _ in range(self.state_window_length):
            self.current_state_window.append(np.zeros(self.model_hidden_size // self.dimensionality_reduction_factor_state_and_action))

        # Select translation sentence for the episode
        self.translation_candidate = self.get_translation_candidate() # this function must be called at the beginning of each episode

        # Get the observation for the first time step
        src_sentence = self.data[self.translation_candidate][0]
        observation = self.str2representation[src_sentence]

        #if not self.skip_populate_knn:
        #    observation = self.str2representation[src_sentence]
        #else:
        #    if src_sentence in self.str2representation.keys():
        #        observation = self.str2representation[src_sentence]
        #    else:
        #        observation = self.get_translations([src_sentence], only_representation=True)[0]
        #        self.str2representation[src_sentence] = observation

        self.current_state_window.append(observation)

        observation = self.state_window_type_callback(self.current_state_window)

        return observation, info

    def reset(self, seed=None, options=None):
        self.logger_wrapper(gym.logger.debug, "Seed and options: %s %s", seed, options)

        self.episode += 1

        assert isinstance(options, dict) or isinstance(options, type(None)), f"Options must be a dictionary or None, got {type(options)}: {options}"

        options = {} if options is None else options

        assert isinstance(options, dict), f"Options must be a dictionary, got {type(options)}: {options}"

        seed = seed if seed is not None else self._seed if (self.reset_times == 0 and self._seed is not None) else None

        if seed is not None:
            self.logger_wrapper(gym.logger.debug, "Reset seed: %s", seed)

            utils.set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)

        if not options.get("skip_hard_reset", False) and (self.reset_times == 0 or utils.dict_or_default(options, "always_hard_reset", False)):
            observation, info = self._hard_reset(seed=seed, options=options)
        else:
            # After first reset, _soft_reset is the default option if "always_hard_reset" is not defined in options
            observation, info = self._soft_reset(seed=seed, options=options)

        assert observation.shape == (self.state_dim,), f"{observation.shape} vs {(self.state_dim,)}"

        self.reset_times += 1

        return observation, info

    def render(self):
        if self.translation_candidate < 0:
            data = {}
        else:
            data = {
                "src_sentence": self.data[self.translation_candidate][0],
                "reference": self.data[self.translation_candidate][1],
                "icl_examples": self.data_icl_examples,
            }

        data["episode"] = self.episode

        try:
            return json.dumps(data, indent=4)
        except:
            return data

    def close(self):
        pass

    def load_data(self):
        assert self.file_data is not None
        assert self.file_data_icl_examples is not None

        self.logger_wrapper(gym.logger.info, "Loading data")

        src_data_set = set()

        with open(self.file_data, "rt") as fd:
            for idx, url_entry in enumerate(fd, 1):
                # Format: source<tab>reference

                try:
                    entry_data = url_entry.rstrip("\r\n").split('\t')

                    assert len(entry_data) == 2

                    #src_sentence, trg_sentence = entry_data

                    src_data_set.add(entry_data[0])
                    self.data.append(entry_data)
                except Exception as e:
                    self.logger_wrapper(gym.logger.error, "Loading data: error in line #%d", idx)

                    raise e

                if idx % 10000 == 0:
                    self.logger_wrapper(gym.logger.info, "Loading data: %d entries read (%d URLs loaded)", idx, len(self.data))

                if self.max_data_entries > 0 and idx >= self.max_data_entries:
                    break

        self.logger_wrapper(gym.logger.info, "Loading data: finished! %d entries read (%d URLs loaded)", idx, len(self.data))
        self.logger_wrapper(gym.logger.info, "Loading data (ICL examples)")

        with open(self.file_data_icl_examples, "rt") as fd:
            for idx, url_entry in enumerate(fd, 1):
                # Format: source<tab>reference

                try:
                    entry_data = url_entry.rstrip("\r\n").split('\t')

                    assert len(entry_data) == 2

                    #src_sentence, trg_sentence = entry_data

                    if entry_data[0] in src_data_set:
                        self.src_data_overlap_src_icl_examples += 1

                    self.data_icl_examples.append(entry_data)
                except Exception as e:
                    self.logger_wrapper(gym.logger.error, "Loading data: error in line #%d", idx)

                    raise e

                if idx % 10000 == 0:
                    self.logger_wrapper(gym.logger.info, "Loading data: %d entries read (%d URLs loaded)", idx, len(self.data_icl_examples))

                if self.max_data_icl_examples_entries > 0 and idx >= self.max_data_icl_examples_entries:
                    break

        self.logger_wrapper(gym.logger.info, "Loading data (ICL examples): finished! %d entries read (%d URLs loaded)", idx, len(self.data_icl_examples))

    def insert_embeddings(self, urls, embeddings, _index=None, _urls_representation=None, _urls_representation_url2idx=None, update_representation=True, check_l2_norm=True):
        assert isinstance(urls, list), f"Expected urls to be a list, got {type(urls)}: {urls}"
        assert len(urls) > 0, "urls must not be an empty list"
        assert isinstance(urls[0], str), f"Expected urls to be a list of strings, got {type(urls[0])}: {urls[0]}"

        embeddings = utils.embeddings_index_sanity_check(embeddings, last_dimmension_shape=self.action_dim, check_l2_norm=check_l2_norm)

        if self.knn_api_insert is not None:
            assert _index is None
            assert _urls_representation is None
            assert _urls_representation_url2idx is None
            assert update_representation

            payload = []

            for v_str, v_emb in zip(urls, embeddings):
                v_emb_str = pickle.dumps(v_emb)
                v_emb_str = base64.b64encode(v_emb_str).decode() # base64 tensor

                payload.append(('src_sentence', v_str))
                payload.append(('embedding', v_emb_str))
                payload.append(('check_l2_norm', '1' if check_l2_norm else '0'))

            response = requests.post(self.knn_api_insert, data=payload)

            assert response.status_code == 200, f"Response status code is not 200: {response.status_code}"
            assert len(response.text) > 0, f"Response text is empty"

            response_text = json.loads(response.text)

            assert response_text["err"] == "null", f"Response error: {response_text['err']}"

            return

        index = self.embeddings_index if _index is None else _index
        urls_representation = self.icl_example_representation if _urls_representation is None else _urls_representation
        urls_representation_url2idx = self.icl_example_representation_icl2idx if _urls_representation_url2idx is None else _urls_representation_url2idx

        utils.insert_embeddings(urls, embeddings, index, urls_representation, urls_representation_url2idx, self.action_dim, update_representation=update_representation, check_l2_norm=check_l2_norm)

    def get_reward(self, src_sentence, reference, translation=None):
        if isinstance(src_sentence, str):
            src_sentence = [src_sentence]
        if isinstance(reference, str):
            reference = [reference]
        if translation is not None:
            if isinstance(translation, str):
                translation = [translation]

            assert isinstance(translation, list), f"Expected translation to be a list, got {type(translation)}: {translation}"

        assert isinstance(src_sentence, list), f"Expected src_sentence to be a list, got {type(src_sentence)}: {src_sentence}"
        assert isinstance(reference, list), f"Expected reference to be a list, got {type(reference)}: {reference}"

        if translation is None:
            reward = 0.0
        else:
            if self.eval_strategy == "comet-22-da":
                avg_eval_values, single_eval_values = self.comet_eval(src_sentence, translation, reference)
                reward = avg_eval_values
            elif self.eval_strategy == "chrf2":
                score = CHRF().corpus_score(translation, reference)
                reward = score.score / 100.0
            else:
                raise Exception(f"Unknown eval_strategy: {self.eval_strategy}")

            assert 0.0 <= reward <= 1.0, f"Invalid reward value: {reward}"

        return reward

    def comet_eval(self, src, mt, ref):
        assert isinstance(src, list), "Source should be a list"
        assert isinstance(mt, list), "MT should be a list"
        assert isinstance(ref, list), "Reference should be a list"
        assert len(src) == len(mt) == len(ref), "Source, MT, and Reference lists must have the same length"

        url = self.eval_model_api
        batch_size = self.batch_size
        scores = []
        data = [{"src": utils.encode_base64(s), "mt": utils.encode_base64(m), "ref": utils.encode_base64(r)} for s, m, r in zip(src, mt, ref)]

        for idx, batch in enumerate(utils.batchify(data, batch_size)):
            payload = []

            for sample in batch:
                payload.append(('src_sentence', sample["src"]))
                payload.append(('mt_sentence', sample["mt"]))
                payload.append(('ref_sentence', sample["ref"]))

            response = requests.post(url, data=payload)

            assert response.status_code == 200, f"Response status code is not 200 (idx: {idx}): {response.status_code}"
            assert len(response.text) > 0, f"Response text is empty (idx: {idx})"

            response_text = json.loads(response.text)

            assert response_text["err"] == "null", f"Response error (idx: {idx}): {response_text['err']}"

            response_result = [float(d) for d in response_text["ok"]]

            scores.extend(response_result)

        avg = sum(scores) / len(scores)

        return avg, scores

    def get_icl_example_representation(self, icl_examples, numpy=True):
        # format icl_examples: list of lists with two elements: [[src1, trg1], [src2, trg2], ...]

        assert isinstance(icl_examples, list), f"Expected icl_examples to be a list, got {type(icl_examples)}: {icl_examples}"
        assert len(icl_examples) > 0, "ICL examples must not be an empty list"
        assert isinstance(icl_examples[0], list), f"Expected icl_examples to be a list of lists, got {type(icl_examples[0])}: {icl_examples[0]}"
        assert len(icl_examples[0]) == 2, f"Expected each icl example to be a list of two elements, got {len(icl_examples[0])}: {icl_examples[0]}"
        assert isinstance(icl_examples[0][0], str) and isinstance(icl_examples[0][1], str), f"Expected each icl example to be a list of two strings, got {type(icl_examples[0][0])} and {type(icl_examples[0][1])}: {icl_examples[0]}"

        url = self.embedding_pooling_model_api
        batch_size = self.batch_size
        representations = []
        src_sentences = [icl_example[0] for icl_example in icl_examples]
        trg_sentences = [icl_example[1] for icl_example in icl_examples]
        data = [{"src": utils.encode_base64(s), "trg": utils.encode_base64(t)} for s, t in zip(src_sentences, trg_sentences)]

        for idx, batch in enumerate(utils.batchify(data, batch_size)):
            payload = []

            for sample in batch:
                payload.append(('src_lang', self.src_lang))
                payload.append(('trg_lang', self.trg_lang))
                payload.append(('src_sentence', sample["src"]))
                payload.append(('trg_sentence', sample["trg"]))

            payload.append(('pooling', self.embedding_pooling_model_method))
            payload.append(('layer', self.embedding_pooling_model_layer))

            response = requests.post(url, data=payload)

            assert response.status_code == 200, f"Response status code is not 200 (idx: {idx}): {response.status_code}"
            assert len(response.text) > 0, f"Response text is empty (idx: {idx})"

            response_text = json.loads(response.text)

            assert response_text["err"] == "null", f"Response error (idx: {idx}): {response_text['err']}"

            response_result = base64.b64decode(response_text["ok"]) # base64 tensor representation
            response_result = pickle.loads(response_result) # transform to tensor from pickle serialization
            response_result = response_result.detach().cpu() if isinstance(response_result, torch.Tensor) else torch.tensor(response_result)

            representations.append(response_result)

        representations = torch.cat(representations, dim=0)

        assert representations.shape == (len(icl_examples), self.model_hidden_size), f"Representations shape mismatch: {representations.shape} vs {(len(icl_examples), self.model_hidden_size)}"

        if numpy:
            representations = representations.numpy()

        if self.dimensionality_reduction_factor_state_and_action > 1:
            representations = utils.fixed_orthogonal_projection(representations, self.action_dim)

        if self.apply_l2_normalization:
            representations = utils.l2_normalize(representations)

        return representations

    def get_token_representation(self, tokens, numpy=True):
        assert isinstance(tokens, list), f"Expected tokens to be a list, got {type(tokens)}: {tokens}"
        assert len(tokens) > 0, "Tokens must not be an empty list"
        assert isinstance(tokens[0], str), f"Expected tokens to be a list of strings, got {type(tokens[0])}: {tokens[0]}"

        url = self.embedding_single_token_model_api
        batch_size = self.batch_size
        representations = []
        data = [utils.encode_base64(s) for s in tokens]

        for idx, batch in enumerate(utils.batchify(data, batch_size)):
            payload = []

            for sample in batch:
                payload.append(('token', sample))

            response = requests.post(url, data=payload)

            assert response.status_code == 200, f"Response status code is not 200 (idx: {idx}): {response.status_code}"
            assert len(response.text) > 0, f"Response text is empty (idx: {idx})"

            response_text = json.loads(response.text)

            assert response_text["err"] == "null", f"Response error (idx: {idx}): {response_text['err']}"

            response_result = base64.b64decode(response_text["ok"]) # base64 tensor representation
            response_result = pickle.loads(response_result) # transform to tensor from pickle serialization
            response_result = response_result.detach().cpu() if isinstance(response_result, torch.Tensor) else torch.tensor(response_result)

            representations.append(response_result)

        representations = torch.cat(representations, dim=0)

        assert representations.shape == (len(tokens), self.model_hidden_size), f"Representations shape mismatch: {representations.shape} vs {(len(tokens), self.model_hidden_size)}"

        if numpy:
            representations = representations.numpy()

        if self.dimensionality_reduction_factor_state_and_action > 1:
            representations = utils.fixed_orthogonal_projection(representations, self.action_dim)

        if self.apply_l2_normalization:
            representations = utils.l2_normalize(representations)

        return representations

    def get_translations(self, src_sentences, icl_examples=None, only_representation=False, numpy=True):
        # format icl_examples if is not None: list of lists of (optionally) lists with two elements: [[[src11, trg11], [src12, trg12]], [], [[src31, trg31]], ...]
        ## len(icl_examples) must be equal to len(src_sentences)

        assert isinstance(src_sentences, list), f"Expected src_sentences to be a list, got {type(src_sentences)}: {src_sentences}"
        assert len(src_sentences) > 0, "src_sentences must not be an empty list"
        assert isinstance(src_sentences[0], str), f"Expected src_sentences to be a list of strings, got {type(src_sentences[0])}: {src_sentences[0]}"

        url = self.translate_model_api if not only_representation else self.embedding_pooling_model_api
        batch_size = self.batch_size
        translations = []
        data = [{"src_sentence": utils.encode_base64(s), "src_examples": [], "trg_examples": [], "icl_idx_src_sentence": []} for s in src_sentences]

        if icl_examples is not None:
            assert isinstance(icl_examples, list), f"Expected icl_examples to be a list, got {type(icl_examples)}: {icl_examples}"
            assert len(icl_examples) == len(src_sentences), f"ICL examples length mismatch: {len(icl_examples)} vs {len(src_sentences)}"

            for idx, icl_example in enumerate(icl_examples):
                assert isinstance(icl_example, list), f"Expected icl_example to be a list, got {type(icl_example)}: {icl_example} (idx: {idx})"

                for icl_example_data in icl_example:
                    assert isinstance(icl_example_data, list), f"Expected icl_example_data to be a list, got {type(icl_example_data)}: {icl_example_data} (idx: {idx})"
                    assert len(icl_example_data) == 2, f"Expected each icl example data to be a list of two elements, got {len(icl_example_data)}: {icl_example_data} (idx: {idx})"

                    src_example, trg_example = icl_example_data

                    assert isinstance(src_example, str), f"Expected src_example to be a string, got {type(src_example)}: {src_example} (idx: {idx})"
                    assert isinstance(trg_example, str), f"Expected trg_example to be a string, got {type(trg_example)}: {trg_example} (idx: {idx})"

                    data[idx]["src_examples"].append(utils.encode_base64(src_example))
                    data[idx]["trg_examples"].append(utils.encode_base64(trg_example))
                    data[idx]["icl_idx_src_sentence"].append(str(idx + 1)) # server expects index starting from 1

        for idx, batch in enumerate(utils.batchify(data, batch_size)):
            payload = []

            for sample in batch:
                payload.append(('src_lang', self.src_lang))
                payload.append(('trg_lang', self.trg_lang))
                payload.append(('src_sentence', sample["src_sentence"]))

                for src_example, trg_example, icl_idx in zip(sample["src_examples"], sample["trg_examples"], sample["icl_idx_src_sentence"]):
                    payload.append(('src_example', src_example))
                    payload.append(('trg_example', trg_example))
                    payload.append(('icl_idx_src_sentence', icl_idx))

            if only_representation:
                payload.append(('pooling', self.embedding_pooling_model_method))
                payload.append(('layer', self.embedding_pooling_model_layer))

            response = requests.post(url, data=payload)

            assert response.status_code == 200, f"Response status code is not 200 (idx: {idx}): {response.status_code}"
            assert len(response.text) > 0, f"Response text is empty (idx: {idx})"

            response_text = json.loads(response.text)

            assert response_text["err"] == "null", f"Response error (idx: {idx}): {response_text['err']}"

            response_result = response_text["ok"]

            if only_representation:
                response_result = base64.b64decode(response_result) # base64 tensor representation
                response_result = pickle.loads(response_result) # transform to tensor from pickle serialization
                response_result = response_result.detach().cpu() if isinstance(response_result, torch.Tensor) else torch.tensor(response_result)

            translations.append(response_result)

        if only_representation:
            translations = torch.cat(translations, dim=0)

            assert translations.shape == (len(src_sentences), self.model_hidden_size), f"Translations shape mismatch: {translations.shape} vs {(len(src_sentences), self.model_hidden_size)}"

            if numpy:
                translations = translations.numpy()

            if self.dimensionality_reduction_factor_state_and_action > 1:
                translations = utils.fixed_orthogonal_projection(translations, self.action_dim)

            if self.apply_l2_normalization:
                translations = utils.l2_normalize(translations)

        assert len(translations) == len(src_sentences), f"Translations length mismatch: {len(translations)} vs {len(src_sentences)}"

        return translations

    def is_done(self):
        limit_examples = len(self.current_icl_examples) >= self.max_icl_examples
        early_stopping_terminated = self.early_stopping or self.early_stopping_saturation
        terminated = limit_examples or early_stopping_terminated
        truncated = False # we have not defined an artificial termination of the environment

        if self.knn_api_insert is None and self.knn_api_retrieve is None:
            assert len(self.icl_example_representation) == self.embeddings_index.ntotal

        return terminated, truncated

    def get_closest_neighbors_urls(self, proto_actions, k=1, distance_expected_zero=False, get_representations_instead_of_embeddings=True, observations=None,
                                   _index=None, _urls_representation=None, _urls_representation_are_embeddings=False,
                                   remove_overlapping_actions=True, add_saturated_action=False, max_distance_threshold=np.inf, debug=False):
        """
            observations: states from which proto_actions were generated
        """
        #proto_actions = utils.l2_normalize(proto_actions) if self.apply_l2_normalization else proto_actions
        #proto_actions = utils.embeddings_index_sanity_check(proto_actions, last_dimmension_shape=self.action_dim)
        proto_actions = utils.embeddings_index_sanity_check(proto_actions, last_dimmension_shape=self.action_dim, check_l2_norm=False) # check_l2_norm=True -> conflict with self.add_n_random_saturated_actions > 0
        assert isinstance(proto_actions, np.ndarray), f"Expected proto_actions to be a numpy array, got {type(proto_actions)}: {proto_actions}"
        assert len(proto_actions.shape) == 2, f"Expected proto_actions to be a 2D numpy array, got shape {proto_actions.shape}: {proto_actions}"
        assert proto_actions.shape[-1] == self.model_hidden_size // self.dimensionality_reduction_factor_state_and_action, f"Expected proto_actions last dimension to be {self.model_hidden_size // self.dimensionality_reduction_factor_state_and_action}, got {proto_actions.shape[-1]}"
        assert isinstance(k, int), k
        assert k > 0, "k must be greater than 0"

        if add_saturated_action:
            if random.random() < self.prob_add_saturated_action:
                saturated_vector = np.ones_like(proto_actions)
                saturated_vector[proto_actions < 0.0] = -1.0
                str_key = [','.join(map(str, v.astype(np.int64).tolist())) for v in saturated_vector]
                not_seen_str_key = [s not in self.add_saturated_action_storage for s in str_key]
                not_seen_str_key_count = collections.defaultdict(int)
                not_seen_str_key_unique = list(not_seen_str_key)

                for idx, v in enumerate(str_key):
                    not_seen_str_key_count[v] += 1

                    if not_seen_str_key_count[v] > 1:
                        not_seen_str_key_unique[idx] = False

                not_seen_str_key_unique = np.array(not_seen_str_key_unique, dtype=bool)
                saturated_vector = saturated_vector[not_seen_str_key_unique == True]
                n_saturated_vector = (not_seen_str_key_unique == True).sum().item()

                assert len(saturated_vector.shape) == 2, saturated_vector.shape
                assert n_saturated_vector == saturated_vector.shape[0], f"Expected {saturated_vector.shape[0]} unseen saturated vectors, got {n_saturated_vector}"
                #assert self.is_self_knn_callback, "If you want to use saturated actions, knn_callback must be set to self.get_closest_neighbors_urls"

                if n_saturated_vector > 0:
                    saturated_vector_str = [f"saturated_vector_env_{self.custom_env_id}_{(len(self.add_saturated_action_storage) - 1) * self.add_saturated_action_k + idx}" for idx in range(self.add_saturated_action_k * n_saturated_vector)]
                    saturated_vector = np.tile(saturated_vector, (self.add_saturated_action_k, 1)).astype(np.float32)

                    assert len(saturated_vector_str) == self.add_saturated_action_k * n_saturated_vector
                    assert saturated_vector.shape == (self.add_saturated_action_k * n_saturated_vector, self.action_dim), f"Expected shape {(self.add_saturated_action_k, self.action_dim)}, got {saturated_vector.shape}"

                    for s in str_key:
                        self.add_saturated_action_storage.add(s)

                    self.insert_embeddings(saturated_vector_str, saturated_vector, check_l2_norm=False,
                                            _index=None if self.is_self_knn_callback else self.knn_callback.__self__.embeddings_index,
                                            _urls_representation=None if self.is_self_knn_callback else self.knn_callback.__self__.icl_example_representation,
                                            _urls_representation_url2idx=None if self.is_self_knn_callback else self.knn_callback.__self__.icl_example_representation_icl2idx)

                    for _str, emb in zip(saturated_vector_str, saturated_vector):
                        _str2representation = self.str2representation if self.is_self_knn_callback else self.knn_callback.__self__.str2representation

                        assert emb.shape == (self.action_dim,), f"Expected embedding shape {self.action_dim}, got {emb.shape[0]} for {_str}"
                        assert _str not in _str2representation, _str

                        _str2representation[_str] = emb

                self.logger_wrapper(gym.logger.info, "Saturated actions added: %d (total: %d)", self.add_saturated_action_k * n_saturated_vector, self.add_saturated_action_k * (len(self.add_saturated_action_storage) - 1))
#            nbefore = len(self.add_saturated_action_storage)
#
#            for proto_action in proto_actions:
#                assert len(proto_action.shape) == 1
#                assert proto_action.shape[0] == self.action_dim
#
#                if random.random() < self.prob_add_saturated_action:
#                    saturated_vector = np.ones_like(proto_action)
#                    saturated_vector[proto_action < 0.0] = -1.0
#                    str_key = ','.join(map(str, saturated_vector.astype(np.int64).tolist()))
#
#                    #assert self.is_self_knn_callback, "If you want to use saturated actions, knn_callback must be set to self.get_closest_neighbors_urls"
#
#                    if str_key not in self.add_saturated_action_storage:
#                        saturated_vector_str = [f"saturated_vector_env_{self.custom_env_id}_{len(self.add_saturated_action_storage) * self.add_saturated_action_k + idx}" for idx in range(self.add_saturated_action_k)]
#                        saturated_vector = np.tile(saturated_vector, (self.add_saturated_action_k, 1)).astype(np.float32)
#
#                        assert len(saturated_vector_str) == self.add_saturated_action_k
#                        assert saturated_vector.shape == (self.add_saturated_action_k, self.action_dim), f"Expected shape {(self.add_saturated_action_k, self.action_dim)}, got {saturated_vector.shape}"
#
#                        self.add_saturated_action_storage.add(str_key)
#                        self.insert_embeddings(saturated_vector_str, saturated_vector, check_l2_norm=False,
#                                                _index=None if self.is_self_knn_callback else self.knn_callback.__self__.embeddings_index,
#                                                _urls_representation=None if self.is_self_knn_callback else self.knn_callback.__self__.icl_example_representation,
#                                                _urls_representation_url2idx=None if self.is_self_knn_callback else self.knn_callback.__self__.icl_example_representation_icl2idx)
#
#                        for _str, emb in zip(saturated_vector_str, saturated_vector):
#                            assert emb.shape == (self.action_dim,), f"Expected embedding shape {self.action_dim}, got {emb.shape[0]} for {_str}"
#
#                            _str2representation = self.str2representation if self.is_self_knn_callback else self.knn_callback.__self__.str2representation
#
#                            assert _str not in _str2representation
#
#                            _str2representation[_str] = emb
#
#            nafter = len(self.add_saturated_action_storage)
#            ntotal = nafter - nbefore
#
#            if ntotal > 0:
#                self.logger_wrapper(gym.logger.info, "Saturated actions added: %d (total: %d)", ntotal * self.add_saturated_action_k, self.add_saturated_action_k * len(self.add_saturated_action_storage))

        if self.knn_api_retrieve is not None:
            _action = pickle.dumps(proto_actions)
            _action = base64.b64encode(_action).decode() # base64 tensor
            payload = [
                ("embedding", _action),
                ("get_representations_instead_of_embeddings", '1' if get_representations_instead_of_embeddings else '0'),
                ("k", str(k))
            ]
            response = requests.post(self.knn_api_retrieve, data=payload)

            assert response.status_code == 200, f"Response status code is not 200: {response.status_code}"
            assert len(response.text) > 0, f"Response text is empty"

            response_text = json.loads(response.text)

            assert response_text["err"] == "null", f"Response error: {response_text['err']}"

            response_result = base64.b64decode(response_text["ok"]) # base64 tensor representation
            response_result = pickle.loads(response_result) # transform to tensor from pickle serialization
            results = response_result["results"]
            D = response_result["D"]
            I = response_result["I"]

            if not get_representations_instead_of_embeddings:
                results = results.to(self.device)

            return results, D, I

        results = []
        index = self.embeddings_index if _index is None else _index
        urls_representation = self.icl_example_representation if _urls_representation is None else _urls_representation

        assert observations is None, "You only need this argument if you want to reconstruct the index based on previous observations, which is not implemented yet"

        if observations is not None:
            # Create and reconstruct index to perform search using A(provided_state) set and not A(current_state) set (here, capital A is the set of actions)
            pass # Given that the set of actions do not change given a state, we do not need to reconstruct the index here

        extra_k = remove_overlapping_actions and self.src_data_overlap_src_icl_examples > 0

        if extra_k:
            k += 1

        if index.ntotal == 0:
            # Faiss index is empty
            self.logger_wrapper(gym.logger.warn, "Faiss index seems to be empty: %d (sentences in pool: %d)", index.ntotal, len(urls_representation))

        #self.logger_wrapper(gym.logger.debug, "Default representation is %s", self.eos_token_str)

        D, I = index.search(proto_actions, k) # [D]istance, [I]ndex
        expected_shape = (proto_actions.shape[0], k)

        assert D.shape == expected_shape, f"Expected D.shape to be {expected_shape}, got {D.shape}"
        assert I.shape == expected_shape, f"Expected I.shape to be {expected_shape}, got {I.shape}"

        _fake_representation_str = self.saturated_action_embedding_name # Default representation if no hits are found
        _fake_representation = self.str2representation[_fake_representation_str] if _urls_representation_are_embeddings else _fake_representation_str

        # Modify D
        D[I == -1] = -100.0 # Set distance to a negative value for invalid indices
        d_modified_idxs = [(_a, _b) for _a, _b in zip(*np.where(I == -1))] if np.any(I == -1) else []

        # Obtain representations (str) from kNN idxs
        for idx1, (i, d) in enumerate(zip(I, D)):
            overlapping_hits = 0

            results.append([])

            for idx2, (value_idx, value_distance) in enumerate(zip(i, d)):
                if len(results[-1]) >= k:
                    break
                if value_idx < 0:
                    assert i[idx2:] == -1 * np.ones_like(i[idx2:]), f"Expected all remaining indices to be -1, got {i[idx2:]}"

                    break # No more valid indices as -1 values are at the end of the list

                url = urls_representation[value_idx]

                assert isinstance(url, str), f"Expected url to be a string, got {type(url)}: {url}"

                if url != self.eos_token_str and not url.startswith("random_saturated_") and not url.startswith("saturated_vector_env_"):
                    # Check if we need to remove this hit
                    src_icl_example, trg_icl_example = url.split('\t')

                    if extra_k and self.data[self.translation_candidate][0] == src_icl_example:
                        self.logger_wrapper(gym.logger.debug, "Removing overlapping action: %s", src_icl_example)

                        overlapping_hits += 1
                        d[idx2] = -200.0
                        i[idx2] = -2

                        d_modified_idxs.append((idx1, idx2))

                        continue # do not add this entry

                if value_distance > max_distance_threshold and not url.startswith("random_saturated_") and not url.startswith("saturated_vector_env_"):
                    self.logger_wrapper(gym.logger.debug, "Removing distant action: %s (distance: %s > %s)", url, value_distance, max_distance_threshold)

                    d[idx2] = -400.0
                    i[idx2] = -4

                    d_modified_idxs.append((idx1, idx2))

                    continue # do not add this entry

                results[-1].append(url)

            assert len(results[-1]) <= k, f"Expected results[-1] to have at most {k} elements, got {len(results[-1])}"
            assert overlapping_hits <= 1, f"Expected at most one overlapping hit, got {overlapping_hits}: this might happen if same source is repeated in the ICL examples"

            if extra_k and overlapping_hits == 0:
                # Remove the less similar neighbor

                assert d.shape == (k,), f"Expected d to have shape ({k},), got {d.shape}"
                assert len(results[-1]) == k, f"Expected results[-1] to have length {k}, got {len(results[-1])}"

                idx2 = np.argmax(d)
                d[idx2] = -300.0
                i[idx2] = -3

                del results[-1][idx2]

            if len(results[-1]) < (k - (1 if extra_k else 0)):
                # Add items to avoid tensor errors because dimensions don't match
                self.logger_wrapper(gym.logger.debug, "Not enough entries close for entry %d/%d (found: %d): returning %d default representation(s) (%s)", idx1 + 1, len(I), len(results[-1]), k - len(results[-1]), _fake_representation_str)

            while len(results[-1]) < (k - (1 if extra_k else 0)):
                results[-1].append(_fake_representation)

            assert len(results[-1]) == (k - (1 if extra_k else 0))

        if distance_expected_zero:
            for idx1, d1 in enumerate(D):
                for idx2, d2 in enumerate(d1):
                    if (idx1, idx2) in d_modified_idxs:
                        continue

                    if not np.isclose(d2, 0.0):
                        self.logger_wrapper(gym.logger.warn, "Expected distance was 0, but got %s in D[%d][%d]: check https://github.com/facebookresearch/faiss/issues/1272", d2, idx1, idx2)

        if not get_representations_instead_of_embeddings:
            # Get embeddings instead of strings

            assert isinstance(results, list)

            if len(results) > 0:
                assert isinstance(results[0], list)

            all_urls = [q for w in results for q in w] # Flatten list of lists

            if len(all_urls) > 0:
                assert isinstance(all_urls[0], str), f"Expected all_urls to be a list of strings, got {type(all_urls[0])}: {all_urls[0]}"

            if debug:
                self.logger_wrapper(gym.logger.error, "faiss.I (first and last 5): %s ... %s", I[:,:5], I[:,-5:])
                #self.logger_wrapper(gym.logger.error, "faiss.D (min max mean stdev): %s %s %s %s", np.min(D, axis=1), np.max(D, axis=1), np.mean(D, axis=1), np.std(D, ddof=1, axis=1))

            _all_urls_subset = [url for url in all_urls if url not in self.str2representation]
            _all_urls_representation = [] if len(_all_urls_subset) == 0 else self.get_icl_example_representation(_all_urls_subset)

            assert len(_all_urls_subset) == len(_all_urls_representation)
            assert len(_all_urls_subset) == 0, f"This should not happen in this environment: {_all_urls_subset}"

            for _child_url, _child_url_observation in zip(_all_urls_subset, _all_urls_representation):
                assert _child_url_observation.shape == (self.action_dim,)

                self.str2representation[_child_url] = _child_url_observation

            results = [torch.tensor(self.str2representation[url]) for url in all_urls]
            results = torch.stack(results, dim=0).to(self.device)

            assert len(results.shape) == 2, results.shape
            assert results.shape == (proto_actions.shape[0] * (k - (1 if extra_k else 0)), proto_actions.shape[1])

            results = results.reshape((proto_actions.shape[0], k - (1 if extra_k else 0), proto_actions.shape[1])).to(self.device)

        return results, D, I

    def get_translation_candidate(self):
        n = len(self.data)
        repeat = False

        # Repeat current translation candidate?
        if self.repeat_translation_candidates and self.translation_candidate >= 0:
            idx = self.translation_candidate

            assert 0.0 <= self.translation_candidates_reward_mean_exponential_decay_episode[idx] <= 1.0, f"Reward mean exponential decay must be in [0, 1], got {self.translation_candidates_reward_mean_exponential_decay_episode[idx]} for idx {idx}"

            p = (1.0 - self.translation_candidates_reward_mean_exponential_decay_episode[idx]) / (1 + self.translation_candidates_selected_consecutive_episode[idx])
            #p = ((1.0 - self.translation_candidates_reward_mean_exponential_decay_episode[idx]) * np.exp(-1.0 * self.translation_candidates_selected_consecutive_episode[idx])).item()

            assert 0.0 <= p <= 1.0, f"Probability p must be in [0, 1], got {p} for idx {idx}: {self.translation_candidates_reward_mean_exponential_decay_episode} / {self.translation_candidates_selected_consecutive_episode}"

            if random.random() < p:
                src_translation_candidate = self.data[self.translation_candidate][0]

                self.logger_wrapper(gym.logger.info, "Translation candidate (repeating) #%d (p: %.4f): %s", idx, p, src_translation_candidate)

                self.translation_candidates_selected_consecutive_episode[idx] += 1
                repeat = True
            else:
                self.translation_candidates_selected_consecutive_episode[idx] = 0

        translation_candidate = self.translation_candidate if repeat else None

        # Select translation candidate
        if translation_candidate is None:
            # UCB exploration modification
            weights = -1.0 * self.translation_candidates_reward_mean_exponential_decay_episode + \
                        self.translation_candidates_exploration_rate * np.sqrt(np.log(self.episode) / (self.translation_candidates_selected_episode + 1))

            if self.episode == 1:
                assert np.all(weights == 0.0), f"Weights must be zero at the first episode, got {weights}"

                # utils.softmax will return uniform distribution when all values are the same, even if they are zero

            prob_dist = utils.softmax(weights)
            translation_candidate = np.random.choice(n, p=prob_dist)
            src_translation_candidate = self.data[translation_candidate][0]

            self.logger_wrapper(gym.logger.info, "Translation candidate #%d (p: %.4f): %s", translation_candidate, prob_dist[translation_candidate], src_translation_candidate)

        assert 0 <= translation_candidate < n, f"Translation candidate index must be in [0, {n}), got {translation_candidate}"

        self.translation_candidates_selected_episode[translation_candidate] += 1

        return translation_candidate

    def apply_step(self, current_action):
        assert isinstance(current_action, str), f"Expected current_action to be a string, got {type(current_action)}: {current_action}"
        assert len(self.current_icl_examples) < self.max_icl_examples, f"Current length of ICL examples ({self.current_icl_examples}) must be less than max ICL examples ({self.max_icl_examples})"

        if current_action == self.eos_token_str:
            # Early stopping action
            self.logger_wrapper(gym.logger.info, "Early stopping action (%s) received in time step #%d", current_action, self.time_step)

            assert self.early_stopping is False, "Early stopping action already received in this episode"

            self.early_stopping = True
        elif current_action.startswith("random_saturated_") or current_action.startswith("saturated_vector_env_"):
            self.logger_wrapper(gym.logger.info, "Saturated action (%s) received in time step #%d", current_action, self.time_step)

            assert self.early_stopping_saturation is False, "Early stopping action (saturation) already received in this episode"

            self.early_stopping_saturation = True
        else:
            self.current_icl_examples.append(current_action.split('\t'))

            assert len(self.current_icl_examples[-1]) == 2, f"Expected current ICL example to have two elements (source and target), got {len(self.current_icl_examples[-1])}: {self.current_icl_examples[-1]}"

        terminated, truncated = self.is_done()
        reward = 0.0

        if self.early_stopping or self.early_stopping_saturation:
            assert terminated or truncated, f"Early stopping action received but not terminated or truncated: {terminated}, {truncated}"
            assert self.early_stopping ^ self.early_stopping_saturation, f"Only one of early stopping or early stopping saturation should be True: {self.early_stopping}, {self.early_stopping_saturation}"

            if self.early_stopping:
                observation = self.str2representation[self.eos_token_str]
            else:
#                if current_action.startswith("random_saturated_"):
#                    _str2representation = self.str2representation
#                elif current_action.startswith("saturated_vector_env_"):
#                    _str2representation = self.str2representation if self.is_self_knn_callback else self.knn_callback.__self__.str2representation
#                else:
#                    raise Exception(f"Unknown saturated action: {current_action}")
#
#                assert current_action in _str2representation, current_action
#
#                observation = _str2representation[current_action]
                observation = self.saturated_action_embedding_state

            if self.state_window_type == "concatenate" and self.state_representation == "sentence_and_icl_examples":
                assert self.state_window_length == 1, self.state_window_length
                assert self.current_state_window[-1].shape == (self.action_dim,), self.current_state_window[-1].shape
                assert observation.shape == self.current_state_window[-1].shape, observation.shape

                observation += self.current_state_window[-1]

                if self.apply_l2_normalization:
                    observation = utils.l2_normalize(observation)
        else:
            # Update state

            if self.state_representation  == "sentence_and_icl_examples":
                src_sentence = self.data[self.translation_candidate][0]
                observation = self.get_translations([src_sentence], icl_examples=[self.current_icl_examples], only_representation=True)[0]
            elif self.state_representation == "sentence_and_actions":
                icl_example = '\t'.join(self.current_icl_examples[-1])

                assert icl_example in self.str2representation, icl_example

                observation = self.str2representation[icl_example]
            else:
                raise Exception(f"Unknown state representation: {self.state_representation}")

        if self.apply_l2_normalization:
            assert utils.check_l2_normalized(observation), "Observation must be l2 normalized"

        self.current_state_window.append(observation)

        if terminated or truncated:
            # Compute reward
            if self.early_stopping_saturation:
                translation = [f"fake_translation_for_{current_action}"]
                reward = -1.0 # punish
            else:
                src_sentence, reference = self.data[self.translation_candidate]
                translation = self.get_translations([src_sentence], icl_examples=[self.current_icl_examples])[0]
                reward = self.get_reward(src_sentence, reference, translation=translation)

                # Update translation candidate mean reward
                previous_value = self.translation_candidates_reward_mean_exponential_decay_episode[self.translation_candidate]
                self.translation_candidates_reward_mean_exponential_decay_episode[self.translation_candidate] = \
                    previous_value + self.translation_candidates_reward_mean_exponential_decay_alpha * (reward - previous_value)
                self.translation_candidates_reward_mean_episode.append(reward)

            return terminated, truncated, reward, translation

        # Return
        terminated, truncated = self.is_done()
        reward = 0.0
        translation = None

        assert not terminated and not truncated, "Step should not terminate or truncate immediately after applying an action"

        return terminated, truncated, reward, translation

if __name__ == "__main__":
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    file_data = sys.argv[3]
    file_data_icl_examples = sys.argv[4]
    parsed_kwargs = utils.parse_args(sys.argv[5:])

    # Initialize and check environment
    env = MTICLEnv(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)

    check_env(env)
