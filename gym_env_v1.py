
import sys
import json
import time
import pickle
import base64
import random
import datetime
import collections
import statistics

import utils

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import torch
import numpy as np
import requests
import faiss
from sacrebleu.metrics import CHRF
import rank_bm25

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

class ActionBoxSampleFromList(gym.spaces.Box):
    def __init__(self, *args, sample_p=None, sample_list_actions=None, sample_list_actions_is_callable=False, sample_list_actions_args=(), error_when_sampling=False, max_icl_examples=None, remove_overlapping_actions=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_p = sample_p
        self.sample_list_actions = sample_list_actions
        self.sample_list_actions_is_callable = sample_list_actions_is_callable
        self.sample_list_actions_args = sample_list_actions_args
        self.error_when_sampling = error_when_sampling
        self.max_icl_examples = max_icl_examples
        self.remove_overlapping_actions = remove_overlapping_actions
        self.bm25 = None

        assert max_icl_examples is not None

        if self.sample_p is not None:
            assert isinstance(self.sample_p, float), type(self.sample_p)
            assert 0.0 <= self.sample_p <= 1.0, self.sample_p

        if self.sample_list_actions is not None and not self.sample_list_actions_is_callable:
            #assert isinstance(self.sample_list_actions, (list, np.ndarray)), type(self.sample_list_actions)
            assert isinstance(self.sample_list_actions, list), type(self.sample_list_actions)
            assert len(self.sample_list_actions) > 0, len(self.sample_list_actions)
            assert len(self.sample_list_actions[0]) == 2, len(self.sample_list_actions[0])
            assert isinstance(self.sample_list_actions[0][0], str), type(self.sample_list_actions[0][0])
            assert isinstance(self.sample_list_actions[0][1], np.ndarray), type(self.sample_list_actions[0][1])
            assert all(action_emb.shape == self._shape for action_str, action_emb in self.sample_list_actions), f"{len(self.sample_list_actions[0][1].shape)}... vs {self._shape}"

        if self.sample_list_actions is not None:
            corpus = [action_str.split('\t') for action_str, action_emb in self.sample_list_actions]
            self.bm25 = rank_bm25.BM25Okapi([icl_example[0].split() for icl_example in corpus])

    def sample(self, *args, **kwargs):
        assert not self.error_when_sampling, "Sampling from ActionBoxSampleFromList is disabled (error_when_sampling=True)"

        if self.sample_list_actions is not None and self.sample_p is not None and random.random() < self.sample_p:
            if self.sample_list_actions_is_callable:
                action_str, action = self.sample_list_actions(*self.sample_list_actions_args)
            elif len(self.sample_list_actions):
                action_str, action = random.choice(self.sample_list_actions)

            assert action.shape == self._shape, f"{action.shape} vs {self._shape}"

            return action

        return super().sample(*args, **kwargs)

    def sample_bm25(self, *args, time_step=None, src_translation_candidate=None, custom_env_id="none", **kwargs):
        if self.sample_list_actions is not None and self.sample_p is not None and random.random() < self.sample_p:
            #from rank_bm25 import BM25Okapi

            assert self.bm25 is not None
            assert time_step is not None
            assert src_translation_candidate is not None
            assert len(self.sample_list_actions) >= self.max_icl_examples
            assert 0 <= time_step < self.max_icl_examples

            tokenized_query = src_translation_candidate.split()
            scores = self.bm25.get_scores(tokenized_query).tolist()

            assert len(scores) == len(self.sample_list_actions), f"BM25 scores length mismatch: {len(scores)} vs {len(self.sample_list_actions)}"

            top_n_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

            if self.remove_overlapping_actions:
                sentence = self.sample_list_actions[top_n_indices[0]][0].split('\t')[0] # if there are overlapping actions, the first will be an overlapping

                while sentence == src_translation_candidate:
                    del top_n_indices[0] # warning: if many elements are removed, there may be an index out of range
                    del scores[0]

                    sentence = self.sample_list_actions[top_n_indices[0]][0].split('\t')[0]

            idx = top_n_indices[time_step]

            gym.logger.info("[id: %s] [x:%d -> x] Sampling strategy: BM25: translation_candidate (idx: %d): %s", custom_env_id, time_step, idx, src_translation_candidate)
            gym.logger.info("[id: %s] [x:%d -> x] Sampling strategy: BM25: scores: %s ...", custom_env_id, time_step, scores[:self.max_icl_examples])
            gym.logger.info("[id: %s] [x:%d -> x] Sampling strategy: BM25: top_n_indices: %s ...", custom_env_id, time_step, top_n_indices[:self.max_icl_examples])
            gym.logger.info("[id: %s] [x:%d -> x] Sampling strategy: BM25: selected: %s", custom_env_id, time_step, self.sample_list_actions[idx][0])

            return self.sample_list_actions[idx][1]

        return super().sample(*args, **kwargs)

    def __str__(self):
        return "ActionBoxSampleFromList"

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
        self.data_icl_examples_bm25_corpus = []
        self.data_icl_examples_bm25_corpus_tokenized = []
        self.knn_always_add_eos_action = utils.dict_or_default(kwargs, "knn_always_add_eos_action", False) # critic will evaluate when is needed

        # Log args
        initial_sample_list_actions = utils.dict_or_default(kwargs, "initial_sample_list_actions", None)

        if "initial_sample_list_actions" in kwargs:
            del kwargs["initial_sample_list_actions"] # self.logger_wrapper message TOO long...

        self.logger_wrapper(gym.logger.info, "Provided arguments: %s", kwargs)

        self.state_window_length = utils.dict_or_default(kwargs, "state_window_length", 4)
        self.max_icl_examples = utils.dict_or_default(kwargs, "max_icl_examples", 4)
        self.max_data_entries = utils.dict_or_default(kwargs, "max_data_entries", -1)
        self.max_data_icl_examples_entries = utils.dict_or_default(kwargs, "max_data_icl_examples_entries", -1)
        self.state_representation = utils.dict_or_default(kwargs, "state_representation", "model_single_representation")
        self.action_representation = utils.dict_or_default(kwargs, "action_representation", "llm")
        self.action_sampling_strategy = utils.dict_or_default(kwargs, "action_sampling_strategy", "none") # used by td3.py
        self.select_max_icl_examples_randomly = utils.dict_or_default(kwargs, "select_max_icl_examples_randomly", False)
        self.current_max_icl_examples = self.max_icl_examples

        if self.select_max_icl_examples_randomly:
            if self.state_representation != "representation_per_token_with_features":
                self.select_max_icl_examples_randomly = False

                self.logger_wrapper(gym.logger.warn, "self.select_max_icl_examples_randomly set to False because self.state_representation != 'representation_per_token_with_features'")
            else:
                self.logger_wrapper(gym.logger.info, "Each new episode will change the total number of ICL examples to select between 1 and %d", self.max_icl_examples)

        assert self.state_representation in ("model_single_representation", "sentence_and_actions", "model_single_representation+sentence_and_actions", "representation_per_token_with_features"), f"Unexpected state representation: {self.state_representation}"
        assert self.action_sampling_strategy in ("none", "bm25"), self.action_sampling_strategy

        if self.state_representation == "model_single_representation" and self.state_window_length > 1:
            self.logger_wrapper(gym.logger.warn, "State window type is 'concatenate' and state window length is greater than 1: %d > 1. Modifying value to 1", self.state_window_length)

            self.state_window_length = 1
        elif self.state_representation == "sentence_and_actions" and self.state_window_length != self.max_icl_examples + 1:
            self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d != self.max_icl_examples + 1 = %d. Modifying value to the latter", self.state_window_length, self.max_icl_examples + 1)

            self.state_window_length = self.max_icl_examples + 1
        elif self.state_representation == "model_single_representation+sentence_and_actions" and self.state_window_length != self.max_icl_examples + 2:
            self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d != self.max_icl_examples + 2 = %d. Modifying value to the latter", self.state_window_length, self.max_icl_examples + 2)

            self.state_window_length = self.max_icl_examples + 2
        elif self.state_representation == "representation_per_token_with_features" and self.state_window_length < 512 + 3:
            #self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d < self.max_icl_examples = %d. Modifying value to the latter", self.state_window_length, 512 + 1)
            #self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d. Value too low: modifying value to %d", self.state_window_length, 512 + 1)
            self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d. Value too low", self.state_window_length)

            #self.state_window_length = 512 + 1 + 1 + 1 # +1 for the action representation at the beginning and +1 for the state representation regarding self.current_max_icl_examples and +1 for the current step
        elif self.state_window_length < self.max_icl_examples:
            self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d < self.max_icl_examples = %d. Modifying value to the latter", self.state_window_length, self.max_icl_examples)

            self.state_window_length = self.max_icl_examples

        # API URLs
        self.translate_model_api = utils.dict_or_default(kwargs, "translate_model_api", "http://127.0.0.1:8000/translate")
        self.embedding_single_token_model_api = utils.dict_or_default(kwargs, "embedding_single_token_model_api", "http://127.0.0.1:8000/get_embedding_from_model_embedding_matrix")
        self.embedding_pooling_model_api = utils.dict_or_default(kwargs, "embedding_pooling_model_api", "http://127.0.0.1:8000/get_embedding_pooling")
        self.eval_model_api = utils.dict_or_default(kwargs, "eval_model_api", "http://127.0.0.1:8000/evaluate_comet_22")
        self.embedding_external_system = utils.dict_or_default(kwargs, "embedding_external_system", "http://127.0.0.1:8000/get_embedding_from_given_model", f=lambda s: s.rstrip('/'))

        assert isinstance(self.translate_model_api, str), f"translate_model_api: {type(self.translate_model_api)}: {self.translate_model_api}"
        assert isinstance(self.embedding_single_token_model_api, str), f"embedding_single_token_model_api: {type(self.embedding_single_token_model_api)}: {self.embedding_single_token_model_api}"
        assert isinstance(self.embedding_pooling_model_api, str), f"embedding_pooling_model_api: {type(self.embedding_pooling_model_api)}: {self.embedding_pooling_model_api}"
        assert isinstance(self.eval_model_api, str), f"eval_model_api: {type(self.eval_model_api)}: {self.eval_model_api}"

        self.translate_model_api = self.translate_model_api.split('|')
        self.embedding_pooling_model_api = self.embedding_pooling_model_api.split('|')
        self.embedding_external_system = self.embedding_external_system.split('|')

        self.logger_wrapper(gym.logger.debug, "translate_model_api (pool size: %d): %s", len(self.translate_model_api), self.translate_model_api)
        self.logger_wrapper(gym.logger.debug, "embedding_single_token_model_api: %s", self.embedding_single_token_model_api)
        self.logger_wrapper(gym.logger.debug, "embedding_pooling_model_api (pool size: %d): %s", len(self.embedding_pooling_model_api), self.embedding_pooling_model_api)
        self.logger_wrapper(gym.logger.debug, "eval_model_api: %s", self.eval_model_api)

        ## Other API parameters
        self.embedding_pooling_model_method_state = "mean"
        self.embedding_pooling_model_method_action = "mean"
        self.embedding_pooling_model_layer = utils.dict_or_default(kwargs, "embedding_pooling_model_layer", -1)

        #if self.state_representation == "representation_per_token":
        if self.state_representation == "representation_per_token_with_features":
            #self.embedding_pooling_model_method_state = "none"
            self.embedding_pooling_model_method_state = "features"

        # Model conf
        self.batch_size = utils.dict_or_default(kwargs, "batch_size", 16)
        self.device = torch.device(utils.dict_or_default(kwargs, "device", "cuda"))
        self.model_hidden_size = utils.dict_or_default(kwargs, "model_hidden_size", 4096) # (former self.max_transformer_output_length) https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json#L9
        self.model_hidden_size_action_src_sentence = utils.dict_or_default(kwargs, "model_hidden_size_action_src_sentence", self.model_hidden_size, f=int)
        self.model_hidden_size_action_trg_sentence = utils.dict_or_default(kwargs, "model_hidden_size_action_trg_sentence", self.model_hidden_size, f=int)
        self.eos_token_str = utils.dict_or_default(kwargs, "eos_token_str", "</s>")

        if self.state_representation == "representation_per_token_with_features":
            self.state_dim = 4 # 4 features per token: constant, observed, most_likely, entropy
        else:
            self.state_dim = self.model_hidden_size

        if self.action_representation != "llm":
            self.logger_wrapper(gym.logger.info,
                                "Remember to set the correct model_hidden_size_action_{src,trg}_sentence according to the appropriate embedding sizes: %s and %s",
                                self.model_hidden_size_action_src_sentence, self.model_hidden_size_action_trg_sentence)

        if self.action_representation == "llm":
            self.action_dim = self.model_hidden_size
            self.action_representation_src_sentence = "llm"
            self.action_representation_trg_sentence = "llm"

            assert self.model_hidden_size_action_src_sentence == self.model_hidden_size, self.model_hidden_size_action_src_sentence
            assert self.model_hidden_size_action_trg_sentence == self.model_hidden_size, self.model_hidden_size_action_trg_sentence
        elif self.action_representation.startswith("src_embedding:") and ";trg_embedding:" in self.action_representation: # "src_embedding:llm;trg_embedding:llm" is valid but different from "llm"
            self.action_dim = self.model_hidden_size_action_src_sentence + self.model_hidden_size_action_trg_sentence
            self.action_representation_src_sentence = self.action_representation[14:].split(";trg_embedding:")[0]
            self.action_representation_trg_sentence = self.action_representation.split(";trg_embedding:")[1]

            assert self.model_hidden_size_action_src_sentence > 0, self.model_hidden_size_action_src_sentence
            assert self.model_hidden_size_action_trg_sentence > 0, self.model_hidden_size_action_trg_sentence
        elif self.action_representation.startswith("src_embedding:"):
            self.action_dim = self.model_hidden_size_action_src_sentence
            self.model_hidden_size_action_trg_sentence = 0
            self.action_representation_src_sentence = self.action_representation[14:]
            self.action_representation_trg_sentence = None

            assert self.model_hidden_size_action_src_sentence > 0, self.model_hidden_size_action_src_sentence
            assert self.model_hidden_size_action_trg_sentence == 0, self.model_hidden_size_action_trg_sentence
        #elif self.action_representation.startswith("trg_embedding:"): # we always need a method to represent the source sentence
        #    self.action_dim = self.model_hidden_size_action_trg_sentence
        #    self.model_hidden_size_action_src_sentence = 0
        #    self.action_representation_src_sentence = None
        #    self.action_representation_trg_sentence = self.action_representation[14:]
        #
        #    assert self.model_hidden_size_action_src_sentence == 0, self.model_hidden_size_action_src_sentence
        #    assert self.model_hidden_size_action_trg_sentence > 0, self.model_hidden_size_action_trg_sentence
        else:
            raise Exception(f"Action representation not supported: {self.action_representation}")

        assert self.action_representation_src_sentence is not None # We always need a method to represent the source sentence

        # Changes due to state concatenation
        self.state_window_type_callback = lambda l: np.concatenate(l, axis=0, dtype=np.float32)
        self.state_dim_per_token = self.state_dim
        self.state_dim *= self.state_window_length - 1 # -1 because we add the action in the next line
        self.state_dim += self.action_dim # we add the action at the beginning of the state to check overlapping actions with the translation candidate source sentence

        if self.state_representation == "representation_per_token_with_features":
            assert self.action_dim % self.state_dim_per_token == 0, f"{self.action_dim} % {self.state_dim_per_token} != 0" # we need to be sure that action can be split into tokens of state_dim size
            assert self.state_dim % self.state_dim_per_token == 0, f"{self.state_dim} % {self.state_dim_per_token} != 0"

        # Other
        self.data_already_loaded = False
        self.translation_candidates_exploration_rate = utils.dict_or_default(kwargs, "translation_candidates_exploration_rate", 1.0) # UCB c
        self.translation_candidates_reward_mean_exponential_decay_alpha = utils.dict_or_default(kwargs, "translation_candidates_reward_mean_exponential_decay_alpha", 0.1) # alpha for exponential decay
        self.repeat_translation_candidates = utils.dict_or_default(kwargs, "repeat_translation_candidates", False) # reward-based repetition
        self.repeat_translation_candidates_times = utils.dict_or_default(kwargs, "repeat_translation_candidates_times", -1)
        self.repeat_translation_candidates_times_counter = self.repeat_translation_candidates_times
        self.apply_l2_normalization_state = utils.dict_or_default(kwargs, "apply_l2_normalization_state", True)
        self.apply_l2_normalization_action = utils.dict_or_default(kwargs, "apply_l2_normalization_action", True)
        self.eval_strategy_training = utils.dict_or_default(kwargs, "eval_strategy_training", "chrf2")
        self.eval_strategy_eval = utils.dict_or_default(kwargs, "eval_strategy_eval", "chrf2")
        self.initial_time_sleep = utils.dict_or_default(kwargs, "initial_time_sleep", 5)
        self.is_eval_env = utils.dict_or_default(kwargs, "is_eval_env", False)
        self.enable_eos_action = utils.dict_or_default(kwargs, "enable_eos_action", True)
        self.translation_candidate_strategy = utils.dict_or_default(kwargs, "translation_candidate_strategy", "choice_with_replacement")
        self.reward_power = utils.dict_or_default(kwargs, "reward_power", 1)
        self.actions_without_replacement = utils.dict_or_default(kwargs, "actions_without_replacement", False)
        self.best_reward_seen = {}
        self.observation_skip_first_n = utils.dict_or_default(kwargs, "observation_skip_first_n", 0)

        if self.select_max_icl_examples_randomly and self.is_eval_env:
            self.select_max_icl_examples_randomly = False

        assert self.reward_power > 0, self.reward_power

        if not self.enable_eos_action:
            self.knn_always_add_eos_action = False

        if self.state_representation == "representation_per_token_with_features":
            assert self.embedding_pooling_model_method_state == "features", self.embedding_pooling_model_method_state

            if self.apply_l2_normalization_state:
                self.logger_wrapper(gym.logger.warn, "L2 normalization should not be enabled with 'representation_per_token_with_features' state representation: disabling")

                self.apply_l2_normalization_state = False

        self.logger_wrapper(gym.logger.info, "EoS action is %s", "enabled" if self.enable_eos_action else "disabled")

        self.multi_step_eval_strategies = ("actions-bm25",) # reward will be computed for all steps in the episode for these strategies

        assert self.eval_strategy_training in ("api-eval", "chrf2", "actions-bm25"), self.eval_strategy_training
        assert self.eval_strategy_eval in ("api-eval", "chrf2", "actions-bm25"), self.eval_strategy_eval
        assert self.translation_candidate_strategy in ("sequential", "sequential_shuffle_per_epoch", "choice_with_replacement"), self.translation_candidate_strategy

        self.str2representation_valid_actions_k = []

        # Env configuration
        self.logger_wrapper(gym.logger.debug, "State and action embedding size: %d %d", self.state_dim, self.action_dim)
        self.logger_wrapper(gym.logger.info,
                            "Model hidden size and EoS (enabled: %s) token/sentence (you may need to specify the correct values according to your LLM or arbitrary sentence if external embedding model is used): %d %s",
                            self.enable_eos_action, self.model_hidden_size, self.eos_token_str)

        # Define action and observation space (embeddings)
        #self.action_space = gym.spaces.Box(-sys.float_info.max, sys.float_info.max, shape=(self.model_hidden_size,))
        #self.observation_space = gym.spaces.Box(-sys.float_info.max, sys.float_info.max, shape=(self.model_hidden_size,))
        #self.action_space = gym.spaces.Box(-1., 1., shape=(self.action_dim,)) # input/output model is expected to have tanh in order to be in [-1, 1]

        error_when_sampling = False

        if initial_sample_list_actions is not None:
            assert isinstance(initial_sample_list_actions, list), type(initial_sample_list_actions)
            assert len(initial_sample_list_actions) > 0
            assert isinstance(initial_sample_list_actions[0], tuple), type(initial_sample_list_actions[0])
            assert len(initial_sample_list_actions[0]) == 2, len(initial_sample_list_actions[0])
            assert isinstance(initial_sample_list_actions[0][0], str), type(initial_sample_list_actions[0][0])
            assert isinstance(initial_sample_list_actions[0][1], np.ndarray), type(initial_sample_list_actions[0][1])
            assert len(initial_sample_list_actions[0][1].shape) == 1, initial_sample_list_actions[0][1].shape
            assert initial_sample_list_actions[0][1].shape[0] == self.action_dim, initial_sample_list_actions[0][1].shape

            self.logger_wrapper(gym.logger.info, "Loading initial sample list actions with %d entries", len(initial_sample_list_actions))

            _initial_sample_list_actions = [(initial_sample_list_actions[i][0], initial_sample_list_actions[i][1].astype(np.float32)) for i in range(len(initial_sample_list_actions))]
        else:
            #self.logger_wrapper(gym.logger.warning, "Generating initial sample list actions with random normal values: this is not expected by the Wolpertinger policy and should only be used for non-main training environments")

            #_initial_sample_list_actions = [np.random.normal(np.zeros(self.action_dim), 0.1 * np.ones(self.action_dim)).astype(np.float32) for _ in range(1000)] # TODO use parameter
            _initial_sample_list_actions = None
            error_when_sampling = True

        action_space_kwargs = {
            "sample_list_actions": _initial_sample_list_actions,
            "sample_list_actions_is_callable": False, # SubprocVecEnv does pickle environments, and local functions can't be pickled
            "max_icl_examples": self.max_icl_examples,
        }
        self.action_space = ActionBoxSampleFromList(-1., 1., shape=(self.action_dim,), sample_p=1.0, error_when_sampling=error_when_sampling, **action_space_kwargs)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(self.state_dim,)) # input/output model is expected to have tanh in order to be in [-1, 1]

    def get_int_env_id(self):
        try:
            return int(self.custom_env_id)
        except:
            return None

    def preprocess_action(self, action):
        assert isinstance(action, np.ndarray), type(action)
        assert action.shape == (self.action_dim,), f"Expected action shape {(self.action_dim,)}, got {action.shape}"

        translation_candidate_observation = np.zeros((1, self.state_dim), dtype=np.float32) # dummy observation
        translation_candidate_actual_representation = self.str2representation[self.data[self.translation_candidate][0]]
        translation_candidate_observation[0, :translation_candidate_actual_representation.shape[0]] = translation_candidate_actual_representation # self.get_closest_neighbors_urls expects this representation at the very beginning
        action_url, action_url_distance, action_url_idx = self.get_closest_neighbors_urls(np.expand_dims(action, axis=0), k=1, observations=translation_candidate_observation, debug=False, actions_without_replacement=self.actions_without_replacement)

        assert len(action_url) == 1, len(action_url)
        assert len(action_url[0]) == 1, len(action_url[0])
        assert action_url_distance.shape == (1, 1), action_url_distance.shape
        assert action_url_idx.shape == (1, 1), action_url_idx.shape

        action_url = action_url[0][0]
        action_url_distance = action_url_distance[0][0]
        action_url_idx = np.array([[action_url_idx[0,0]]])

        assert action_url_idx.shape == (1, 1), action_url_idx.shape
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
            self.logger_wrapper(gym.logger.error, "Episode was finished but the reward is not being calculated again...")

            for idx in range(self.observation_skip_first_n):
                if idx == 0:
                    # action (for removing the overlapping action, if needed)
                    self.current_state_window.append(np.zeros(self.action_dim))
                else:
                    self.current_state_window.append(np.zeros(self.state_dim_per_token))

            observation = self.state_window_type_callback(self.current_state_window).copy()
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
        self.logger_wrapper(gym.logger.info,
                            "Action in time step #%d (reward: %s; distance: %s; max_icl_examples: %s; translation_candidate: %s): %s",
                            self.time_step, reward, action_url_distance, self.current_max_icl_examples, self.translation_candidate, action_url)

        #previous_observation = self.state_window_type_callback(self.current_state_window) # former: before adding the new observation, code which have been removed
        ## ...
        for idx in range(self.observation_skip_first_n):
            if idx == 0:
                # action (for removing the overlapping action, if needed)
                self.current_state_window.append(np.zeros(self.action_dim))
            else:
                self.current_state_window.append(np.zeros(self.state_dim_per_token))

        observation = self.state_window_type_callback(self.current_state_window).copy()

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
            reward_mean = statistics.mean(self.translation_candidates_reward_mean_episode) if reward_steps > 0 else -100.0
            reward_stdev = statistics.stdev(self.translation_candidates_reward_mean_episode) if reward_steps > 1 else -100.0
            reward_mean_all_episodes = sum(list(self.best_reward_seen.values())) / len(self.best_reward_seen) if len(self.best_reward_seen) > 0 else -100.0

            self.logger_wrapper(gym.logger.info, "Result episode:\n    src: %s\n    ref: %s\n     mt: %s", src_sentence, reference, translation)
            self.logger_wrapper(gym.logger.info, "All episodes statistics: {'sum': %s, 'mean': %s, 'stdev': %s, 'last_episode_reward': %s, 'last_episode_steps': %s, 'max_icl_examples': %s}", reward_sum, reward_mean, reward_stdev, reward, self.time_step, self.current_max_icl_examples)
            self.logger_wrapper(gym.logger.info, "Best mean reward so far across all episodes (%d elements seen): %s", len(self.best_reward_seen), reward_mean_all_episodes)

        sys.stdout.flush()
        sys.stderr.flush()

        return observation, reward ** self.reward_power, terminated, truncated, info

    def _init_translation_candidate_variables(self):
        # Variables for selecting the translation sentences for the episodes
        self.translation_candidate = -1 # dummy value in order to skip repetition of current ICL example the first time
        self.translation_candidates_selected_episode = np.array([0] * len(self.data)) # N_episode
        self.translation_candidates_selected_consecutive_episode = np.array([0] * len(self.data)) # N'_episode

        # Other
        self.translation_candidates_reward_mean_episode = []
        self.translation_candidates_idx = list(range(len(self.data)))

        assert len(self.data) > 0

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
        self.src_sentences_index = faiss.IndexFlatL2(self.action_dim)

        # Load data
        self.load_data()

        assert len(self.data) > 0, "Data must not be empty"
        assert len(self.data_icl_examples) > 0, "ICL examples must not be empty"

        l = gym.logger.warn if self.src_data_overlap_src_icl_examples > 0 else gym.logger.info
        pr = self.src_data_overlap_src_icl_examples * 100 / len(self.data_icl_examples)

        self.logger_wrapper(l, "Source data overlaps with ICL examples source sentences: %d out of %d (%.2f%%)", self.src_data_overlap_src_icl_examples, len(self.data_icl_examples), pr)

        if self.src_data_overlap_src_icl_examples > 0:
            self.logger_wrapper(gym.logger.info,
                                "Given that ICL examples source sentences overlap with the source sentences of the data set, self.get_closest_neighbors_urls will add as "
                                "many additional EoS embeddings as overlapping sentences found, after removing the embedding of the overlapping ICL example source sentences, so "
                                "that the model does not cheat in the translation")

        # Insert all ICL examples in the embeddings index
        ## This should be placed in self._soft_reset if the index changes after each episode (e.g., ICL examples are removed during the episode)
        self.embeddings_index = faiss.IndexFlatL2(self.action_dim)
        #self.embeddings_index = faiss.IndexFlatIP(self.action_dim)
        self.icl_example_representation = {} # former self.active_urls_representation # idx (insertion order) to icl example
        self.icl_example_representation_icl2idx = {} # former self.active_urls_representation_url2idx

        ## ICL examples
        self.logger_wrapper(gym.logger.info, "Obtaining representations for %d ICL examples", len(self.data_icl_examples))

        representations_str = [f"{src_icl}\t{trg_icl}" for src_icl, trg_icl in self.data_icl_examples]
        representations_emb = self.get_action_representation(self.data_icl_examples)

        #np.save("actions.npy", representations_emb)
        #self.logger_wrapper(gym.logger.error, "DONE")
        #time.sleep(100)
        #sys.exit(0)

        assert len(representations_str) == len(representations_emb), f"Expected {len(representations_str)} representations, got {len(representations_emb)}"

        self.insert_embeddings(representations_str, representations_emb)

        if len(representations_str) != len(set(representations_str)):
            self.logger_wrapper(gym.logger.warn, "Duplicate ICL example representations found: %d", len(representations_str) - len(set(representations_str)))

        for _str, emb in zip(representations_str, representations_emb):
            assert emb.shape[0] == self.action_dim, f"Expected embedding shape {self.action_dim}, got {emb.shape[0]} for {_str}"

            self.str2representation[_str] = emb

            self.str2representation_valid_actions_k.append(_str)

        distances = []
        abs_sum_values = []

        for idx1 in range(len(representations_emb)):
            idx2 = idx1 + 1

            abs_sum_values.append(np.sum(np.abs(representations_emb[idx1])))

            while idx2 < len(representations_emb):
                distances.append(np.linalg.norm(representations_emb[idx1] - representations_emb[idx2]))

                idx2 += 1

        distances_max = max(distances)
        distances_min = min(distances)
        distances_avg = sum(distances) / len(distances) if len(distances) > 0 else 0.0
        distances_mean = statistics.mean(distances) if len(distances) > 1 else 0.0
        distances_stdev = statistics.stdev(distances) if len(distances) > 1 else 0.0
        distances_var = statistics.variance(distances) if len(distances) > 1 else 0.0
        abs_sum_values_max = max(abs_sum_values)
        abs_sum_values_min = min(abs_sum_values)
        abs_sum_values_avg = sum(abs_sum_values) / len(abs_sum_values) if len(abs_sum_values) > 0 else 0.0
        abs_sum_values_mean = statistics.mean(abs_sum_values) if len(abs_sum_values) > 1 else 0.0
        abs_sum_values_stdev = statistics.stdev(abs_sum_values) if len(abs_sum_values) > 1 else 0.0
        abs_sum_values_var = statistics.variance(abs_sum_values) if len(abs_sum_values) > 1 else 0.0

        self.logger_wrapper(gym.logger.info, "ICL examples pairwise distances: min %.6f, max %.6f, avg %.6f, mean %.6f, stdev %.6f, var %.6f", distances_min, distances_max, distances_avg, distances_mean, distances_stdev, distances_var)
        self.logger_wrapper(gym.logger.info, "ICL examples abs sum values: min %.6f, max %.6f, avg %.6f, mean %.6f, stdev %.6f, var %.6f", abs_sum_values_min, abs_sum_values_max, abs_sum_values_avg, abs_sum_values_mean, abs_sum_values_stdev, abs_sum_values_var)

        time.sleep(self.initial_time_sleep) # wait so ICL examples and EoS token are not mixed due to parallel envs (i.e., num_envs > 1) and service-streamer, raising an error

        ## EoS token (early stopping action)
        if self.enable_eos_action:
            self.logger_wrapper(gym.logger.info, "Obtaining representations for EoS token")

            representations_str = [self.eos_token_str]
            if self.action_representation in ("llm", "src_embedding:llm"): # dimensionality should match
                representations_emb = self.get_token_representation(representations_str)
            else:
                representations_emb = self.get_action_representation(representations_str, trg_is_empty=True)

            assert len(representations_str) == len(representations_emb), f"Expected {len(representations_str)} representations, got {len(representations_emb)}"

            self.insert_embeddings(representations_str, representations_emb)

            for _str, emb in zip(representations_str, representations_emb):
                assert emb.shape[0] == self.action_dim, f"Expected token shape {self.action_dim}, got {emb.shape[0]} for {_str}"

                self.str2representation[_str] = emb

                self.str2representation_valid_actions_k.append(_str)

            time.sleep(self.initial_time_sleep)

        assert self.embeddings_index.ntotal >= self.max_icl_examples, f"{self.embeddings_index.ntotal} < {self.max_icl_examples}"

        ## Source sentences (do not insert, but add the representation to self.str2representation)
        ### These representation are intenteded to be used as initial first state for the translation candidate in order to detect if some ICL example is selected and results to be the same source sentence as the translation candidate
        #### TODO remove this representation from the state when processing with the transformer? This may be done passing an argument to the transformer implementation (something like skip_first_n_word_embeddings)
        self.logger_wrapper(gym.logger.info, "Obtaining representations for %d sentences", len(self.data))

        representations_str = [d[0] for d in self.data]
        representations_emb = self.get_action_representation(representations_str, trg_is_empty=True)

        assert len(representations_str) == len(representations_emb), f"Expected {len(representations_str)} representations, got {len(representations_emb)}"

        for src_sentence, observation in zip(representations_str, representations_emb):
            assert observation.shape[0] == self.action_dim, f"Expected source sentence shape {self.action_dim}, got {observation.shape[0]} for {src_sentence}"

            self.str2representation[src_sentence] = observation

        ### Insert all source sentence representations in a different index for retrieval during training in order to remove overlapping sentences
        representations_emb = utils.embeddings_index_sanity_check(representations_emb, last_dimmension_shape=self.action_dim, check_l2_norm=self.apply_l2_normalization_action)

        self.src_sentences_index.add(np.array(representations_emb).astype(np.float32))

        time.sleep(self.initial_time_sleep)

        self.close_action_representation_server()

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
        observation, info = self._soft_reset(seed, {**(options if isinstance(options, dict) else {}), **{"reset_from_hard_reset": True}})

        return observation, info

    def _reset_state(self):
        sum_dim = 0

        for idx in range(self.state_window_length):
            if idx == 0:
                # action (for removing the overlapping action, if needed)
                self.current_state_window.append(np.zeros(self.action_dim))
            else:
                self.current_state_window.append(np.zeros(self.state_dim_per_token))

            sum_dim += self.current_state_window[-1].shape[0]

        assert len(self.current_state_window) == self.current_state_window.maxlen
        assert self.current_state_window[0].shape[0] == self.action_dim, f"{self.current_state_window[0].shape[0]} vs {self.action_dim}"
        assert sum_dim == self.state_dim, f"{sum_dim} vs {self.state_dim}"

        for v in self.current_state_window:
            assert np.allclose(v, np.zeros_like(v))

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
        #self.last_representation_str = [] # former self.last_downloaded_url_representation_url
        #self.last_representation_emb = [] # former self.last_downloaded_url_representation
        self.current_datetime = datetime.datetime.now()

        if self.select_max_icl_examples_randomly:
            self.current_max_icl_examples = np.random.randint(1, self.max_icl_examples + 1)

            self.logger_wrapper(gym.logger.info, "Current max. ICL examples: %d (max.: %d)", self.current_max_icl_examples, self.max_icl_examples)

        # Reset state (self.current_state_window)
        self._reset_state()

        # Select translation sentence for the episode
        self.translation_candidate = self.get_translation_candidate() # this function must be called at the beginning of each episode

        # Get the initial observation
        src_sentence = self.data[self.translation_candidate][0]
        action_representation = self.str2representation[src_sentence]

        assert len(action_representation) == self.action_dim

        self.current_state_window[0] = action_representation

        if self.state_representation == "representation_per_token_with_features":
            self.current_state_window[1] = np.ones(self.state_dim_per_token) * self.current_max_icl_examples / 10 # state representing the number of maximum examples that will be selected. 10 as constant so the value is less than 1 (as long as self.max_icl_examples is less than 10)
            self.current_state_window[2] = np.ones(self.state_dim_per_token) * (self.time_step + 1) / 10 # state representing the current step. 10 as constant so the value is less than 1 (as long as the current step is less than 10)

            # Fill the rest of the state window with the token representations of the source sentence

            token_representations = self.get_state_representation([src_sentence])[0]

            assert isinstance(token_representations, np.ndarray), type(token_representations)
            assert len(token_representations.shape) == 1, f"Expected token_representations to be a 1D numpy array, got shape {token_representations.shape}: {token_representations}"
            assert token_representations.shape[0] % self.state_dim_per_token == 0, f"Token representations shape mismatch: {token_representations.shape[0]} vs model_hidden_size {self.state_dim_per_token}"

            token_representations = token_representations.reshape(-1, self.state_dim_per_token) # (seq_len, dim)
            is_zero_vector = (token_representations == 0).all(axis=1)
            sum_zero = is_zero_vector.sum(axis=0).item()
            used_tokens = token_representations.shape[0] - sum_zero
            num_tokens = min(used_tokens, self.state_window_length - 1 - 1 - 1)

            assert used_tokens > 0

            if sum_zero > 0:
                # LLMs use left padding, but we adapt the embeddings so padding is located at right position
                assert (token_representations[-1] == 0).all(axis=0), f"{token_representations.shape}: {token_representations[:5]} ... {token_representations[-5:]}"
                assert (token_representations[used_tokens:] == 0).all(axis=1).all(axis=0)
                assert (token_representations[0] != 0).any(axis=0)
                assert (token_representations[:used_tokens] != 0).any(axis=1).all(axis=0), f"{sum_zero}: {token_representations[:used_tokens]} ... {token_representations[used_tokens:used_tokens + 5]}"

            initial_idx = used_tokens - num_tokens # use the last embeddings instead of the first ones (they are more relevant, as last embeddings in the LLM have information of all the previous ones)

            for i in range(num_tokens):
                self.current_state_window[i + 1 + 1 + 1] = token_representations[i + initial_idx] # +1 for the action and +1 for self.current_max_icl_examples and +1 for the current step

            assert (token_representations[i + initial_idx] != 0).any(axis=0)

            if sum_zero > 0:
                assert (token_representations[i + initial_idx + 1] == 0).all(axis=0)

            self.logger_wrapper(gym.logger.debug, "representation_per_token_with_features: %s tokens (used: min(%d, %d))", token_representations.shape, used_tokens, self.state_window_length - 1 - 1 - 1)
            self.logger_wrapper(gym.logger.debug, "First ... last tokens: %s %s %s %s ... %s %s %s %s",
                                self.current_state_window[1], self.current_state_window[2], self.current_state_window[3], self.current_state_window[4],
                                self.current_state_window[-4], self.current_state_window[-3], self.current_state_window[-2], self.current_state_window[-1])

        for idx in range(self.observation_skip_first_n):
            if idx == 0:
                # action (for removing the overlapping action, if needed)
                self.current_state_window.append(np.zeros(self.action_dim))
            else:
                self.current_state_window.append(np.zeros(self.state_dim_per_token))

        observation = self.state_window_type_callback(self.current_state_window).copy()

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
                    self.data_icl_examples_bm25_corpus.append(entry_data[0])
                    self.data_icl_examples_bm25_corpus_tokenized.append(self.data_icl_examples_bm25_corpus[-1].split())
                except Exception as e:
                    self.logger_wrapper(gym.logger.error, "Loading data: error in line #%d", idx)

                    raise e

                if idx % 10000 == 0:
                    self.logger_wrapper(gym.logger.info, "Loading data: %d entries read (%d URLs loaded)", idx, len(self.data_icl_examples))

                if self.max_data_icl_examples_entries > 0 and idx >= self.max_data_icl_examples_entries:
                    break

        self.logger_wrapper(gym.logger.info, "Loading data (ICL examples): finished! %d entries read (%d URLs loaded)", idx, len(self.data_icl_examples))

        self.data_icl_examples_bm25 = rank_bm25.BM25Okapi(self.data_icl_examples_bm25_corpus_tokenized)

    def insert_embeddings(self, urls, embeddings, _index=None, _urls_representation=None, _urls_representation_url2idx=None, update_representation=True, check_l2_norm=True):
        assert isinstance(urls, list), f"Expected urls to be a list, got {type(urls)}: {urls}"
        assert len(urls) > 0, "urls must not be an empty list"
        assert isinstance(urls[0], str), f"Expected urls to be a list of strings, got {type(urls[0])}: {urls[0]}"

        embeddings = utils.embeddings_index_sanity_check(embeddings, last_dimmension_shape=self.action_dim, check_l2_norm=check_l2_norm)
        index = self.embeddings_index if _index is None else _index
        urls_representation = self.icl_example_representation if _urls_representation is None else _urls_representation
        urls_representation_url2idx = self.icl_example_representation_icl2idx if _urls_representation_url2idx is None else _urls_representation_url2idx

        utils.insert_embeddings(urls, embeddings, index, urls_representation, urls_representation_url2idx, self.action_dim, update_representation=update_representation, check_l2_norm=check_l2_norm)

    def get_reward(self, src_sentence, reference, translation=None, icl_examples=None):
        if isinstance(src_sentence, str):
            src_sentence = [src_sentence]
        if isinstance(reference, str):
            reference = [reference]
        if translation is not None:
            if isinstance(translation, str):
                translation = [translation]

            assert isinstance(translation, list), f"Expected translation to be a list, got {type(translation)}: {translation}"
        if icl_examples is not None:
            assert isinstance(icl_examples, list), type(icl_examples)

            if len(icl_examples) > 0:
                assert isinstance(icl_examples[0], list), f"Expected icl_examples to be a list of lists, got {type(icl_examples[0])}: {icl_examples[0]}"
                assert len(icl_examples[0]) == 2, f"Expected each icl example to be a list of two elements, got {len(icl_examples[0])}: {icl_examples[0]}"
                assert isinstance(icl_examples[0][0], str) and isinstance(icl_examples[0][1], str), f"Expected each icl example to be a list of two strings, got {type(icl_examples[0][0])} and {type(icl_examples[0][1])}: {icl_examples[0]}"

        assert isinstance(src_sentence, list), f"Expected src_sentence to be a list, got {type(src_sentence)}: {src_sentence}"
        assert isinstance(reference, list), f"Expected reference to be a list, got {type(reference)}: {reference}"

        eval_strategy = self.eval_strategy_eval if self.is_eval_env else self.eval_strategy_training

        if translation is None and eval_strategy not in self.multi_step_eval_strategies:
            reward = 0.0
        else:
            if eval_strategy == "api-eval":
                avg_eval_values, single_eval_values = self.api_eval(src_sentence, translation, reference)
                reward = avg_eval_values * 100.0 # scale to 0--100
            elif eval_strategy == "chrf2":
                assert len(translation) == 1, len(translation)
                assert len(reference) == 1, len(reference)
                assert isinstance(translation[0], str), type(translation[0])
                assert isinstance(reference[0], str), type(reference[0])

                score = CHRF().sentence_score(translation[0], reference) # expected format: (hypothesis, [reference]) (string, list of strings)
                reward = score.score # scale is 0--100
            elif eval_strategy == "actions-bm25":
                reward = self.get_score_from_icl_example_bm25()
            else:
                raise Exception(f"Unknown eval_strategy ({'eval' if self.is_eval_env else 'training'}): {eval_strategy}")

        return reward

    def api_eval(self, src, mt, ref):
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

            response = utils.requests_post(url, data=payload)

            assert response.status_code == 200, f"Response status code is not 200 (idx: {idx}): {response.status_code}"
            assert len(response.text) > 0, f"Response text is empty (idx: {idx})"

            response_text = json.loads(response.text)

            assert response_text["err"] == "null", f"Response error (idx: {idx}): {response_text['err']}"

            response_result = [float(d) for d in response_text["ok"]]

            scores.extend(response_result)

        avg = sum(scores) / len(scores)

        return avg, scores

    def _get_action_representation_llm(self, idx, batch, src_is_empty=False, trg_is_empty=False):
        assert not (src_is_empty and trg_is_empty), "At least one of src_is_empty or trg_is_empty must be False"

        payload = []
        url = self.embedding_pooling_model_api
        url_idx = self.get_int_env_id()
        url_idx = 0 if url_idx is None else url_idx % len(url)
        url = url[url_idx]

        for sample in batch:
            if src_is_empty:
                # source sentence cannot be empty, so let's hack the trg sentence as src sentence (and swap src and trg languages)
                assert sample["src"] is None, sample["src"]
                assert sample["trg"] is not None, sample["trg"]

                payload.append(('src_lang', self.trg_lang))
                payload.append(('trg_lang', self.src_lang))
                payload.append(('src_sentence', sample["trg"]))
            else:
                assert isinstance(sample["src"], str), type(sample["src"])

                payload.append(('src_lang', self.src_lang))
                payload.append(('trg_lang', self.trg_lang))
                payload.append(('src_sentence', sample["src"]))

            if trg_is_empty:
                assert sample["trg"] is None, sample["trg"]
            elif not src_is_empty: # if src_is_empty, trg sentence is used as src sentence
                assert isinstance(sample["trg"], str), type(sample["trg"])

                payload.append(('trg_sentence', sample["trg"]))

        payload.append(('pooling', self.embedding_pooling_model_method_action))
        payload.append(('layer', self.embedding_pooling_model_layer))

        response = utils.requests_post(url, data=payload)

        assert response.status_code == 200, f"Response status code is not 200 (idx: {idx}): {response.status_code}"
        assert len(response.text) > 0, f"Response text is empty (idx: {idx})"

        response_text = json.loads(response.text)

        assert response_text["err"] == "null", f"Response error (idx: {idx}): {response_text['err']}"

        response_result = base64.b64decode(response_text["ok"]) # base64 tensor representation
        response_result = pickle.loads(response_result) # transform to tensor from pickle serialization
        response_result = response_result.detach().cpu() if isinstance(response_result, torch.Tensor) else torch.tensor(response_result)

        return response_result

    def _get_action_representation_external(self, idx, batch, lang, embedding_name, batch_side_key):
        assert batch_side_key in ("src", "trg"), f"Invalid batch_side_key: {batch_side_key}"

        if embedding_name == "llm":
            batch = list(batch)

            for idx in range(len(batch)):
                if batch_side_key == "src":
                    batch[idx]["trg"] = None
                elif batch_side_key == "trg":
                    batch[idx]["src"] = None

            return self._get_action_representation_llm(idx, batch, src_is_empty=False if batch_side_key == "src" else True, trg_is_empty=False if batch_side_key == "trg" else True)

        payload = []
        url = self.embedding_external_system
        url_idx = self.get_int_env_id()
        url_idx = 0 if url_idx is None else url_idx % len(url)
        url = url[url_idx]

        for sample in batch:
            payload.append(('lang', lang))
            payload.append(('name', embedding_name))
            payload.append(('sentence', sample[batch_side_key]))

        response = utils.requests_post(url, data=payload)

        assert response.status_code == 200, f"Response status code is not 200 (idx: {idx}): {response.status_code}"
        assert len(response.text) > 0, f"Response text is empty (idx: {idx})"

        response_text = json.loads(response.text)

        assert response_text["err"] == "null", f"Response error (idx: {idx}): {response_text['err']}"

        response_result = base64.b64decode(response_text["ok"]) # base64 tensor representation
        response_result = pickle.loads(response_result) # transform to tensor from pickle serialization
        response_result = response_result.detach().cpu() if isinstance(response_result, torch.Tensor) else torch.tensor(response_result)

        return response_result

    def close_action_representation_server(self):
        payload = []

        for name in (self.action_representation_src_sentence, self.action_representation_trg_sentence):
            if name is not None and name != "llm":
                payload.append(("name", name))

        if len(payload) == 0:
            return

        self.logger_wrapper(gym.logger.info, "Server for external actions closed: %s", payload)

        url = [f"{u}_close" for u in self.embedding_external_system]
        url_idx = self.get_int_env_id()
        url_idx = 0 if url_idx is None else url_idx % len(url)
        url = url[url_idx]
        response = utils.requests_post(url, data=payload)

        assert response.status_code == 200, f"Response status code is not 200: {response.status_code}"
        assert len(response.text) > 0, f"Response text is empty"

        response_text = json.loads(response.text)

        assert response_text["err"] == "null", f"Response error: {response_text['err']}"

    def get_action_representation(self, icl_examples, numpy=True, trg_is_empty=False):
        # format icl_examples: list of lists with two elements: [[src1, trg1], [src2, trg2], ...]

        assert isinstance(icl_examples, list), f"Expected icl_examples to be a list, got {type(icl_examples)}: {icl_examples}"
        assert len(icl_examples) > 0, "ICL examples must not be an empty list"

        if trg_is_empty:
            assert isinstance(icl_examples[0], str), f"Expected icl example to be a list of a single string, got {type(icl_examples[0])}"
        else:
            assert isinstance(icl_examples[0], list), f"Expected icl_examples to be a list of lists, got {type(icl_examples[0])}: {icl_examples[0]}"
            assert len(icl_examples[0]) == 2, f"Expected each icl example to be a list of two elements, got {len(icl_examples[0])}: {icl_examples[0]}"
            assert isinstance(icl_examples[0][0], str) and isinstance(icl_examples[0][1], str), f"Expected each icl example to be a list of two strings, got {type(icl_examples[0][0])} and {type(icl_examples[0][1])}: {icl_examples[0]}"

        src_embedding_model = self.action_representation_src_sentence
        trg_embedding_model = self.action_representation_trg_sentence
        batch_size = self.batch_size
        representations = []

        if trg_is_empty:
            src_sentences = list(icl_examples)
            data = [{"src": utils.encode_base64(s), "trg": None} for s in src_sentences]
        else:
            src_sentences = [icl_example[0] for icl_example in icl_examples]
            trg_sentences = [icl_example[1] for icl_example in icl_examples]
            data = [{"src": utils.encode_base64(s), "trg": utils.encode_base64(t)} for s, t in zip(src_sentences, trg_sentences)]

        for idx, batch in enumerate(utils.batchify(data, batch_size)):
            assert isinstance(batch, list), f"Expected batch to be a list, got {type(batch)}: {batch}"

            if src_embedding_model == "llm" and trg_embedding_model == "llm":
                response_result = self._get_action_representation_llm(idx, batch, trg_is_empty=trg_is_empty)

                representations.append(response_result)
            else:
                response_result_src = self._get_action_representation_external(idx, batch, self.src_lang, src_embedding_model, "src")

                assert response_result_src.shape[1] == self.model_hidden_size_action_src_sentence, f"Source representation shape mismatch: {response_result_src.shape} vs model_hidden_size_action_src_sentence {self.model_hidden_size_action_src_sentence}"

                if not trg_is_empty and trg_embedding_model is not None:
                    response_result_trg = self._get_action_representation_external(idx, batch, self.trg_lang, trg_embedding_model, "trg")

                    assert response_result_trg.shape[1] == self.model_hidden_size_action_trg_sentence, f"Target representation shape mismatch: {response_result_trg.shape} vs model_hidden_size_action_trg_sentence {self.model_hidden_size_action_trg_sentence}"
                    assert response_result_trg.shape[0] == response_result_src.shape[0], f"Source and target representation batch size mismatch: {response_result_src.shape[0]} vs {response_result_trg.shape[0]}"

                if trg_is_empty or trg_embedding_model is None:
                    representations.append(response_result_src)
                else:
                    combined_representation = torch.cat([response_result_src, response_result_trg], dim=1) # concatenate along the feature dimension

                    representations.append(combined_representation)

        representations = torch.cat(representations, dim=0)

        assert representations.shape == (len(icl_examples), self.action_dim), f"Representations shape mismatch: {representations.shape} vs {(len(icl_examples), self.action_dim)}"

        if numpy:
            representations = representations.numpy()

        if self.apply_l2_normalization_action:
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

            response = utils.requests_post(url, data=payload)

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

        if self.apply_l2_normalization_action:
            representations = utils.l2_normalize(representations)

        return representations

    def get_state_representation(self, *args, **kwargs):
        assert "only_representation" not in kwargs, "get_state_representation does not accept only_representation argument"

        return self.get_translations(*args, only_representation=True, **kwargs)

    def get_translations(self, src_sentences, icl_examples=None, only_representation=False, numpy=True, api_idx=None):
        # format icl_examples if is not None: list of lists of (optionally) lists with two elements: [[[src11, trg11], [src12, trg12]], [], [[src31, trg31]], ...]
        ## len(icl_examples) must be equal to len(src_sentences)

        assert isinstance(src_sentences, list), f"Expected src_sentences to be a list, got {type(src_sentences)}: {src_sentences}"
        assert len(src_sentences) > 0, "src_sentences must not be an empty list"
        assert isinstance(src_sentences[0], str), f"Expected src_sentences to be a list of strings, got {type(src_sentences[0])}: {src_sentences[0]}"

        url = self.translate_model_api if not only_representation else self.embedding_pooling_model_api
        url_idx = self.get_int_env_id() if api_idx is None else api_idx
        url_idx = 0 if url_idx is None else url_idx % len(url)
        url = url[url_idx]
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
                payload.append(('pooling', self.embedding_pooling_model_method_state))
                payload.append(('layer', self.embedding_pooling_model_layer))

            response = utils.requests_post(url, data=payload)

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

            if self.state_representation == "representation_per_token_with_features":
                assert len(translations.shape) == 3, f"Translations shape mismatch for representation_per_token_with_features: {translations.shape}"
            else:
                assert len(translations.shape) == 2, f"Translations shape mismatch for representation: {translations.shape}"

            assert translations.shape[0] == len(src_sentences), f"Translations shape mismatch for representation_per_token_with_features first dimension: {translations.shape} vs {len(src_sentences)}"
            assert translations.shape[-1] == self.state_dim_per_token, f"Translations shape mismatch for representation_per_token_with_features last dimension: {translations.shape} vs {self.state_dim_per_token}"

            if numpy:
                translations = translations.numpy()

            if self.apply_l2_normalization_state:
                translations = utils.l2_normalize(translations)

            if self.state_representation == "representation_per_token_with_features":
                translations = translations.reshape(len(src_sentences), -1) # flatten to (num_sentences, seq_len * model_hidden_size)
                new_translations = np.zeros((translations.shape[0], self.state_dim), dtype=translations.dtype)
                max_dim = min(translations.shape[-1], self.state_dim)
                new_translations[:,:max_dim] = translations[:, :max_dim] # truncate or keep the same if equal (tokens are obtained with padding at the right and prompt tokens are right-to-left, so the first tokens are the most relevant ones)
                translations = new_translations

            assert translations.shape[-1] == self.state_dim, f"Translations shape mismatch after flattening: {translations.shape} vs {(len(src_sentences), self.state_dim)}"

        assert len(translations) == len(src_sentences), f"Translations length mismatch: {len(translations)} vs {len(src_sentences)}"

        return translations

    def is_done(self):
        limit_examples = len(self.current_icl_examples) >= self.current_max_icl_examples
        early_stopping_terminated = self.early_stopping
        terminated = limit_examples or early_stopping_terminated
        truncated = False # we have not defined an artificial termination of the environment

        assert len(self.icl_example_representation) == self.embeddings_index.ntotal

        return terminated, truncated

    def get_closest_neighbors_urls(self, proto_actions, k=1, distance_expected_zero=False, get_representations_instead_of_embeddings=True, observations=None,
                                   _index=None, _urls_representation=None, remove_overlapping_actions=True, check_l2_norm=False, debug=False, return_all_neighbors=False,
                                   actions_without_replacement=False):
        """
            observations: states from which proto_actions were generated
        """
        #proto_actions = utils.l2_normalize(proto_actions) if self.apply_l2_normalization else proto_actions
        #proto_actions = utils.embeddings_index_sanity_check(proto_actions, last_dimmension_shape=self.action_dim)
        proto_actions = utils.embeddings_index_sanity_check(proto_actions, last_dimmension_shape=self.action_dim, check_l2_norm=check_l2_norm)
        assert isinstance(proto_actions, np.ndarray), f"Expected proto_actions to be a numpy array, got {type(proto_actions)}: {proto_actions}"
        assert len(proto_actions.shape) == 2, f"Expected proto_actions to be a 2D numpy array, got shape {proto_actions.shape}: {proto_actions}"
        assert proto_actions.shape[-1] == self.action_dim, f"Expected proto_actions last dimension to be {self.action_dim}, got {proto_actions.shape[-1]}"
        assert isinstance(k, int), k
        assert k > 0, "k must be greater than 0"

        if return_all_neighbors and not actions_without_replacement:
            results = [list(self.str2representation_valid_actions_k) for _ in range(proto_actions.shape[0])]

            if not get_representations_instead_of_embeddings:
                flatten = [q for w in results for q in w] # Flatten list of lists
                results = np.stack([self.str2representation[s] for s in flatten], axis=0)
                results = torch.from_numpy(results)

                assert len(results.shape) == 2, results.shape
                assert results.shape == (proto_actions.shape[0] * k, proto_actions.shape[1]), f"Results shape mismatch: {results.shape} vs {(proto_actions.shape[0] * k, proto_actions.shape[1])}"

                results = results.reshape((proto_actions.shape[0], k, proto_actions.shape[1]))
                results = results.to(self.device)

            return results, None, None

        translation_candidate = None

        if observations is not None:
            # get translation_candidate from observations

            if isinstance(observations, torch.Tensor):
                observations = observations.detach().cpu().numpy()

            assert isinstance(observations, np.ndarray), f"Expected observations to be a numpy array, got {type(observations)}: {observations}"
            assert len(observations.shape) == 2, f"Expected observations to be a 2D numpy array, got shape {observations.shape}: {observations}"
            assert observations.shape[0] == proto_actions.shape[0], f"Expected observations first dimension to be equal to proto_actions first dimension ({proto_actions.shape[0]}), got {observations.shape[0]}"
            assert observations.shape[1] == self.state_dim, f"Expected observations second dimension to be {self.state_dim}, got {observations.shape[1]}"

            translation_candidate = []

            for idx, observation in enumerate(observations):
                _observation2 = observation[:self.action_dim]

                assert not np.allclose(_observation2, np.zeros_like(_observation2)), f"Element was expected to be found in the first position: {idx}: {_observation2}"

                D_overlap, I_overlap = self.src_sentences_index.search(np.expand_dims(_observation2, axis=0), 1)

                assert len(D_overlap) == 1 and len(I_overlap) == 1, f"Expected single result from src_sentences_index search, got distances {D_overlap} and indices {I_overlap}"

                d_overlap = D_overlap[0]
                i_overlap = I_overlap[0]

                assert len(d_overlap) == 1 and len(i_overlap) == 1, f"Expected single result from src_sentences_index search, got distances {d_overlap} and indices {i_overlap}"
                assert i_overlap[0] >= 0, f"Expected to find overlapping source sentence in src_sentences_index, got distance {d_overlap[0]} and index {i_overlap[0]}: {_observation2}"

                src_sentence = self.data[i_overlap[0]][0]
                _observation3 = self.str2representation[src_sentence]
                atol = 1e-2
                rtol = 1e-2
                check1 = np.allclose(_observation2, _observation3, atol=atol, rtol=rtol) # np.allclose is not symmetric!
                check2 = np.allclose(_observation3, _observation2, atol=atol, rtol=rtol)

                if not check1 and not check2:
                    allclose_left_part = np.absolute(_observation2 - _observation3)
                    allclose_right_part1 = atol + rtol * np.absolute(_observation2)
                    allclose_right_part2 = atol + rtol * np.absolute(_observation3)
                    allclose_non_assert_part1 = allclose_left_part > allclose_right_part1
                    allclose_non_assert_part2 = allclose_left_part > allclose_right_part2

                    self.logger_wrapper(gym.logger.error, "Found source sentence with non-matching representation (index distance: %s): representation from index %s " \
                                                          "vs representation from str2representation %s for src_sentence %s", d_overlap[0], _observation2, _observation3, src_sentence)
                    self.logger_wrapper(gym.logger.error, "observation: %s: %s", observation.shape, observation)
                    self.logger_wrapper(gym.logger.error, "Non-matching values (1): %s and %s", _observation2[allclose_non_assert_part1], _observation3[allclose_non_assert_part1])
                    self.logger_wrapper(gym.logger.error, "Non-matching values (2): %s and %s", _observation2[allclose_non_assert_part2], _observation3[allclose_non_assert_part2])

                    assert check1 or check2

                translation_candidate.append(src_sentence)

                #self.logger_wrapper(gym.logger.debug, "Translation candidate kNN #%d: %s", idx, translation_candidate[-1])

        results = []
        index = self.embeddings_index if _index is None else _index
        urls_representation = self.icl_example_representation if _urls_representation is None else _urls_representation
        _add_eos_action = self.knn_always_add_eos_action and k > 1

        if index.ntotal == 0:
            # Faiss index is empty
            self.logger_wrapper(gym.logger.warn, "Faiss index seems to be empty: %d (sentences in pool: %d)", index.ntotal, len(urls_representation))

        #self.logger_wrapper(gym.logger.debug, "Default representation is %s", self.eos_token_str)

        D, I = index.search(proto_actions, k) # [D]istance, [I]ndex

        if translation_candidate is not None:
            assert len(translation_candidate) == proto_actions.shape[0] == D.shape[0], f"Expected translation_candidate length to be equal to proto_actions first dimension ({proto_actions.shape[0]} == {D.shape[0]}), got {len(translation_candidate)}"

        if self.enable_eos_action:
            eos_representation = self.str2representation[self.eos_token_str].copy()
            expected_eos_index = len(urls_representation) - 1

            assert len(eos_representation.shape) == 1 and eos_representation.shape[0] == self.action_dim, f"Expected EOS representation shape to be ({self.action_dim},), got {eos_representation.shape}"
            assert eos_representation.shape == proto_actions.shape[1:], f"Expected tiled EOS representation shape to be {proto_actions[1:].shape}, got {eos_representation.shape}"
            assert urls_representation[expected_eos_index] == self.eos_token_str, f"Expected last element in urls_representation to be EOS token string ({self.eos_token_str}), got {urls_representation[expected_eos_index]}"

        expected_shape = (proto_actions.shape[0], k)

        assert D.shape == expected_shape, f"Expected D.shape to be {expected_shape}, got {D.shape}"
        assert I.shape == expected_shape, f"Expected I.shape to be {expected_shape}, got {I.shape}"
        assert (D[I == -1] == np.finfo(np.float32).max * np.ones_like(D[I == -1])).all()

        d_modified_idxs = [(_a, _b) for _a, _b in zip(*np.where(I == -1))] if np.any(I == -1) else []

        # Obtain representations (str) from kNN idxs
        for idx1, (i, d) in enumerate(zip(I, D)):
            assert len(i.shape) == 1, i.shape
            assert len(d.shape) == 1, d.shape

            overlapping_hits = 0
            _translation_candidate = None if translation_candidate is None else translation_candidate[idx1]
            eos_found = False

            results.append([])

            for idx2, (value_idx, value_distance) in enumerate(zip(i, d)):
                if len(results[-1]) >= k:
                    break
                if value_idx < 0:
                    assert (i[idx2:] == -1 * np.ones_like(i[idx2:])).all(), f"Expected all remaining indices to be -1, got {i[idx2:]}"

                    self.logger_wrapper(gym.logger.warn, "Not entries close (idx: %d): %s: %s ... %s", idx2, proto_actions.shape, proto_actions[0], proto_actions[-1])

                    break # No more valid indices as -1 values are at the end of the list

                url = urls_representation[value_idx]

                assert isinstance(url, str), f"Expected url to be a string, got {type(url)}: {url}"

                if url != self.eos_token_str:
                    # Check if we need to remove this hit
                    src_icl_example, trg_icl_example = url.split('\t')

                    if remove_overlapping_actions and _translation_candidate == src_icl_example:
                        self.logger_wrapper(gym.logger.debug, "Removing overlapping action: %s", src_icl_example)

                        overlapping_hits += 1
                        d[idx2] = np.finfo(np.float32).max
                        i[idx2] = -2

                        d_modified_idxs.append((idx1, idx2))

                        #continue # do not add this entry
                        # add the entry since it is going to be modified
                else:
                    eos_found = True

                results[-1].append(url)

            assert len(results[-1]) <= k, f"Expected results[-1] to have at most {k} elements, got {len(results[-1])}"
            assert overlapping_hits <= 1, f"Expected at most one overlapping hit, got {overlapping_hits}: this might happen if same source is repeated in the ICL examples"

            if actions_without_replacement:
                assert k == 1, k
                assert len(results[-1]) == 1, len(results[-1])
                assert len(D[idx1]) == 1, len(D[idx1])
                assert len(I[idx1]) == 1, len(I[idx1])
                assert 2 + len(self.current_icl_examples) <= index.ntotal, f"Not enough entries in the index to perform search without replacement: {index.ntotal} entries vs {2 + len(self.current_icl_examples)} required (1 for the hit and 1 for the replacement if needed, plus the number of ICL examples to remove if needed)"

                dist_idxs = np.flip(d.argsort())
                longest_dist_idx = dist_idxs[0]

                assert longest_dist_idx == 0, f"Expected longest_dist_idx to be 0 since k=1, got {longest_dist_idx}"

                if overlapping_hits == 1:
                    assert len(dist_idxs) == k, f"{len(dist_idxs)} vs {k}"
                    assert I[idx1][longest_dist_idx] == -2, I[idx1][longest_dist_idx]
                    assert D[idx1][longest_dist_idx] == np.finfo(np.float32).max, D[idx1][longest_dist_idx]

                new_search = False
                current_result = results[-1][0].split('\t')[0]

                for src_icl_example, trg_icl_example in self.current_icl_examples:
                    if src_icl_example == current_result: # source part of the hit
                        new_search = True

                        break

                if new_search or overlapping_hits == 1:
                    self.logger_wrapper(gym.logger.info, "Removing actions that were previously selected (%s, %s)", new_search, overlapping_hits)

                    D_aux, I_aux = index.search(np.expand_dims(proto_actions[idx1], axis=0), 2 + len(self.current_icl_examples))

                    assert len(I_aux) == 1, len(I_aux)
                    assert len(D_aux) == 1, len(D_aux)

                    for i_aux in I_aux[0]:
                        assert i_aux >= 0, i_aux

                    src_icl_example_aux = [urls_representation[I_aux[0][_idx]].split('\t')[0] for _idx in range(len(I_aux[0]))]
                    D_aux_min = np.finfo(np.float32).max
                    D_aux_min_idx = None

                    for _idx, src_icl_example in enumerate(src_icl_example_aux):
                        if remove_overlapping_actions and _translation_candidate == src_icl_example:
                            D_aux[0][_idx] = np.finfo(np.float32).max
                        elif src_icl_example in [src for src, trg in self.current_icl_examples]: # source part of the ICL examples to remove
                            D_aux[0][_idx] = np.finfo(np.float32).max
                        elif D_aux[0][_idx] < D_aux_min:
                            D_aux_min = D_aux[0][_idx]
                            D_aux_min_idx = _idx

                    assert D_aux_min_idx is not None, "Expected to find a valid replacement entry in the index for search without replacement"

                    I_aux = I_aux[0][D_aux_min_idx] # take closest
                    results_aux = urls_representation[I_aux]
                    D_aux_repr = self.str2representation[results_aux].copy()
                    dist = np.linalg.norm(proto_actions[idx1] - D_aux_repr, axis=0) ** 2 # we assume euclidean distance

                    assert np.isclose(dist, D_aux[0][D_aux_min_idx]), f"{dist} vs {D_aux[0][D_aux_min_idx]}" # should work as long as euclidean distance is used
                    assert I_aux >= 0, I_aux
                    assert isinstance(results_aux, str), f"Expected results_aux to be a string, got {type(results_aux)}: {results_aux}"
                    assert len(D_aux_repr.shape) == 1 and D_aux_repr.shape[0] == self.action_dim, f"Expected D_aux_repr shape to be ({self.action_dim},), got {D_aux_repr.shape}"

                    D[idx1][longest_dist_idx] = np.linalg.norm(proto_actions[idx1] - D_aux_repr, axis=0) ** 2 # we assume euclidean distance
                    I[idx1][longest_dist_idx] = I_aux
                    results[-1][longest_dist_idx] = results_aux
                    overlapping_hits = 0

            if (_add_eos_action and not eos_found) or overlapping_hits == 1:
                # Replace the value with longest distance with EoS (or duplicate if EoS is disabled)
                self.logger_wrapper(gym.logger.info, "Adding EoS action (%s)/Removing overlapping action (%s): %s", _add_eos_action and not eos_found, overlapping_hits, src_icl_example)

                dist_idxs = np.flip(d.argsort())
                longest_dist_idx = dist_idxs[0]

                assert len(dist_idxs) == k, f"{len(dist_idxs)} vs {k}"

                if overlapping_hits == 1:
                    assert I[idx1][longest_dist_idx] == -2, I[idx1][longest_dist_idx]
                    assert D[idx1][longest_dist_idx] == np.finfo(np.float32).max, D[idx1][longest_dist_idx]

                if self.enable_eos_action:
                    D_aux_repr = eos_representation
                    I_aux = expected_eos_index
                    results_aux = self.eos_token_str
                elif len(dist_idxs) > 1:
                    I_aux = i[dist_idxs[1]] # duplicate item (next idx with longest distance)
                    results_aux = urls_representation[I_aux]
                    D_aux_repr = self.str2representation[results_aux].copy()
                else:
                    # k = 1, so we need to find any other representation
                    assert k == 1
                    assert index.ntotal > 1, "index is too small to find another entry"

                    D_aux, I_aux = index.search(np.expand_dims(proto_actions[idx1], axis=0), 2)
                    src_icl_example_aux = urls_representation[I_aux[0][0]].split('\t')[0] # closest example source part

                    assert translation_candidate[idx1] == src_icl_example_aux, f"{translation_candidate[idx1]} vs {src_icl_example_aux}" # i[0] is -2, but this is equivalent to I_aux[0][0] == i[0]
                    assert I_aux[0][1] >= 0, f"{I_aux[0][1]}"
                    assert I_aux[0][1] != I_aux[0][0], f"{I_aux[0][1]} vs {I_aux[0][0]}"

                    I_aux = I_aux[0][1] # take second closest
                    results_aux = urls_representation[I_aux]
                    D_aux_repr = self.str2representation[results_aux].copy()
                    dist = np.linalg.norm(proto_actions[idx1] - D_aux_repr, axis=0) ** 2 # we assume euclidean distance

                    assert np.isclose(dist, D_aux[0][1]), f"{dist} vs {D_aux[0][1]}" # should work as long as euclidean distance is used

                assert I_aux >= 0, I_aux
                assert isinstance(results_aux, str), f"Expected results_aux to be a string, got {type(results_aux)}: {results_aux}"
                assert len(D_aux_repr.shape) == 1 and D_aux_repr.shape[0] == self.action_dim, f"Expected D_aux_repr shape to be ({self.action_dim},), got {D_aux_repr.shape}"

                D[idx1][longest_dist_idx] = np.linalg.norm(proto_actions[idx1] - D_aux_repr, axis=0) ** 2 # we assume euclidean distance
                I[idx1][longest_dist_idx] = I_aux
                results[-1][longest_dist_idx] = results_aux

            if len(results[-1]) < k:
                raise Exception(f"This should never happen! {index.ntotal} {k} {len(results[-1])}")

                # Add items to avoid tensor errors because dimensions don't match (default representation is EoS)
                assert index.ntotal < k, f"{index.ntotal} >= {k}"

                self.logger_wrapper(gym.logger.debug, "Not enough entries close for entry %d/%d (found: %d): returning %d default representation(s) (%s)", idx1 + 1, len(I), len(results[-1]), k - len(results[-1]), self.eos_token_str)

                while len(results[-1]) < k:
                    results[-1].append(self.eos_token_str) # Default representation if no hits are found

            assert len(results[-1]) == k

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

            flatten = [q for w in results for q in w] # Flatten list of lists

            if len(flatten) > 0:
                assert isinstance(flatten[0], str), f"Expected flatten to be a list of strings, got {type(flatten[0])}: {flatten[0]}"

            if debug:
                self.logger_wrapper(gym.logger.error, "faiss.I (first and last 5): %s: %s ... %s", I.shape, I[:,:5], I[:,-5:])
                self.logger_wrapper(gym.logger.error, "faiss.D (first and last 5): %s: %s ... %s", D.shape, D[:,:5], D[:,-5:])

            _all_urls_subset = [url for url in flatten if url not in self.str2representation]

            assert len(_all_urls_subset) == 0, f"This should not happen in this environment: {_all_urls_subset}"

            results = np.stack([self.str2representation[s] for s in flatten], axis=0)
            results = torch.from_numpy(results)

            assert len(results.shape) == 2, results.shape
            assert results.shape == (proto_actions.shape[0] * k, proto_actions.shape[1]), f"Results shape mismatch: {results.shape} vs {(proto_actions.shape[0] * k, proto_actions.shape[1])}"

            results = results.reshape((proto_actions.shape[0], k, proto_actions.shape[1]))

            if debug:
                results2 = results[0,:5]
                results3 = results[0,-5:]

                self.logger_wrapper(gym.logger.error, "faiss.closest_embedding (first and last 5): %s: %s ... %s", results.shape, results2, results3)

            results = results.to(self.device)

        return results, D, I

    def get_translation_candidate(self):
        n = len(self.data)
        repeat = False

        if self.translation_candidate < 0 or self.episode <= 1:
            assert self.translation_candidate == -1, self.translation_candidate
            assert self.episode == 1, self.episode

        # Repeat current translation candidate?
        if self.repeat_translation_candidates and self.translation_candidate >= 0:
            idx = self.translation_candidate
            if_stm = False

            if self.repeat_translation_candidates_times > 0:
                assert self.repeat_translation_candidates_times_counter >= 0, self.repeat_translation_candidates_times_counter

                if_stm = self.repeat_translation_candidates_times_counter > 0
                self.repeat_translation_candidates_times_counter -= 1

            if if_stm:
                src_translation_candidate = self.data[self.translation_candidate][0]

                self.logger_wrapper(gym.logger.info, "Translation candidate (repeating) #%d: %s", idx, src_translation_candidate)

                self.translation_candidates_selected_consecutive_episode[idx] += 1
                repeat = True
            else:
                self.translation_candidates_selected_consecutive_episode[idx] = 0
                self.repeat_translation_candidates_times_counter = self.repeat_translation_candidates_times

        translation_candidate = self.translation_candidate if repeat else None

        # Select translation candidate
        if translation_candidate is None:
            if self.translation_candidate_strategy in ("sequential", "sequential_shuffle_per_epoch"):
                translation_candidate = (self.episode - 1) % n # sequential sweep

                if self.translation_candidate_strategy == "sequential_shuffle_per_epoch" and translation_candidate == 0:
                    # shuffle once per epoch
                    random.shuffle(self.translation_candidates_idx)

                    self.logger_wrapper(gym.logger.debug, "Shuffle idxs for translation candidate selection (first and last 5 idxs): %s ... %s", self.translation_candidates_idx[:5], self.translation_candidates_idx[-5:])

                translation_candidate = self.translation_candidates_idx[translation_candidate]
                src_translation_candidate = self.data[translation_candidate][0]

                self.logger_wrapper(gym.logger.info, "Translation candidate #%d: %s", translation_candidate, src_translation_candidate)
            elif self.translation_candidate_strategy == "choice_with_replacement":
                translation_candidate = random.choice(range(n))
                src_translation_candidate = self.data[translation_candidate][0]

                self.logger_wrapper(gym.logger.info, "Translation candidate #%d: %s", translation_candidate, src_translation_candidate)
            else:
                raise Exception(f"Unknown translation candidate selection strategy: {self.translation_candidate_strategy}")

        assert 0 <= translation_candidate < n, f"Translation candidate index must be in [0, {n}), got {translation_candidate}"

        self.translation_candidates_selected_episode[translation_candidate] += 1

        return translation_candidate

    def apply_step(self, current_action):
        assert isinstance(current_action, str), f"Expected current_action to be a string, got {type(current_action)}: {current_action}"
        assert len(self.current_icl_examples) < self.current_max_icl_examples, f"Current length of ICL examples ({self.current_icl_examples}) must be less than max ICL examples ({self.current_max_icl_examples})"
        assert self.time_step > 0, self.time_step

        if self.enable_eos_action and current_action == self.eos_token_str:
            # Early stopping action
            self.logger_wrapper(gym.logger.info, "Early stopping action (%s) received in time step #%d", current_action, self.time_step)

            assert self.early_stopping is False, "Early stopping action already received in this episode"

            self.early_stopping = True
        else:
            self.current_icl_examples.insert(0, current_action.split('\t')) # the new example is added at the beginning, so the prompt left-to-right is new-to-old

            assert len(self.current_icl_examples[-1]) == 2, f"Expected current ICL example to have two elements (source and target), got {len(self.current_icl_examples[-1])}: {self.current_icl_examples[-1]}"

        terminated, truncated = self.is_done()
        reward = 0.0

        if self.early_stopping:
            assert terminated or truncated, f"Early stopping action received but not terminated or truncated: {terminated}, {truncated}"
            assert self.enable_eos_action

            observation = self.str2representation[self.eos_token_str]

            assert isinstance(observation, np.ndarray), type(observation)

            observation = observation.copy() # This is relevant to avoid changing the initial representations in case the observations are modified

            if self.state_representation == "model_single_representation":
                assert self.state_window_length == 1, self.state_window_length
                assert len(self.current_state_window) == 1, len(self.current_state_window)
                assert self.current_state_window[0].shape == (self.action_dim,), self.current_state_window[0].shape
                assert observation.shape == self.current_state_window[0].shape, observation.shape

                observation += self.current_state_window[0]

                if self.apply_l2_normalization_state:
                    observation = utils.l2_normalize(observation)
        else:
            # Update state

            if self.state_representation in ("model_single_representation", "representation_per_token_with_features"):
                src_sentence = self.data[self.translation_candidate][0]
                observation = self.get_state_representation([src_sentence], icl_examples=[self.current_icl_examples])[0]

                assert observation.shape[0] % self.state_dim_per_token == 0, f"Observation shape mismatch: {observation.shape[0]} vs {self.state_dim_per_token}"
            elif self.state_representation in ("sentence_and_actions", "model_single_representation+sentence_and_actions"):
                icl_example = '\t'.join(self.current_icl_examples[-1])

                assert icl_example in self.str2representation, icl_example

                observation = self.str2representation[icl_example]

                assert observation.shape[0] == self.action_dim, f"Observation shape mismatch: {observation.shape[0]} vs {self.action_dim}"
            else:
                raise Exception(f"Unknown state representation: {self.state_representation}")

        assert isinstance(observation, np.ndarray), type(observation)
        assert len(observation.shape) == 1, f"Expected observation to be a 1D numpy array, got shape {observation.shape}: {observation}"

        observation = observation.copy()

        if self.state_representation == "representation_per_token_with_features":
            observation = observation.reshape(-1, self.state_dim_per_token) # (seq_len, model_hidden_size)
        elif self.time_step > 1:
            offset = 0 if self.state_representation != "model_single_representation+sentence_and_actions" or self.time_step < 2 else 1

            assert not np.allclose(self.current_state_window[self.time_step - 1], np.zeros_like(self.current_state_window[self.time_step])), self.time_step

            for idx in range(self.time_step + offset, self.current_state_window.maxlen):
                assert np.allclose(self.current_state_window[idx], np.zeros_like(self.current_state_window[idx])), f"{self.time_step} + {offset} ...  {self.current_state_window.maxlen}: {idx}"

        if self.apply_l2_normalization_state:
            assert utils.check_l2_normalized(observation), "Observation must be l2 normalized"

        if self.state_representation == "model_single_representation":
            self.current_state_window[0] = observation
        elif self.state_representation == "model_single_representation+sentence_and_actions":
            # We need to do it after append() above in order to avoid overwriting the position 1
            src_sentence = self.data[self.translation_candidate][0]
            model_single_representation = self.get_state_representation([src_sentence], icl_examples=[self.current_icl_examples])[0]

            # Set state
            self.current_state_window[self.time_step + 1] = observation
            self.current_state_window[1] = model_single_representation.copy() # the first position is for the src sentence representation
        elif self.state_representation == "representation_per_token_with_features":
            is_zero_vector = (observation == 0).all(axis=1)
            sum_zero = is_zero_vector.sum(axis=0).item()
            used_tokens = observation.shape[0] - sum_zero
            num_tokens = min(used_tokens, self.state_window_length - 1 - 1 - 1)

            assert used_tokens > 0

            if sum_zero > 0:
                # LLMs use left padding, but we adapt the embeddings so padding is located at right position
                assert (observation[-1] == 0).all(axis=0)
                assert (observation[used_tokens:] == 0).all(axis=1).all(axis=0)
                assert (observation[0] != 0).any(axis=0)
                assert (observation[:used_tokens] != 0).any(axis=1).all(axis=0)

            self.current_state_window[2] = np.ones(self.state_dim_per_token) * (self.time_step + 1) / 10 # state representing the current step. 10 as constant so the value is less than 1 (as long as the current step is less than 10)

            # Clean (do not remove the first position, which is reserved for the source sentence representation; do not remove the second position, which is reserved for the information regardin the max. number of ICL examples)
            for idx in range(3, self.state_window_length):
                self.current_state_window[idx] = np.zeros(self.state_dim_per_token)

            # Update
            initial_idx = used_tokens - num_tokens # use the last embeddings instead of the first ones (they are more relevant, as last embeddings in the LLM have information of all the previous ones)

            for idx in range(num_tokens):
                self.current_state_window[idx + 1 + 1 + 1] = observation[idx + initial_idx]

            assert (observation[idx + initial_idx] != 0).any(axis=0)

            if sum_zero > 0:
                assert (observation[idx + initial_idx + 1] == 0).all(axis=0)

            self.logger_wrapper(gym.logger.debug, "representation_per_token_with_features: %s tokens (used: min(%d, %d))", observation.shape, used_tokens, self.state_window_length - 1 - 1 - 1)
            self.logger_wrapper(gym.logger.debug, "First ... last tokens: %s %s %s %s ... %s %s %s %s",
                                self.current_state_window[1], self.current_state_window[2], self.current_state_window[3], self.current_state_window[4],
                                self.current_state_window[-4], self.current_state_window[-3], self.current_state_window[-2], self.current_state_window[-1])
        else:
            self.current_state_window[self.time_step] = observation

        translation = None
        src_sentence, reference = self.data[self.translation_candidate]
        eval_strategy = self.eval_strategy_eval if self.is_eval_env else self.eval_strategy_training

        if terminated or truncated:
            # Generate translation
            if eval_strategy == "actions-bm25":
                translation = ["none"]

                self.logger_wrapper(gym.logger.debug, "Translation not generated: eval_strategy == 'actions-bm25'")
            else:
                translation = self.get_translations([src_sentence], icl_examples=[self.current_icl_examples])[0]

        # Compute reward
        reward = self.get_reward(src_sentence, reference, translation=translation, icl_examples=list(self.current_icl_examples))

        if terminated or truncated:
            if self.translation_candidate not in self.best_reward_seen:
                self.best_reward_seen[self.translation_candidate] = reward
            else:
                self.best_reward_seen[self.translation_candidate] = max(self.best_reward_seen[self.translation_candidate], reward)

            self.translation_candidates_reward_mean_episode.append(reward)

            return terminated, truncated, reward, translation

        # Return
        terminated, truncated = self.is_done()

        assert not terminated and not truncated, "Step should not terminate or truncate immediately after applying an action"

        return terminated, truncated, reward, translation

    def get_score_from_icl_example_bm25(self, remove_overlapping_actions=True):
        assert 0 < self.time_step <= self.current_max_icl_examples, self.time_step
        assert isinstance(self.translation_candidate, int), type(self.translation_candidate)
        assert isinstance(self.current_icl_examples, list), type(self.current_icl_examples)
        assert isinstance(self.data[self.translation_candidate][0], str), type(self.data[self.translation_candidate][0])
        assert isinstance(self.str2representation_valid_actions_k, list), type(self.str2representation_valid_actions_k)
        assert len(self.current_icl_examples) == self.time_step
        assert len(self.current_icl_examples[-1]) == 2, len(self.current_icl_examples[-1])
        assert isinstance(self.current_icl_examples[-1][0], str), type(self.current_icl_examples[-1][0])

        src_translation_candidate = str(self.data[self.translation_candidate][0])
        focus_icl_example = str(self.current_icl_examples[-1][0])

        if remove_overlapping_actions:
            assert focus_icl_example != src_translation_candidate

        # BM25
        #bm25 = BM25Okapi(self.data_icl_examples_bm25_corpus_tokenized)

        assert len(self.data_icl_examples_bm25_corpus) >= self.current_max_icl_examples
        assert focus_icl_example in self.data_icl_examples_bm25_corpus

        tokenized_query = src_translation_candidate.split()
        scores = self.data_icl_examples_bm25.get_scores(tokenized_query).tolist()
        #top_n = self.data_icl_examples_bm25.get_top_n(tokenized_query, corpus, n=len(scores)) # debug

        assert len(scores) == len(self.data_icl_examples_bm25_corpus) == len(self.data_icl_examples_bm25_corpus_tokenized), f"BM25 scores length mismatch: {len(scores)} vs {len(self.data_icl_examples_bm25_corpus)} vs {len(self.data_icl_examples_bm25_corpus_tokenized)}"

        focus_icl_example_idx = self.data_icl_examples_bm25_corpus.index(focus_icl_example)
        focus_icl_example_score = scores[focus_icl_example_idx]

        return focus_icl_example_score

if __name__ == "__main__":
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    file_data = sys.argv[3]
    file_data_icl_examples = sys.argv[4]
    parsed_kwargs = utils.parse_args(sys.argv[5:])

    # Initialize and check environment
    env = MTICLEnv(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)

    check_env(env)
