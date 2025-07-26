
import sys
import time
import gzip
import math
import json
import base64
import pickle
import random
import datetime
import collections

import utils

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import transformers
import torch
import torch.nn as nn
import numpy as np
import requests
import faiss

# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
# https://gymnasium.farama.org/api/env/

# Gym register:
#gym.envs.registration.register(
#     id="focused_crawling_env/FocusedCrawlingLangIdentificationWithTransformersRepresentationActionsAndStatesEnv-v0",
#     entry_point="focused_crawling_env.envs:FocusedCrawlingLangIdentificationWithTransformersRepresentationActionsAndStatesEnv",
#     max_episode_steps=None,
#     nondeterministic=True,
#     order_enforce=True,
#     autoreset=False,
#)

class FocusedCrawlingLangIdentificationWithTransformersRepresentationActionsAndStatesNewMDPEnv(gym.Env):
    """Custom Environment for crawling pages in given languages that follows gym interface."""

    metadata = {"render_modes": ["human"],}

    def logger_wrapper(self, callback, _str, *args, **kwargs):
        d = str(datetime.datetime.now())

        if len(_str) > 0:
            if hasattr(self, "reset_times") and self.reset_times > 0:
                _str = f"[{d}] [{self.episode}:{self.time_step} -> {self.time_step_global}] {_str}"
            else:
                _str = f"[{d}] {_str}"

        callback(_str, *args, **kwargs)

    def __init__(self, file_data, file_data_icl_examples, **kwargs):
        super().__init__()

        gym.logger.set_level(utils.dict_or_default(kwargs, "gym_logger_level", gym.logger.INFO))

        self.file_data = file_data # format: source<tab>reference
        self.file_data_icl_examples = file_data_icl_examples # format: source<tab>reference

        assert utils.file_exists(self.file_data), self.file_data
        assert utils.file_exists(self.file_data_icl_examples), self.file_data_icl_examples

        self.logger_wrapper(gym.logger.info, "Provided arguments: %s", kwargs)

        self.reset_times = 0
        self.episode = 0
        self.gamma = utils.dict_or_default(kwargs, "gamma", 0.99)
        self.state_window_length = utils.dict_or_default(kwargs, "state_window_length", 4)
        self.state_window_type = utils.dict_or_default(kwargs, "state_window_type", "concatenate")
        self.max_icl_examples = utils.dict_or_default(kwargs, "max_icl_examples", 4)
        self.max_data_entries = utils.dict_or_default(kwargs, "max_data_entries", -1)
        self.max_data_icl_examples_entries = utils.dict_or_default(kwargs, "max_data_icl_examples_entries", -1)

        if self.state_window_length < self.max_icl_examples:
            self.logger_wrapper(gym.logger.warn, "self.state_window_length = %d < self.max_icl_examples = %d", self.state_window_length, self.max_icl_examples)

        # Model conf
        self.batch_size = utils.dict_or_default(kwargs, "batch_size", 16)
        self.device = torch.device(utils.dict_or_default(kwargs, "device", "cuda"))
        self.translate_model_api = utils.dict_or_default(kwargs, "translate_model_api", None)
        self.embedding_single_token_model_api = utils.dict_or_default(kwargs, "embedding_single_token_model_api", None)
        self.embedding_translation_model_api = utils.dict_or_default(kwargs, "embedding_translation_model_api", None)
        self.eval_model_api = utils.dict_or_default(kwargs, "eval_model_api", None)

        assert self.translate_model_api is not None
        assert self.embedding_single_token_model_api is not None
        assert self.embedding_translation_model_api is not None
        assert self.eval_model_api is not None

        self.model_hidden_size = utils.dict_or_default(kwargs, "model_hidden_size", 4096) # (former self.max_transformer_output_length) https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json#L9
        self.state_dim = self.model_hidden_size
        self.action_dim = self.model_hidden_size

        if self.state_window_type == "concatenate":
            self.state_window_type_callback = lambda l: np.concatenate(l, axis=0, dtype=np.float32)
            self.state_dim *= self.state_window_length
        elif self.state_window_type == "average":
            self.state_window_type_callback = lambda l: np.mean(l, axis=0, dtype=np.float32)
        elif self.state_window_type == "maxpooling":
            self.state_window_type_callback = lambda l: np.float32(np.max(l, axis=0))
        else:
            raise Exception(f"Given window type is not valid: {self.state_window_type} (valid: {self.valid_state_window_type})")

        self.logger_wrapper(gym.logger.debug, "State and action embeddnig size: %d %d", self.state_dim, self.action_dim)

        # Define action and observation space (embeddings)
        #self.action_space = gym.spaces.Box(-sys.float_info.max, sys.float_info.max, shape=(self.model_hidden_size,))
        #self.observation_space = gym.spaces.Box(-sys.float_info.max, sys.float_info.max, shape=(self.model_hidden_size,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(self.action_dim,)) # input/output model is expected to have tanh in order to be in [-1, 1]
        self.observation_space = gym.spaces.Box(-1., 1., shape=(self.state_dim,)) # input/output model is expected to have tanh in order to be in [-1, 1]

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
            observation = self.apply_extra_dimension_to_observation(observation)
            reward = 0.0

            return observation, reward, terminated, truncated, info

        self.time_step += 1
        self.time_step_global += 1

    def _hard_reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.logger_wrapper(gym.logger.debug, "Env reset (hard): episode %d", self.episode)

        self.time_step_global = 0
        self.data = []
        self.data_icl_examples = []
        #self.observation_hash_dict = {} # We do not remove this data in _soft_reset because the replay buffer is not reseted after an episode ends
        #self.sentences2representation = {} # former self.url2representation

        observation, info = self._soft_reset(seed, {**(options if isinstance(options, dict) else {}), **{"reset_from_hard_reset": True}})

        return observation, info

    def _soft_reset(self, seed=None, options=None):
        # Difference with self._hard_reset: we keep all the results from the models files to avoid computing them again
        is_not_hard_reset = not isinstance(options, dict) or "reset_from_hard_reset" not in options or not options["reset_from_hard_reset"]
        is_soft_reset = is_not_hard_reset
        is_hard_reset = not is_not_hard_reset

        if is_soft_reset:
            super().reset(seed=seed)

            self.logger_wrapper(gym.logger.debug, "Env reset (soft): episode %d", self.episode)

        info = {}
        self.time_step = 0
        self.current_translations = 0 # former self.current_downloaded_urls TODO
        self.current_state_window = collections.deque(maxlen=self.state_window_length)
        self.translation_embeddings_representation = {} # formet self.active_urls_representation
        self.translation_embeddings_representation_url2idx = {} # formet self.active_urls_representation_url2idx
        self.embeddings_index = faiss.IndexFlatL2(self.action_dim)
        self.rewards = []
        self.current_datetime = datetime.datetime.now()

        for _ in range(self.state_window_length):
            self.current_state_window.append(np.zeros(self.model_hidden_size))

        self.insert_embeddings([self.url_seeds[0]], representation) # TODO add all icl examples and EoS token

        observation = self.state_window_type_callback(self.current_state_window)

        if is_hard_reset:
            self.load_data()

        # Shuffle data in order to avoid model memorization
        random.shuffle(self.data)
        random.shuffle(self.data_icl_examples)

        # TODO get initial observation from self.data using self.time_step

        return observation, info

    def reset(self, seed=None, options=None):
        self.episode += 1

        if seed is not None:
            utils.set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)

        if self.reset_times == 0 or (isinstance(options, dict) and "always_hard_reset" in options and options["always_hard_reset"]):
            rtn = self._hard_reset(seed=seed, options=options)
        else:
            # After first reset, _soft_reset is the default option if "always_hard_reset" is not defined in options
            rtn = self._soft_reset(seed=seed, options=options)

        observation, info = rtn

        assert observation.shape == (self.state_dim,), observation.shape

        self.reset_times += 1

        return observation, info

    def render(self):
        return self.current_translations

    def close(self):
        pass

    def load_data(self):
        assert self.file_data is not None
        assert self.file_data_icl_examples is not None

        self.logger_wrapper(gym.logger.info, "Loading data")

        with open(self.file_data, "rt") as fd:
            for idx, url_entry in enumerate(fd, 1):
                # Format: source<tab>reference

                try:
                    entry_data = url_entry.rstrip("\r\n").split('\t')

                    assert len(entry_data) == 2

                    #src_sentence, trg_sentence = entry_data

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

                    self.data_icl_examples.append(entry_data)
                except Exception as e:
                    self.logger_wrapper(gym.logger.error, "Loading data: error in line #%d", idx)

                    raise e

                if idx % 10000 == 0:
                    self.logger_wrapper(gym.logger.info, "Loading data: %d entries read (%d URLs loaded)", idx, len(self.data_icl_examples))

                if self.max_data_icl_examples_entries > 0 and idx >= self.max_data_icl_examples_entries:
                    break

        self.logger_wrapper(gym.logger.info, "Loading data (ICL examples): finished! %d entries read (%d URLs loaded)", idx, len(self.data_icl_examples))

    def remove_embedding(self, url, I):
        # TODO url is src_sentence<tab>trg_sentence

        assert I.shape == (1, 1), I.shape

        I = I[0][0]

        n_remove = self.embeddings_index.remove_ids(np.array([I]))

        if n_remove != 1:
            self.logger_wrapper(gym.logger.warn, "Could not remove embedding: %d. Why?", n_remove)
        else:
            # Adjust self.translation_embeddings_representation to fit the index situation after removing the embedding
            ids = sorted(self.translation_embeddings_representation.keys()) # We assume that ids[idx] + 1 == ids[idx + 1]

            for i in range(len(ids) - 1):
                # Sanity check
                assert ids[i] + 1 == ids[i + 1], ids

            current_id_idx = I

            assert self.translation_embeddings_representation[current_id_idx] == url, f"{self.translation_embeddings_representation[current_id_idx]} vs {url}"
            assert self.translation_embeddings_representation_url2idx[url] == current_id_idx, f"{self.translation_embeddings_representation_url2idx[url]} vs {current_id_idx}"

            while current_id_idx < ids[-1]:
                # Shift all URLs from [I + 1, len(ids)] one position to the left, then the last element can safely been removed
                next_url = self.translation_embeddings_representation[current_id_idx + 1]

                assert self.translation_embeddings_representation_url2idx[next_url] == current_id_idx + 1, f"{I} <= {current_id_idx + 1}"

                self.translation_embeddings_representation[current_id_idx] = self.translation_embeddings_representation[current_id_idx + 1]
                self.translation_embeddings_representation_url2idx[next_url] -= 1

                current_id_idx += 1

            if len(ids) > 1 and I < ids[-1]:
                assert self.translation_embeddings_representation[ids[-1]] == self.translation_embeddings_representation[ids[-2]], f"{ids[-2:]} - {self.translation_embeddings_representation[ids[-1]]} != {self.translation_embeddings_representation[ids[-2]]}"

            del self.translation_embeddings_representation[ids[-1]] # Remove safely last element
            del self.translation_embeddings_representation_url2idx[url]

    def insert_embeddings(self, urls, embeddings, _index=None, _urls_representation=None, _urls_representation_url2idx=None, update_representation=True):
        # TODO urls is a list of elements with format src_sentence<tab>trg_sentence

        #embeddings = utils.embeddings_index_sanity_check(embeddings, last_dimmension_shape=self.action_dim)
        index = self.embeddings_index if _index is None else _index
        urls_representation = self.translation_embeddings_representation if _urls_representation is None else _urls_representation
        urls_representation_url2idx = translation_embeddings_representation_url2idx if _urls_representation_url2idx is None else _urls_representation_url2idx

        utils.insert_embeddings(urls, embeddings, index, urls_representation, urls_representation_url2idx, self.action_dim, update_representation=update_representation)

    def get_reward(self, src_sentence, reference, translation=None):
        if translation is None:
            reward = 0.0
        else:
            eval_value = 0.0 # TODO obtain value using comet
            reward = eval_value

        return reward

if __name__ == "__main__":
    url_seed = sys.argv[1]
    target_langs = sys.argv[2:]
    env = FocusedCrawlingLangIdentificationWithTransformersRepresentationActionsAndStatesNewMDPEnv(
        url_seed, target_langs, gym_logger_level=gym.logger.DEBUG)

    check_env(env)
