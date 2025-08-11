
import sys
import json
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
            if hasattr(self, "reset_times") and self.reset_times > 0:
                _str = f"[{d}] [{self.episode}:{self.time_step} -> {self.time_step_global}] {_str}"
            else:
                _str = f"[{d}] {_str}"

        callback(_str, *args, **kwargs)

    def __init__(self, src_lang, trg_lang, file_data, file_data_icl_examples, **kwargs):
        super().__init__()

        gym.logger.set_level(utils.dict_or_default(kwargs, "gym_logger_level", gym.logger.INFO))

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.file_data = file_data # format: source<tab>reference
        self.file_data_icl_examples = file_data_icl_examples # format: source<tab>reference

        assert utils.file_exists(self.file_data), self.file_data
        assert utils.file_exists(self.file_data_icl_examples), self.file_data_icl_examples

        self.logger_wrapper(gym.logger.info, "Provided arguments: %s", kwargs)

        self.reset_times = 0
        self.episode = 0
        self.state_window_length = utils.dict_or_default(kwargs, "state_window_length", 4)
        self.state_window_type = utils.dict_or_default(kwargs, "state_window_type", "concatenate")
        self.max_icl_examples = utils.dict_or_default(kwargs, "max_icl_examples", 4)
        self.max_data_entries = utils.dict_or_default(kwargs, "max_data_entries", -1)
        self.max_data_icl_examples_entries = utils.dict_or_default(kwargs, "max_data_icl_examples_entries", -1)
        self.state_representation = utils.dict_or_default(kwargs, "state_representation", "translation_and_icl_examples")

        assert self.state_representation in ("translation_and_icl_examples", "actions"), f"Unexpected state representation: {self.state_representation}"

        if self.state_representation == "translation_and_icl_examples" and self.state_window_type == "concatenate" and self.state_window_length > 1:
            self.logger_wrapper(gym.logger.warn, "State window type is 'concatenate' and state window length is greater than 1: %d > 1. Modifying value to 1", self.state_window_length)

            self.state_window_length = 1

        if self.state_window_type == "concatenate" and self.state_window_length < self.max_icl_examples:
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

        self.logger_wrapper(gym.logger.info, "translate_model_api: %s", self.translate_model_api)
        self.logger_wrapper(gym.logger.info, "embedding_single_token_model_api: %s", self.embedding_single_token_model_api)
        self.logger_wrapper(gym.logger.info, "embedding_pooling_model_api: %s", self.embedding_pooling_model_api)
        self.logger_wrapper(gym.logger.info, "eval_model_api: %s", self.eval_model_api)

        ## Other API parameters
        self.embedding_pooling_model_method = utils.dict_or_default(kwargs, "embedding_pooling_model_method", "mean")
        self.embedding_pooling_model_layer = utils.dict_or_default(kwargs, "embedding_pooling_model_layer", -1)
        self.l2_normalize_api_embeddings = utils.dict_or_default(kwargs, "l2_normalize_api_embeddings", True)

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
        self.translation_candidates_exploration_rate = utils.dict_or_default(kwargs, "translation_candidates_exploration_rate", 1.0) # UCB c
        self.translation_candidates_reward_mean_exponential_decay_alpha = utils.dict_or_default(kwargs, "translation_candidates_reward_mean_exponential_decay_alpha", 0.1) # alpha for exponential decay

        # Env configuration
        self.logger_wrapper(gym.logger.debug, "State and action embedding size: %d %d", self.state_dim, self.action_dim)
        self.logger_wrapper(gym.logger.info, "Model hidden size and EoS token (you may need to specify the correct values according to your LLM): %d %s", self.model_hidden_size, self.eos_token_str)

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
            reward = 0.0

            return observation, reward, terminated, truncated, info

        self.time_step += 1
        self.time_step_global += 1

        if isinstance(action, str):
            # Sentence given
            action_url = action
            action_url_idx = None

            for idx, url in self.icl_embeddings_representation.items():
                if url == action_url:
                    action_url_idx = idx

                    break

            if action_url_idx is None:
                raise Exception(f"URL not found: {action_url}")

            action_url_distance = 0.0
            action_url_idx = np.array([[action_url_idx]])
        else:
            # Embedding given
            action_url, action_url_distance, action_url_idx = self.get_closest_neighbors_urls(action)
            action_url = action_url[0][0]
            action_url_distance = action_url_distance[0][0]

        assert action_url_idx.shape == (1, 1), action_url_idx.shape
        assert action_url == self.icl_embeddings_representation[action_url_idx[0][0]], f"{action_url} vs {self.icl_embeddings_representation[action_url_idx[0][0]]}"
        assert action_url_idx[0][0] == self.icl_embeddings_representation_icl2idx[action_url], f"{action_url_idx[0][0]} vs {self.icl_embeddings_representation_icl2idx[action_url]}"

        terminated, truncated, reward = self.apply_step(action_url)
        #representation = self.str2representation[action_url] # former self.get_url_representation(action_url, apply_model=True).squeeze(0)

        #assert representation.shape == (self.action_dim,), representation.shape

        #self.last_representation_str.append(representation)
        #self.last_representation_emb.append(action_url)

        self.rewards.append(reward)
        self.logger_wrapper(gym.logger.info, "Action in time step #%d (reward: %s; distance: %s): %s",
                            self.time_step, reward, action_url_distance, action_url)

        #previous_observation = self.state_window_type_callback(self.current_state_window) # former: before adding the new observation, code which have been removed
        ## ...
        observation = self.state_window_type_callback(self.current_state_window)

        #self.new_observation(observation, list(self.last_representation_str), list(self.last_representation_emb), children_urls, previous_observation)

        _terminated, _truncated = self.is_done()

        assert _terminated == terminated, f"Expected terminated: {_terminated}, got: {terminated}"
        assert _truncated == truncated, f"Expected truncated: {_truncated}, got: {truncated}"

        if terminated or truncated:
            average_reward = -np.inf if len(self.rewards) == 0 else (sum(self.rewards) / len(self.rewards))

            self.logger_wrapper(gym.logger.info, "Average reward for %d steps: %s (sum: %s)", self.time_step, average_reward, sum(self.rewards))
            reward_sum = sum(self.translation_candidates_reward_mean_episode)
            reward_steps = len(self.translation_candidates_reward_mean_episode)
            reward_mean = reward_sum / reward_steps

            self.logger_wrapper(gym.logger.info, "All episodes statistics: {'sum': %s, 'mean': %s, 'last_episode_reward': %s, 'last_episode_steps': %s}", reward_sum, reward_mean, reward, self.time_step)

        sys.stdout.flush()
        sys.stderr.flush()

        return observation, reward, terminated, truncated, info

    def _hard_reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.logger_wrapper(gym.logger.debug, "Env reset (hard): episode %d", self.episode)

        self.time_step_global = 0
        self.data = []
        self.data_icl_examples = []
        #self.observation_hash_dict = {} # We do not remove this data in _soft_reset because the replay buffer is not reseted after an episode ends
        self.str2representation = {} # Needed for knn search

        # Load data
        self.load_data() # TODO this was previously in _soft_reset and called only after a hard reset (why not directly here?)

        assert len(self.data) > 0, "Data must not be empty"
        assert len(self.data_icl_examples) > 0, "ICL examples must not be empty"

        # Shuffle data in order to avoid model memorization
        random.shuffle(self.data)
        random.shuffle(self.data_icl_examples)

        # Insert all ICL examples in the embeddings index
        ## This should be placed in self._soft_reset if the index changes after each episode (e.g., ICL examples are removed during the episode)
        self.embeddings_index = faiss.IndexFlatL2(self.action_dim)
        self.icl_embeddings_representation = {} # former self.active_urls_representation # idx (insertion order) to embedding
        self.icl_embeddings_representation_icl2idx = {} # former self.active_urls_representation_url2idx

        ## ICL examples
        representations_str = [f"{src_icl}\t{trg_icl}" for src_icl, trg_icl in self.data_icl_examples]
        representations_emb = self.get_icl_example_representation(self.data_icl_examples)
        self.insert_embeddings(representations_str, representations_emb)

        if len(representations_str) != len(set(representations_str)):
            self.logger_wrapper(gym.logger.warn, "Duplicate ICL example representations found: %d", len(representations_str) - len(set(representations_str)))

        for _str, emb in zip(representations_str, representations_emb):
            assert emb.shape[0] == self.action_dim, f"Expected embedding shape {self.action_dim}, got {emb.shape[0]} for {_str}"

            self.str2representation[_str] = emb

        ## EoS token (early stopping action)
        representations_str = [self.eos_token_str]
        representations_emb = self.get_token_representation(representations_str)
        self.insert_embeddings(representations_str, representations_emb)

        for _str, emb in zip(representations_str, representations_emb):
            assert emb.shape[0] == self.action_dim, f"Expected token shape {self.action_dim}, got {emb.shape[0]} for {_str}"

            self.str2representation[_str] = emb

        # Variables for selecting the translation sentences for the episodes
        self.translation_candidate = -1 # dummy value in order to skip repetition of current ICL example the first time
        self.translation_candidates_selected_episode = np.array([0] * len(self.data_icl_examples)) # N_episode
        self.translation_candidates_selected_consecutive_episode = np.array([0] * len(self.data_icl_examples)) # N'_episode
        self.translation_candidates_reward_mean_exponential_decay_episode = np.array([0.0] * len(self.data_icl_examples)) # Q_episode

        # Other
        self.translation_candidates_reward_mean_episode = []

        # soft reset
        observation, info = self._soft_reset(seed, {**(options if isinstance(options, dict) else {}), **{"reset_from_hard_reset": True}})

        return observation, info

    def _soft_reset(self, seed=None, options=None):
        # Difference with self._hard_reset: we keep all the results from the models to avoid computing them again
        options = {} if options is None else options

        assert isinstance(options, dict), f"Options must be a dictionary, got {type(options)}: {options}"

        is_hard_reset = utils.dict_or_default(options, "reset_from_hard_reset", False)
        is_soft_reset = not is_hard_reset

        if is_soft_reset:
            super().reset(seed=seed)

            self.logger_wrapper(gym.logger.debug, "Env reset (soft): episode %d", self.episode)

        info = {}
        self.time_step = 0
        #self.current_translations = 0 # former self.current_downloaded_urls
        self.current_icl_examples = []
        self.current_state_window = collections.deque(maxlen=self.state_window_length)
        self.rewards = []
        self.early_stopping = False
        #self.last_representation_str = [] # former self.last_downloaded_url_representation_url
        #self.last_representation_emb = [] # former self.last_downloaded_url_representation
        self.current_datetime = datetime.datetime.now()

        for _ in range(self.state_window_length):
            self.current_state_window.append(np.zeros(self.model_hidden_size))

        # Select translation sentence for the episode
        self.translation_candidate = self.get_translation_candidate() # this function must be called at the beginning of each episode

        # Get the observation for the first time step
        src_sentence = self.data[self.translation_candidate][0]
        observation = self.get_translations([src_sentence], only_representation=True)[0]

        self.current_state_window.append(observation)

        observation = self.state_window_type_callback(self.current_state_window)

        return observation, info

    def reset(self, seed=None, options=None):
        self.episode += 1
        options = {} if options is None else options

        assert isinstance(options, dict), f"Options must be a dictionary, got {type(options)}: {options}"

        if seed is not None:
            utils.set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)

        if self.reset_times == 0 or utils.dict_or_default(options, "always_hard_reset", False):
            observation, info = self._hard_reset(seed=seed, options=options)
        else:
            # After first reset, _soft_reset is the default option if "always_hard_reset" is not defined in options
            observation, info = self._soft_reset(seed=seed, options=options)

        assert observation.shape == (self.state_dim,), observation.shape

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

    def insert_embeddings(self, urls, embeddings, _index=None, _urls_representation=None, _urls_representation_url2idx=None, update_representation=True):
        assert isinstance(urls, list), f"Expected urls to be a list, got {type(urls)}: {urls}"
        assert len(urls) > 0, "urls must not be an empty list"
        assert isinstance(urls[0], str), f"Expected urls to be a list of strings, got {type(urls[0])}: {urls[0]}"

        #embeddings = utils.embeddings_index_sanity_check(embeddings, last_dimmension_shape=self.action_dim)
        index = self.embeddings_index if _index is None else _index
        urls_representation = self.icl_embeddings_representation if _urls_representation is None else _urls_representation
        urls_representation_url2idx = self.icl_embeddings_representation_icl2idx if _urls_representation_url2idx is None else _urls_representation_url2idx

        utils.insert_embeddings(urls, embeddings, index, urls_representation, urls_representation_url2idx, self.action_dim, update_representation=update_representation)

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
            avg_eval_values, single_eval_values = self.comet_eval(src_sentence, translation, reference)
            reward = avg_eval_values

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

        if self.l2_normalize_api_embeddings:
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

        if self.l2_normalize_api_embeddings:
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

            if self.l2_normalize_api_embeddings:
                translations = utils.l2_normalize(translations)

        assert len(translations) == len(src_sentences), f"Translations length mismatch: {len(translations)} vs {len(src_sentences)}"

        return translations

    def is_done(self):
        limit_examples = len(self.current_icl_examples) >= self.max_icl_examples
        terminated = limit_examples
        truncated = self.early_stopping

        assert len(self.icl_embeddings_representation) == self.embeddings_index.ntotal

        return terminated, truncated

    def get_closest_neighbors_urls(self, proto_actions, k=1, distance_expected_zero=False, get_url=True, observations=None,
                                   _index=None, _urls_representation=None, _urls_representation_are_embeddings=False):
        """
            observations: states from which proto_actions were generated
        """
        proto_actions = utils.embeddings_index_sanity_check(proto_actions, last_dimmension_shape=self.action_dim)
        results = []
        index = self.embeddings_index if _index is None else _index
        urls_representation = self.icl_embeddings_representation if _urls_representation is None else _urls_representation

        assert isinstance(k, int), k

        assert observations is None, "You only need this argument if you want to reconstruct the index based on previous observations, which is not implemented yet"

        if observations is not None:
            # Create and reconstruct index to perform search using A(provided_state) set and not A(current_state) set (here, capital A is the set of actions)
            pass # Given that the set of actions do not change given a state, we do not need to reconstruct the index here

        if index.ntotal == 0:
            # Faiss index is empty
            self.logger_wrapper(gym.logger.warn, "Faiss index seems to be empty: %d (sentences in pool: %d)", index.ntotal, len(urls_representation))

        #self.logger_wrapper(gym.logger.debug, "Default representation is %s", self.eos_token_str)

        D, I = index.search(proto_actions, k) # [D]istance, [I]ndex
        _fake_representation_str = self.eos_token_str # Default representation if no hits are found # TODO would be better to use the first or a random entry?
        _fake_representation = self.str2representation[_fake_representation_str] if _urls_representation_are_embeddings else _fake_representation_str

        for i in I:
            results.append([])

            for idx in i:
                if len(results[-1]) >= k:
                    break
                if idx < 0:
                    break

                url = urls_representation[idx]

                results[-1].append(url)

            if len(results[-1]) == 0:
                # Use seed URL (we need to add something...)
                self.logger_wrapper(gym.logger.warn, "No entries close: returning default representation (%s)", _fake_representation_str)

                while len(results[-1]) < k:
                    results[-1].append(_fake_representation)
            elif len(results[-1]) != k:
                # Add items to avoid tensor errors because dimensions don't match
                while len(results[-1]) < k:
                    results[-1].append(results[-1][-1])

            assert len(results[-1]) == k

        if distance_expected_zero:
            for idx1, d1 in enumerate(D):
                for idx2, d2 in enumerate(d1):
                    if not np.isclose(d2, 0.0):
                        self.logger_wrapper(gym.logger.warn, "Expected distance was 0, but got %s in D[%d][%d]: check https://github.com/facebookresearch/faiss/issues/1272", d2, idx1, idx2)

        if not get_url:
            assert isinstance(results, list)

            if len(results) > 0:
                assert isinstance(results[0], list)

            all_urls = [q for w in results for q in w] # Flatten list of lists

            if len(all_urls) > 0:
                assert isinstance(all_urls[0], str), f"Expected all_urls to be a list of strings, got {type(all_urls[0])}: {all_urls[0]}"

            _all_urls_subset = [url for url in all_urls if url not in self.str2representation]
            _all_urls_representation = [] if len(_all_urls_subset) == 0 else self.get_icl_example_representation(_all_urls_subset)

            assert len(_all_urls_subset) == len(_all_urls_representation)

            for _child_url, _child_url_observation in zip(_all_urls_subset, _all_urls_representation):
                assert _child_url_observation.shape == (self.action_dim,)

                self.str2representation[_child_url] = _child_url_observation

            results = [torch.tensor(self.str2representation[url]) for url in all_urls]
            results = torch.stack(results, dim=0).to(self.device)

            assert len(results.shape) == 2, results.shape
            assert results.shape == (proto_actions.shape[0] * k, proto_actions.shape[1])

            results = results.reshape((proto_actions.shape[0], k, proto_actions.shape[1])).to(self.device)

        return results, D, I

    def get_translation_candidate(self):
        n = len(self.data)
        repeat = False

        # Repeat current translation candidate?
        if self.translation_candidate >= 0:
            idx = self.translation_candidate

            assert 0.0 <= self.translation_candidates_reward_mean_exponential_decay_episode[idx] <= 1.0, f"Reward mean exponential decay must be in [0, 1], got {self.translation_candidates_reward_mean_exponential_decay_episode[idx]} for idx {idx}"

            p = (1.0 - self.translation_candidates_reward_mean_exponential_decay_episode[idx]) / (1 + self.translation_candidates_selected_consecutive_episode[idx])
            #p = ((1.0 - self.translation_candidates_reward_mean_exponential_decay_episode[idx]) * np.exp(-1.0 * self.translation_candidates_selected_consecutive_episode[idx])).item()

            assert 0.0 <= p <= 1.0, f"Probability p must be in [0, 1], got {p} for idx {idx}: {self.translation_candidates_reward_mean_exponential_decay_episode} / {self.translation_candidates_selected_consecutive_episode}"

            if random.random() < p:
                self.logger_wrapper(gym.logger.info, "Translation candidate (repeating): #%d (p: %.4f)", idx, p)

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

            self.logger_wrapper(gym.logger.info, "Translation candidate: #%d (p: %.4f)", translation_candidate, prob_dist[translation_candidate])

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
        else:
            self.current_icl_examples.append(current_action.split('\t'))

            assert len(self.current_icl_examples[-1]) == 2, f"Expected current ICL example to have two elements (source and target), got {len(self.current_icl_examples[-1])}: {self.current_icl_examples[-1]}"

        terminated, truncated = self.is_done()
        reward = 0.0

        if self.early_stopping:
            assert terminated or truncated, f"Early stopping action received but not terminated or truncated: {terminated}, {truncated}"

        if terminated or truncated:
            # Compute reward
            src_sentence, reference = self.data[self.translation_candidate]
            translation = self.get_translations([src_sentence], icl_examples=[self.current_icl_examples])[0]
            reward = self.get_reward(src_sentence, reference, translation=translation)

            # Update translation candidate mean reward
            previous_value = self.translation_candidates_reward_mean_exponential_decay_episode[self.translation_candidate]
            self.translation_candidates_reward_mean_exponential_decay_episode[self.translation_candidate] = \
                previous_value + self.translation_candidates_reward_mean_exponential_decay_alpha * (reward - previous_value)
            self.translation_candidates_reward_mean_episode.append(reward)

            return terminated, truncated, reward

        # Update state
        src_sentence = self.data[self.translation_candidate][0]

        if self.state_representation  == "translation_and_icl_examples":
            observation = self.get_translations([src_sentence], icl_examples=[self.current_icl_examples], only_representation=True)[0]
        elif self.state_representation == "actions":
            observation = self.get_icl_example_representation([self.current_icl_examples[-1]])
        else:
            raise Exception(f"Unknown state representation: {self.state_representation}")

        self.current_state_window.append(observation)

        # Return
        terminated, truncated = self.is_done()
        reward = 0.0

        assert not terminated and not truncated, "Step should not terminate or truncate immediately after applying an action"

        return terminated, truncated, reward

if __name__ == "__main__":
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    file_data = sys.argv[3]
    file_data_icl_examples = sys.argv[4]
    raw_kwargs = sys.argv[5:]
    parsed_kwargs = {}

    for arg in raw_kwargs:
        key, sep, value = arg.partition("=")

        assert sep == '=', f"Invalid argument format: {arg}"

        parsed_kwargs[key] = value

    env = MTICLEnv(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)

    check_env(env)
