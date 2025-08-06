
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
        self.gamma = utils.dict_or_default(kwargs, "gamma", 0.99)
        self.state_window_length = utils.dict_or_default(kwargs, "state_window_length", 4)
        self.state_window_type = utils.dict_or_default(kwargs, "state_window_type", "concatenate")
        self.max_icl_examples = utils.dict_or_default(kwargs, "max_icl_examples", 4)
        self.max_data_entries = utils.dict_or_default(kwargs, "max_data_entries", -1)
        self.max_data_icl_examples_entries = utils.dict_or_default(kwargs, "max_data_icl_examples_entries", -1)

        if self.state_window_length < self.max_icl_examples:
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
        self.icl_embeddings_representation = {} # former self.active_urls_representation
        self.icl_embeddings_representation_icl2idx = {} # former self.active_urls_representation_url2idx

        ## ICL examples
        representations_str = [f"{src_icl}\t{trg_icl}" for src_icl, trg_icl in self.data_icl_examples]
        representations_emb = self.get_icl_example_representation(self.data_icl_examples)
        self.insert_embeddings(representations_str, representations_emb)

        ## EoS token (early stopping action)
        representations_str = [self.eos_token_str]
        representations_emb = self.get_token_representation(representations_str)
        self.insert_embeddings(representations_str, representations_emb)

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
        self.current_translations = 0 # former self.current_downloaded_urls TODO
        self.current_state_window = collections.deque(maxlen=self.state_window_length)
        self.rewards = []
        self.current_datetime = datetime.datetime.now()

        for _ in range(self.state_window_length):
            self.current_state_window.append(np.zeros(self.model_hidden_size))

        observation = self.state_window_type_callback(self.current_state_window)

        # TODO get initial observation from self.data using self.time_step

        return observation, info

    def reset(self, seed=None, options=None):
        self.episode += 1
        options = {} if options is None else options

        assert isinstance(options, dict), f"Options must be a dictionary, got {type(options)}: {options}"

        if seed is not None:
            utils.set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)

        if self.reset_times == 0 or utils.dict_or_default(options, "always_hard_reset", False):
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

    def insert_embeddings(self, urls, embeddings, _index=None, _urls_representation=None, _urls_representation_url2idx=None, update_representation=True):
        # TODO urls is a list of elements with format src_sentence<tab>trg_sentence (or EoS token)
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

        return representations

if __name__ == "__main__":
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    file_data = sys.argv[3]
    file_data_icl_examples = sys.argv[4]

    env = MTICLEnv(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG)

    check_env(env)
