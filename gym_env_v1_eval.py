
import sys

import gym_env_v1 as gym_env
import utils

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import numpy as np

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

class MTICLEvalEnv(gym_env.MTICLEnv):
    """
        Custom Environment for evaluating a policy that selects ICL examples for MT using LLMs.

        BE AWARE that many variables might not work as intended as we inherit from the training environment and we might have not updated some variables properly
    """

    def logger_wrapper(self, callback, _str, *args, **kwargs):
        super().logger_wrapper(callback, f"[EVALENV] {_str}", *args, **kwargs)

    def reset(self, seed=None, options=None):
        assert isinstance(options, dict) or isinstance(options, type(None)), f"Options must be a dictionary or None, got {type(options)}: {options}"

        options = {} if options is None else options
        _seed = seed if self.reset_times == 0 else None

        if self.reset_times == 0:
            assert self.episode <= 0, f"Reset times is 0, but episode is {self.episode}. This should not happen."

            if "shuffle_all_data" not in options:
                options["shuffle_all_data"] = False # deterministic sweeping
        elif (self.reset_times + 1) % (len(self.data) + 1) == 0: # this avoids off-by-one problem with self.episode since EvalCallback does a last reset...
            self._init_translation_candidate_variables()

            self.episode = 0
            self.reset_times += 1 # to avoid infinite loop

            for _ in range(self.state_window_length):
                self.current_state_window.append(np.zeros(self.model_hidden_size // self.dimensionality_reduction_factor_state_and_action))

            observation = self.state_window_type_callback(self.current_state_window)
            info = {}

            self.logger_wrapper(gym.logger.info, "Resetting environment (fake)")

            return observation, info # fake

        observation, info = super().reset(seed=_seed, options=options)

        assert self.translation_candidate == self.episode - 1, f"Expected translation candidate to be {self.episode - 1}, but got {self.translation_candidate}"

        return observation, info

    def get_translation_candidate(self):
        # We assume that self.translation_candidate is initialized to -1
        translation_candidate = (self.translation_candidate + 1) % len(self.data) # sequential sweep
        src_translation_candidate = self.data[translation_candidate][0]

        self.logger_wrapper(gym.logger.info, "Translation candidate #%d: %s", translation_candidate, src_translation_candidate)

        return translation_candidate

if __name__ == "__main__":
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    file_data = sys.argv[3]
    file_data_icl_examples = sys.argv[4]
    parsed_kwargs = utils.parse_args(sys.argv[5:])

    # Initialize and check environment
    env = MTICLEvalEnv(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)

    check_env(env)
