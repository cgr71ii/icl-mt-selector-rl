
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

class MTICLEvalSingleEpisodeEnv(gym_env.MTICLEnv):
    """
        Custom Environment for evaluating a policy that selects ICL examples for MT using LLMs.

        BE AWARE that many variables might not work as intended as we inherit from the training environment and we might have not updated some variables properly
    """

    def logger_wrapper(self, callback, _str, *args, **kwargs):
        super().logger_wrapper(callback, f"[EVALENV] {_str}", *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # Preprocess action and apply step
        action_url, action_url_idx, action_url_distance = self.preprocess_action(action)
        translation_performed, reward = self.apply_step(action_url)

        if translation_performed:
            self.logger_wrapper(gym.logger.info, "Action in time step #%d (reward: %s; distance: %s): %s",
                                self.time_step, reward, action_url_distance, action_url)

        observation = self.state_window_type_callback(self.current_state_window)

        if translation_performed:
            self.translation_candidate += 1

            self.reset() # TODO check out how to do this

        terminated, truncated = self.is_done()

        if terminated or truncated:
            reward_sum = sum(self.translation_candidates_reward_mean_episode)
            reward_steps = len(self.translation_candidates_reward_mean_episode)
            reward_mean = reward_sum / reward_steps

            self.logger_wrapper(gym.logger.info, "All episodes statistics: {'sum': %s, 'mean': %s, 'last_episode_reward': %s, 'last_episode_steps': %s}", reward_sum, reward_mean, reward, self.time_step)

        sys.stdout.flush()
        sys.stderr.flush()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        assert isinstance(options, dict) or isinstance(options, type(None)), f"Options must be a dictionary or None, got {type(options)}: {options}"

        options = {} if options is None else options
        _seed = seed if self.reset_times == 0 else None

        if self.reset_times == 0:
            assert self.episode == 0, f"Reset times is 0, but episode is {self.episode}. This should not happen."

            if "shuffle_all_data" not in options:
                options["shuffle_all_data"] = False # deterministic sweeping
        elif (self.reset_times + 1) % (len(self.data) + 1) == 0: # this avoids off-by-one problem with self.episode since EvalCallback does a last reset...
            self._init_translation_candidate_variables()

            self.episode = 0
            self.reset_times += 1 # to avoid infinite loop

            for _ in range(self.state_window_length):
                self.current_state_window.append(np.zeros(self.model_hidden_size))

            observation = self.state_window_type_callback(self.current_state_window)
            info = {}

            self.logger_wrapper(gym.logger.info, "Resetting environment (fake)")

            return observation, info # fake

        observation, info = super().reset(seed=_seed, options=options)

        assert self.translation_candidate == self.episode - 1, f"Expected translation candidate to be {self.episode - 1}, but got {self.translation_candidate}"

        return observation, info

    def is_done(self):
        terminated = self.translation_candidate >= len(self.data)
        truncated = terminated

        return terminated, truncated

    def get_translation_candidate(self):
        # We assume that self.translation_candidate is initialized to -1
        translation_candidate = (self.translation_candidate + 1) % len(self.data) # sequential sweep
        src_translation_candidate = self.data[translation_candidate][0]

        self.logger_wrapper(gym.logger.info, "Translation candidate #%d: %s", translation_candidate, src_translation_candidate)

        return translation_candidate

    def apply_step(self, current_action):
        assert isinstance(current_action, str), f"Expected current_action to be a string, got {type(current_action)}: {current_action}"
        assert len(self.current_icl_examples) < self.max_icl_examples, f"Current length of ICL examples ({self.current_icl_examples}) must be less than max ICL examples ({self.max_icl_examples})"

        translation_performed = False

        if current_action == self.eos_token_str:
            # Early stopping action
            self.logger_wrapper(gym.logger.info, "Early stopping action (%s) received in time step #%d", current_action, self.time_step)

            translation_performed = True
        else:
            self.current_icl_examples.append(current_action.split('\t'))

            assert len(self.current_icl_examples[-1]) == 2, f"Expected current ICL example to have two elements (source and target), got {len(self.current_icl_examples[-1])}: {self.current_icl_examples[-1]}"

        terminated, truncated = self.is_done()
        reward = 0.0

        assert not terminated and not truncated, f"This should never happen in the evaluation environment: {terminated}, {truncated}"

        if translation_performed:
            # Compute reward
            src_sentence, reference = self.data[self.translation_candidate]
            translation = self.get_translations([src_sentence], icl_examples=[self.current_icl_examples])[0]
            reward = self.get_reward(src_sentence, reference, translation=translation)

            # Update translation candidate mean reward
            previous_value = self.translation_candidates_reward_mean_exponential_decay_episode[self.translation_candidate]
            self.translation_candidates_reward_mean_exponential_decay_episode[self.translation_candidate] = \
                previous_value + self.translation_candidates_reward_mean_exponential_decay_alpha * (reward - previous_value)
            self.translation_candidates_reward_mean_episode.append(reward)

            return translation_performed, reward

        assert not translation_performed

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

        return translation_performed, reward

if __name__ == "__main__":
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    file_data = sys.argv[3]
    file_data_icl_examples = sys.argv[4]
    parsed_kwargs = utils.parse_args(sys.argv[5:])

    # Initialize and check environment
    env = MTICLEvalSingleEpisodeEnv(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)

    check_env(env)
