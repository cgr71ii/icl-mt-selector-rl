
import sys
import logging

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

import utils
from gym_env_v1 import MTICLEnv

import gymnasium as gym
from stable_baselines3 import DDPG, TD3
import torch
import numpy as np

logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.rl_experiments"), level=logging.DEBUG)

logger.info("Provided args: %s", sys.argv)

# args
src_lang = sys.argv[1]
trg_lang = sys.argv[2]
file_data = sys.argv[3]
file_data_icl_examples = sys.argv[4]
parsed_kwargs = utils.parse_args(sys.argv[5:])

# default values
device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
max_icl_examples = parsed_kwargs.get("max_icl_examples", 4)

# set defaults in case they are not provided
parsed_kwargs["device"] = device
parsed_kwargs["max_icl_examples"] = max_icl_examples

# Environment
#k = 0.01
k = 5
env_class = MTICLEnv
batch_size = 8
gamma = 0.99
#replay_buffer_size = 1000000
replay_buffer_size = 1000 # faster debug
seed = 42
max_episodes = 20
init_training_episodes = 5
max_steps = max_episodes * max_icl_examples
init_training_steps = init_training_episodes * max_icl_examples
env = MTICLEnv(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG, **parsed_kwargs)
retrieve_embeddings = lambda proto_action, _k, observations: env.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False)[0] # Get only the result, not I or D
model = DDPG(
#model = TD3(
    "WolpertingerPolicy",
    env,
    verbose=1,
    seed=seed,
    batch_size=batch_size,
    learning_starts=init_training_steps,
    policy_kwargs={
        #"net_arch": {"pi": [384, 384], "qf": [384, 384]},
        #"net_arch": {"pi": [384, 384], "qf": [384, 384]},
        "net_arch": {"pi": [400, 300], "qf": [400, 300]}, # paper architecture
        "callback_retrieve_knn": retrieve_embeddings,
        #"callback_retrieve_knn_training": retrieve_embeddings_training,
        "k": k,
        #"add_all_knn_to_batch": True, # Faster
        "add_all_knn_to_batch": False, # Better avoid due to removal of overlapping actions
        "apply_rws_inference": False,
    },
    gamma=gamma,
    device=device,
    gradient_steps=-1, # wait until the end of the episode
    #train_freq=(1, "step"),
    #train_freq=(batch_size, "step"),
    train_freq=(1, "episode"), # original config
    buffer_size=replay_buffer_size,
)

assert init_training_episodes < max_episodes

model.learn(max_steps, log_interval=1)

#done = False
#
#while not done:
#    model.learn(download_limit, reset_num_timesteps=False)
#
#    done = env.is_done()
