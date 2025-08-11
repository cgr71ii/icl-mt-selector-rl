
import sys
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

# read data
data_to_be_translated = []

with open(file_data, "rt") as fd:
    for line in fd:
        data_to_be_translated.append(line.rstrip("\r\n"))

# default values
device = parsed_kwargs.get("device", "cuda" if utils.use_cuda() else "cpu")
max_icl_examples = parsed_kwargs.get("max_icl_examples", 4)

# set defaults in case they are not provided
parsed_kwargs["device"] = device
parsed_kwargs["max_icl_examples"] = max_icl_examples
parsed_kwargs["max_data_entries"] = -1 # load all data

# Other values
filename_time = datetime.now().strftime("%Y%m%d_%H%M")
save_freq = len(data_to_be_translated) * max_icl_examples # steps (save model approx. once per epoch)
eval_freq = 100 * max_icl_examples # steps
save_path = "./rl_models/"
name_prefix = f"rl_model_{filename_time}"
max_episodes_epochs = 8 # repeat N times
max_episodes = len(data_to_be_translated) * max_episodes_epochs

# Environment
env_class = MTICLEnv
#env_eval_class = MTICLEvalSingleEpisodeEnv
env_eval_class = MTICLEvalEnv
#k = 0.01
k = 5
batch_size = 8
gamma = 0.99 # TODO we might want to set 1.0 since we do not care about number of time steps but rather the overall performance of each translation
#replay_buffer_size = 1000000
replay_buffer_size = 10000
seed = 42
learning_rate = 1e-3
init_training_episodes = 10
#max_steps = max_episodes * max_icl_examples
max_steps = 1e10 # fake value due to callback StopTrainingOnMaxEpisodes
init_training_steps = init_training_episodes * max_icl_examples
env = env_class(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.DEBUG, **{"max_data_entries": max_data_entries, **parsed_kwargs})
env_eval = env_eval_class(src_lang, trg_lang, file_data, file_data_icl_examples, gym_logger_level=gym.logger.INFO, **{"max_data_entries": max_data_entries, **parsed_kwargs})
retrieve_embeddings = lambda proto_action, _k, observations: env.get_closest_neighbors_urls(proto_action, k=_k, get_representations_instead_of_embeddings=False)[0] # Get only the result, not I or D
callbacks = []
model = DDPG(
#model = TD3(
    "WolpertingerPolicy",
    env,
    verbose=1,
    seed=seed,
    batch_size=batch_size,
    learning_starts=init_training_steps,
    learning_rate=learning_rate,
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

#assert init_training_episodes < max_episodes

# Add callbacks
stop_train_callback = sb3_cb.StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1) # early stopping
# EvalCallback
## it returns the average "sum of undiscounted rewards" per episode (https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html)
## it does not evaluate the model performance when training finishes (we evaluate below)
callbacks.append(sb3_cb.EvalCallback(
    env_eval,
    best_model_save_path=save_path,
    log_path=save_path,
    eval_freq=eval_freq,
    #n_eval_episodes=1,
    n_eval_episodes=len(data_to_be_translated),
    callback_after_eval=stop_train_callback,
    deterministic=True,
    render=False,
    verbose=1,
))
callbacks.append(sb3_cb.CheckpointCallback( # store training data in order to resume training later
  save_freq=save_freq,
  save_path=save_path,
  name_prefix=name_prefix,
  save_replay_buffer=True,
  verbose=1,
))
callbacks.append(sb3_cb.StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1))

callback = sb3_cb.CallbackList(callbacks)

# Train
model.learn(max_steps, log_interval=1, callback=callback)

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=len(data_to_be_translated))

print(f"Mean reward: {mean_reward} +/- {std_reward}")
