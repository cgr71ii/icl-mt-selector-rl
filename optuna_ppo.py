
import sys
from datetime import datetime
import contextlib
import logging
import pickle
import os

import utils
import gym_env_run_experiments_v2_discrete_ppo as environment_script

import optuna

current_datetime = datetime.now().strftime("%Y%m%d_%H%M")

def objective(trial):
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    file_data = sys.argv[3]
    file_data_icl_examples = sys.argv[4]
    parsed_kwargs = utils.parse_args(sys.argv[5:])
    filename = f"optuna_log/ppo/{current_datetime}_trial_{trial.number}.log"

    assert "max_steps" not in parsed_kwargs
    assert "num_envs" not in parsed_kwargs
    assert "disable_eval" not in parsed_kwargs
    assert "patience" not in parsed_kwargs
    assert "eval_freq" not in parsed_kwargs

    parsed_kwargs["max_steps"] = 500000
    parsed_kwargs["num_envs"] = 80
    parsed_kwargs["disable_eval"] = False
    parsed_kwargs["patience"] = 5
    parsed_kwargs["eval_freq"] = 20000

    # Build params
    environment_args = [src_lang, trg_lang, file_data, file_data_icl_examples]

    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [200, 400, 800])
    n_steps = trial.suggest_categorical("n_steps", [10, 20, 50, 100])
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2])
    gae_lambda = trial.suggest_float("gae_lambda", 0.95, 1.0)
    net_arch_str = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "gelu"])
    #linear_bottleneck = trial.suggest_categorical("linear_bottleneck", [128, 256, 512])
    embedding_pooling_model_method_state = trial.suggest_categorical("embedding_pooling_model_method_state", ["last", "mean"])
    embedding_pooling_model_layer = trial.suggest_categorical("embedding_pooling_model_layer", ["50%", "60%", "70%", "80%", "90%", "100%"])
    n_epochs = trial.suggest_int("n_epochs", 4, 16)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    pi_factor = 4
    enable_target_kl = trial.suggest_categorical("enable_target_kl", [True, False])

    if net_arch_str == "small":
        linear_bottleneck = 128
        net_arch = {
            "pi": [64 * pi_factor, 64 * pi_factor],
            "vf": [64, 64]
        }
    elif net_arch_str == "medium":
        linear_bottleneck = 256
        net_arch = {
            "pi": [128 * pi_factor, 128 * pi_factor],
            "vf": [128, 128]
        }
    elif net_arch_str == "large":
        linear_bottleneck = 512
        net_arch = {
            "pi": [256 * pi_factor, 256 * pi_factor],
            "vf": [256, 256]
        }
    else:
        raise ValueError(f"Invalid net_arch_str: {net_arch_str}")

    # Check that the hyperparameters are not already set in parsed_kwargs
    assert "rl_activation_fn" not in parsed_kwargs
    assert "ent_coef" not in parsed_kwargs
    assert "net_arch" not in parsed_kwargs
    assert "linear_bottleneck" not in parsed_kwargs
    assert "gae_lambda" not in parsed_kwargs
    assert "clip_range" not in parsed_kwargs
    assert "rl_batch_size" not in parsed_kwargs
    assert "n_steps" not in parsed_kwargs
    assert "embedding_pooling_model_method_state" not in parsed_kwargs
    assert "embedding_pooling_model_layer" not in parsed_kwargs
    assert "n_epochs" not in parsed_kwargs
    assert "vf_coef" not in parsed_kwargs

    if enable_target_kl:
        assert "target_kl" not in parsed_kwargs

        parsed_kwargs["target_kl"] = trial.suggest_float("target_kl", 0.01, 0.1)
    else:
        parsed_kwargs["target_kl"] = None

    # Set hyperparameters
    parsed_kwargs["rl_activation_fn"] = activation_fn
    parsed_kwargs["ent_coef"] = ent_coef
    parsed_kwargs["net_arch"] = net_arch
    parsed_kwargs["linear_bottleneck"] = linear_bottleneck
    parsed_kwargs["gae_lambda"] = gae_lambda
    parsed_kwargs["clip_range"] = clip_range
    parsed_kwargs["rl_batch_size"] = batch_size
    parsed_kwargs["n_steps"] = n_steps
    parsed_kwargs["embedding_pooling_model_method_state"] = embedding_pooling_model_method_state
    parsed_kwargs["embedding_pooling_model_layer"] = embedding_pooling_model_layer
    parsed_kwargs["n_epochs"] = n_epochs
    parsed_kwargs["vf_coef"] = vf_coef

    print(f"Trial {trial.number}: logging to {filename}")

    # Run training and evaluation
    with open(filename, "wt") as fd:
        with contextlib.redirect_stdout(fd), contextlib.redirect_stderr(fd):
            for logger_name in (None, "MT_ICL.rl_experiments", "gymnasium", "stable_baselines3"):
                root_logger = logging.getLogger(logger_name)
            
                # remove existing handlers
                for h in root_logger.handlers[:]:
                    root_logger.removeHandler(h)

                # add file handler
                file_handler = logging.FileHandler(filename)
                file_handler.setLevel(logging.DEBUG)

                root_logger.addHandler(file_handler)
                root_logger.setLevel(logging.DEBUG)

            assert "redirect_output_filename" not in parsed_kwargs

            parsed_kwargs["redirect_output_filename"] = filename

            print(f"Trial {trial.number} ({current_datetime})")

            try:
                final_reward = environment_script.main(*environment_args, **parsed_kwargs)
            except Exception as e:
                print(f"Trial {trial.number} failed with exception: {e}")

                raise e

    return final_reward

if __name__ == "__main__":
    if "OPTUNA_LOAD_STUDY_NAME" in os.environ:
        load_study = True
        study_name = os.environ["OPTUNA_LOAD_STUDY_NAME"]

        print("Loading study with name: %s", study_name)

        if "OPTUNA_LOAD_STUDY_SAMPLER_FILENAME" in os.environ:
            sampler_fn = os.environ["OPTUNA_LOAD_STUDY_SAMPLER_FILENAME"]

            with open(sampler_fn, "rb") as fd:
                sampler = pickle.load(fd)

            print(f"Loaded sampler from {sampler_fn}")
    else:
        load_study = False
        study_name = f"ppo_hyperparameter_optimization_{current_datetime}"
        sampler = optuna.samplers.TPESampler()

    save_path = "optuna_data"
    #pruner = optuna.pruners.MedianPruner()
    pruner = None # let's rely on early stopping based on patience instead of pruning
    study = optuna.create_study(sampler=sampler, pruner=pruner, study_name=study_name, storage="sqlite:///db.ppo.sqlite3",
                                direction="maximize", load_if_exists=load_study)

    print(f"Sampler is {study.sampler.__class__.__name__}")
    print(f"Pruner is {study.pruner.__class__.__name__}{' (disabled)' if pruner is None else ''}")

    try:
        study.optimize(objective, n_trials=500, show_progress_bar=False, n_jobs=1)
    finally:
        sampler_fn = f"{study.study_name}_sampler.pkl"

        with open(os.path.join(save_path, sampler_fn), "wb") as fd:
            pickle.dump(study.sampler, fd)

        print(f"Study name: {study.study_name}")
        print(f"  Sampler stored in {sampler_fn}")
        print( "Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best value (mean reward dev): {study.best_trial.value}")
        print( "  Best hyperparameters:")

        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
