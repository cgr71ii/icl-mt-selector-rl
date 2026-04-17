
import math
import sys
from datetime import datetime
import contextlib
import logging
import pickle
import os
import time
import socket
from collections import Counter

import utils
import gym_env_run_experiments_v2_discrete_ppo as environment_script

import optuna
import psutil
import numpy as np

current_datetime = datetime.now().strftime("%Y%m%d_%H%M")

def print_open_files(tag=""):
    process = psutil.Process(os.getpid())
    num_files = process.num_fds()

    print(f"[{tag}] Currently open files: {num_files}")

def print_process_resources(tag="", max_items=10):
    print(f"\n[{tag}] Resource usage for process {os.getpid()}:")

    p = psutil.Process(os.getpid())

    try:
        num_fds = p.num_fds()
    except Exception as e:
        num_fds = f"unavailable ({e})"

    print(f"[{tag}] PID={p.pid} num_fds={num_fds}")

    # 1) Regular files opened by this process
    try:
        files = p.open_files()
    except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
        files = []
        print(f"[{tag}] open_files unavailable: {e}")

    print(f"[{tag}] open_files count: {len(files)}")
    for f in files[:max_items]:
        # f has: path, fd, position, mode, flags
        print(f"[{tag}]   file fd={f.fd} mode={f.mode} path={f.path}")

    # 2) Socket connections owned by this process
    try:
        if hasattr(p, "net_connections"):
            conns = p.net_connections(kind="all")
        else:
            conns = p.connections(kind="all")
    except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
        conns = []
        print(f"[{tag}] connections unavailable: {e}")

    print(f"[{tag}] connections count: {len(conns)}")

    status_counts = Counter(c.status for c in conns)
    family_counts = Counter(
        "AF_UNIX" if c.family == socket.AF_UNIX else
        "AF_INET" if c.family == socket.AF_INET else
        "AF_INET6" if c.family == socket.AF_INET6 else
        str(c.family)
        for c in conns
    )
    type_counts = Counter(
        "STREAM" if c.type == socket.SOCK_STREAM else
        "DGRAM" if c.type == socket.SOCK_DGRAM else
        str(c.type)
        for c in conns
    )

    print(f"[{tag}] connection status counts: {dict(status_counts)}")
    print(f"[{tag}] connection family counts: {dict(family_counts)}")
    print(f"[{tag}] connection type counts: {dict(type_counts)}")

    for c in conns[:max_items]:
        # c has: fd, family, type, laddr, raddr, status, pid
        print(
            f"[{tag}]   conn fd={c.fd} family={c.family} type={c.type} "
            f"status={c.status} laddr={c.laddr} raddr={c.raddr}"
        )

    # 3) Child processes
    print(f"[{tag}] Child process count: {len(p.children(recursive=True))}")

    for child in p.children(recursive=True)[:max_items]:
        try:
            child_info = f"pid={child.pid} name={child.name()} status={child.status()}"
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            child_info = f"pid={child.pid} info unavailable: {e}"

        print(f"[{tag}]   child process: {child_info}")

    print(f"[{tag}] End of resource report\n")

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
    assert "optuna_trial" not in parsed_kwargs
    assert "skip_last_eval" not in parsed_kwargs

    skip_last_eval = True
    parsed_kwargs["max_steps"] = 200000
    parsed_kwargs["num_envs"] = 80
    parsed_kwargs["disable_eval"] = False
    parsed_kwargs["patience"] = 99999 # disable -> rely on pruning
    parsed_kwargs["eval_freq"] = 20000
    parsed_kwargs["optuna_trial"] = trial
    parsed_kwargs["skip_last_eval"] = skip_last_eval

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
    assert "learning_rate" not in parsed_kwargs
    assert "activation_fn" not in parsed_kwargs
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
    parsed_kwargs["learning_rate"] = learning_rate
    parsed_kwargs["activation_fn"] = activation_fn
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
    print_process_resources(f"Start of trial {trial.number}")

    # Check intermediate values for MedianPruner (optional, for debugging)
    if trial.study.pruner is not None and isinstance(trial.study.pruner, optuna.pruners.MedianPruner):
        if trial.number > 0:
            #print(f"Trial {trial.number}: checking intermediate values for MedianPruner")

            # Code adapted from https://optuna.readthedocs.io/en/stable/_modules/optuna/pruners/_percentile.html#PercentilePruner

            completed_trials = trial.study.get_trials(deepcopy=False, states=(optuna.trial._state.TrialState.COMPLETE,))
            direction = trial.study.direction
            _percentile = trial.study.pruner._percentile
            _n_min_trials = trial.study.pruner._n_min_trials
            percentile_results = {}
            percentile_result = 0.0
            step = 0

            while not math.isnan(percentile_result):
                percentile_result = optuna.pruners._percentile._get_percentile_intermediate_result_over_trials(completed_trials, direction, step, _percentile, _n_min_trials)

                if not math.isnan(percentile_result):
                    percentile_results[step] = percentile_result

                step += 1
        else:
            percentile_results = {}

        print(f"Trial {trial.number}: MedianPruner intermediate results at each step: {percentile_results}")

    sys.stdout.flush()
    sys.stderr.flush()

    logger_modules = (None, "MT_ICL.rl_experiments", "gymnasium", "stable_baselines3")

    # Run training and evaluation
    with open(filename, "wt") as fd:
        try:
            with contextlib.redirect_stdout(fd), contextlib.redirect_stderr(fd):
                for logger_name in logger_modules:
                    root_logger = logging.getLogger(logger_name)
                
                    # remove existing handlers
                    for h in root_logger.handlers[:]:
                        h.close()
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
        finally:
            for logger_name in logger_modules:
                root_logger = logging.getLogger(logger_name)

                # remove file handlers
                for h in root_logger.handlers[:]:
                    h.close()
                    root_logger.removeHandler(h)

            logging.shutdown()

    time.sleep(2) # give some time for file handlers to release the files
    print_process_resources(f"End of trial {trial.number} (after cleanup)")

    frozen_trial = trial.study._storage.get_trial(trial._trial_id)
    final_reward2 = optuna.pruners._percentile._get_best_intermediate_result_over_steps(frozen_trial, trial.study.direction)
    intermediate_results = np.asarray(list(frozen_trial.intermediate_values.values()), dtype=float)

    assert len(intermediate_results.shape) == 1, f"Expected intermediate_results to be 1D array, but got shape {intermediate_results.shape}"
    assert intermediate_results.shape[0] > 0, "Expected at least one intermediate result, but got zero"

    if skip_last_eval:
        assert np.isclose(final_reward, 0.0), f"final_reward from environment_script should be 0.0 when skip_last_eval is True, but got {final_reward}"

        final_reward = final_reward2
    else:
        assert np.isclose(final_reward, final_reward2), f"final_reward from environment_script ({final_reward}) does not match best intermediate result from Optuna ({final_reward2})"

    best_trial_strategy = os.environ["OPTUNA_BEST_TRIAL_STRATEGY"] if "OPTUNA_BEST_TRIAL_STRATEGY" in os.environ else "mean_intermediate_result_last_2"

    if best_trial_strategy == "best_intermediate_result":
        final_reward = final_reward2
    elif best_trial_strategy == "final_reward":
        final_reward = intermediate_results[-1].item()
    elif best_trial_strategy == "mean_intermediate_result_last_2":
        final_reward = np.mean(intermediate_results[-2:]).item()
    else:
        raise Exception("Invalid OPTUNA_BEST_TRIAL_STRATEGY: %s", best_trial_strategy)

    intermediate_results_dict = {step: reward for step, reward in frozen_trial.intermediate_values.items()}

    print(f"Trial {trial.number}: intermediate results (strategy: {best_trial_strategy}) at each step: {intermediate_results_dict}")
    print(f"Trial {trial.number}: best intermediate result: {final_reward2}")

    sys.stdout.flush()
    sys.stderr.flush()

    return final_reward

if __name__ == "__main__":
    sampler = None

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

    if sampler is None:
        sampler = optuna.samplers.TPESampler(n_startup_trials=30, multivariate=True, seed=42)

    save_path = "optuna_data"
    #pruner = None # let's rely on early stopping based on patience instead of pruning
    # MedianPruner: prune trials that have intermediate results worse than the median of previous trials at the same step (with "step" we mean the parameter passed to trial.report)
    ## n_startup_trials trials before pruning, n_warmup_steps intermediate evaluation steps before pruning, evaluate every interval_steps evaluation steps, n_min_trials reported intermediate results at each step before pruning
    ## 3 complete trials, check pruning after 1 intermediate evaluation step at each trial, check pruning every 1 step, require at least 5 intermediate results at the step before pruning in that step
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1, n_min_trials=5)
    study = optuna.create_study(sampler=sampler, pruner=pruner, study_name=study_name, storage="sqlite:///db.ppo.sqlite3",
                                direction="maximize", load_if_exists=load_study)

    print(f"Sampler is {study.sampler.__class__.__name__}: {study.sampler.__dict__}")
    print(f"Pruner is {study.pruner.__class__.__name__}{' (disabled)' if pruner is None else ''}: {study.pruner.__dict__ if study.pruner is not None else '-'}")

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
