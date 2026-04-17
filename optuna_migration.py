
import sys
from datetime import datetime

import optuna
from optuna.trial import create_trial, TrialState
import numpy as np

old_study_name = sys.argv[1]
average_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1

# Create and load studies
current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
new_study_name = f"ppo_hyperparameter_optimization_{current_datetime}"

# 1. Load your existing, flawed study
old_study = optuna.load_study(
    study_name=old_study_name,
    storage="sqlite:///db.ppo.sqlite3",
)

# 2. Create a brand new, clean study (can be in the same database file!)
new_study = optuna.create_study(
    study_name=new_study_name,
    storage="sqlite:///db.ppo.sqlite3",
    direction="maximize",
    load_if_exists=False,
)

# 3. Iterate through and fix the old trials
corrected_trials = []

for trial in old_study.trials:
    if trial.state in (TrialState.COMPLETE, TrialState.PRUNED):
        intermediate_values = dict(trial.intermediate_values)
        intermediate_values_np = np.asarray(list(intermediate_values.values()), dtype=float)

        print(f"Original trial {trial.number}: intermediate values = {intermediate_values}, final value = {trial.value}")
        print(f"Original trial {trial.number}: parameters = {trial.params}")

        assert len(intermediate_values) > 0, f"Trial {trial.number} has no intermediate values, cannot correct final value!"

        # Ignore trial.value and average the last 2 best intermediate values as the corrected final value
        last_step_key = max(intermediate_values.keys())
        corrected_final_value = np.mean(intermediate_values_np[-average_n:]).item()

        print(f"Correcting trial {trial.number}: original final value = {trial.value}, corrected final value = {corrected_final_value}")

        # Rebuild the trial with the corrected final value
        corrected_trial = create_trial(
            params=trial.params,
            distributions=trial.distributions,
            value=corrected_final_value,
            intermediate_values=intermediate_values,
            state=trial.state,
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs
        )
        corrected_trial.datetime_start = trial.datetime_start
        corrected_trial.datetime_complete = trial.datetime_complete
        corrected_trial.number = trial.number
        #corrected_trial._trial_id = trial._trial_id # do not modify the trial_id! (is used in the DB)

        corrected_trials.append(corrected_trial)

    elif trial.state == TrialState.FAIL:
        # If the trial crashed or was pruned, its final value doesn't matter.
        # Just copy it exactly as it is so the Pruner keeps the history.
        corrected_trials.append(trial)

# 4. Inject all the corrected trials into the new study!
new_study.add_trials(corrected_trials)

print(f"Successfully migrated {len(corrected_trials)} trials with corrected target values!")
