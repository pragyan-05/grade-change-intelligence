"""
recommender.py
---------------
Recommends setpoint adjustments to minimize predicted Basis Weight
deviation, using the trained model as a "surrogate" of the real process
(a common industrial technique: optimize against a fast ML surrogate
instead of the slow/expensive real plant).

Approach: constrained greedy coordinate search.
For each controllable variable, try nudging it up/down within a safe
operating range (+/- a max step per iteration) and keep the change if it
reduces |predicted deviation|. Repeat for a few rounds. This is simple,
fast, explainable to operators ("we changed X because it reduced
predicted deviation by Y"), and does not require any extra ML library.
"""

import numpy as np
import pandas as pd

from src.model import FEATURE_COLUMNS, CONTROLLABLE_FEATURES, predict_deviation

# Safe operating ranges per controllable variable (min, max, max_step_per_round)
SAFE_RANGES = {
    "machine_speed_mpm": (500, 1100, 15),
    "stock_consistency_pct": (2.0, 4.8, 0.08),
    "steam_pressure_bar": (1.5, 6.0, 0.15),
    "headbox_pressure_kpa": (10, 55, 1.5),
    "slice_opening_mm": (6, 25, 0.4),
    "stock_flow_lpm": (2000, 6500, 60),
}


def recommend_setpoints(model, current_state: dict, rounds: int = 25):
    """
    current_state: dict with all FEATURE_COLUMNS filled in (current snapshot).
    Returns: (recommended_state dict, history DataFrame, predicted_before, predicted_after)
    """
    state = dict(current_state)
    predicted_before = predict_deviation(model, state)

    history = [{"round": 0, "variable": "baseline", "predicted_deviation": predicted_before}]

    for r in range(1, rounds + 1):
        best_improvement = 0
        best_var = None
        best_value = None

        current_pred = predict_deviation(model, state)

        for var in CONTROLLABLE_FEATURES:
            lo, hi, step = SAFE_RANGES[var]
            for direction in (-1, 1):
                trial_value = state[var] + direction * step
                trial_value = float(np.clip(trial_value, lo, hi))
                if trial_value == state[var]:
                    continue
                trial_state = dict(state)
                trial_state[var] = trial_value
                trial_pred = predict_deviation(model, trial_state)

                improvement = abs(current_pred) - abs(trial_pred)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_var = var
                    best_value = trial_value

        if best_var is None or best_improvement < 0.01:
            break  # converged - no meaningful improvement found

        state[best_var] = best_value
        new_pred = predict_deviation(model, state)
        history.append({
            "round": r,
            "variable": best_var,
            "predicted_deviation": new_pred,
        })

    predicted_after = predict_deviation(model, state)
    history_df = pd.DataFrame(history)

    changes = []
    for var in CONTROLLABLE_FEATURES:
        delta = state[var] - current_state[var]
        if abs(delta) > 1e-6:
            changes.append({
                "variable": var,
                "current_value": round(current_state[var], 3),
                "recommended_value": round(state[var], 3),
                "change": round(delta, 3),
            })
    changes_df = pd.DataFrame(changes)

    return state, changes_df, history_df, predicted_before, predicted_after
