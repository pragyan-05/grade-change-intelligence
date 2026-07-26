"""
explainability.py
------------------
Provides "why did the model predict this?" answers at two levels:

1. GLOBAL: which process variables matter most across all historical data
   (RandomForest feature_importances_).

2. LOCAL: for one specific prediction (one current process snapshot),
   which variables pushed the prediction up/down and by how much.
   We use SHAP (TreeExplainer) when available since it gives exact,
   additive, theoretically-grounded attributions for tree models.
   If shap isn't installed/working, we fall back to a permutation-based
   approximation so the app never breaks.
"""

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False


def global_feature_importance(model, feature_columns):
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _fallback_local_explanation(model, feature_row: dict, feature_columns, background_df):
    """
    Simple, dependency-free local explanation:
    For each feature, hold every other feature at its historical median
    ("background") and swap in the current value, measuring the change
    in prediction. This is a crude but honest approximation of SHAP.
    """
    baseline_row = background_df[feature_columns].median().to_dict()
    baseline_pred = model.predict(pd.DataFrame([baseline_row]))[0]

    contributions = {}
    for feat in feature_columns:
        row = dict(baseline_row)
        row[feat] = feature_row[feat]
        pred = model.predict(pd.DataFrame([row]))[0]
        contributions[feat] = pred - baseline_pred

    return contributions, baseline_pred


def local_explanation(model, feature_row: dict, feature_columns, background_df):
    """
    Returns a DataFrame with columns: feature, value, contribution (gsm)
    sorted by absolute contribution, descending.
    """
    if _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            X = pd.DataFrame([feature_row])[feature_columns]
            shap_values = explainer.shap_values(X)
            contributions = dict(zip(feature_columns, shap_values[0]))
        except Exception:
            contributions, _ = _fallback_local_explanation(
                model, feature_row, feature_columns, background_df
            )
    else:
        contributions, _ = _fallback_local_explanation(
            model, feature_row, feature_columns, background_df
        )

    df = pd.DataFrame({
        "feature": list(contributions.keys()),
        "value": [feature_row[f] for f in contributions.keys()],
        "contribution_gsm": list(contributions.values()),
    })
    df["abs_contribution"] = df["contribution_gsm"].abs()
    df = df.sort_values("abs_contribution", ascending=False).drop(columns="abs_contribution")
    df = df.reset_index(drop=True)
    return df


def humanize_feature_name(name: str) -> str:
    mapping = {
        "minutes_since_transition": "Time since transition start",
        "machine_speed_mpm": "Machine speed",
        "speed_change_rate": "Speed change (ramp) rate",
        "stock_consistency_pct": "Stock consistency",
        "consistency_setpoint": "Consistency setpoint",
        "steam_pressure_bar": "Steam (dryer) pressure",
        "steam_pressure_setpoint": "Steam pressure setpoint",
        "headbox_pressure_kpa": "Headbox pressure",
        "headbox_pressure_setpoint": "Headbox pressure setpoint",
        "slice_opening_mm": "Slice opening",
        "slice_opening_setpoint": "Slice opening setpoint",
        "stock_flow_lpm": "Stock flow rate",
        "stock_flow_setpoint": "Stock flow setpoint",
    }
    return mapping.get(name, name.replace("_", " ").title())
