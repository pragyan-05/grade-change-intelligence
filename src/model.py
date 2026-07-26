"""
model.py
--------
Trains and serves the core prediction model:
    Given current process conditions during a grade change,
    predict the Basis Weight deviation (gsm) from target.

Model: RandomForestRegressor (robust, interpretable via feature_importances_
and SHAP, handles nonlinearity, needs no scaling - good for a first
production-grade baseline). Swap in XGBoost/LightGBM later without
changing the rest of the app if you keep the same predict() interface.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "historical_grade_change_data.csv"
MODEL_PATH = BASE_DIR / "models" / "bw_deviation_model.pkl"
META_PATH = BASE_DIR / "models" / "model_metadata.pkl"

FEATURE_COLUMNS = [
    "minutes_since_transition",
    "machine_speed_mpm",
    "speed_change_rate",
    "stock_consistency_pct",
    "consistency_setpoint",
    "steam_pressure_bar",
    "steam_pressure_setpoint",
    "headbox_pressure_kpa",
    "headbox_pressure_setpoint",
    "slice_opening_mm",
    "slice_opening_setpoint",
    "stock_flow_lpm",
    "stock_flow_setpoint",
]

TARGET_COLUMN = "basis_weight_deviation_gsm"

# Columns the operator/model can actually adjust (used by the recommender)
CONTROLLABLE_FEATURES = [
    "machine_speed_mpm",
    "stock_consistency_pct",
    "steam_pressure_bar",
    "headbox_pressure_kpa",
    "slice_opening_mm",
    "stock_flow_lpm",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No historical data found at {path}. "
            f"Run: python src/data_generator.py first."
        )
    return pd.read_csv(path)


def train_model(df: pd.DataFrame = None, save: bool = True):
    if df is None:
        df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metadata = {
        "mae": mae,
        "r2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
    }

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(metadata, META_PATH)

    return model, metadata


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run: python train.py first."
        )
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH) if META_PATH.exists() else {}
    return model, metadata


def predict_deviation(model, feature_row: dict) -> float:
    """feature_row: dict with keys matching FEATURE_COLUMNS."""
    X = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    return float(model.predict(X)[0])


if __name__ == "__main__":
    m, meta = train_model()
    print("Model trained.")
    print(f"MAE: {meta['mae']:.3f} gsm | R2: {meta['r2']:.3f}")
