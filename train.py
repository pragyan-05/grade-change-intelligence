"""
train.py
--------
Run this ONCE after installing requirements, and again any time you want
to retrain (e.g. after collecting operator feedback / new historian data).

Usage:
    python train.py
"""

from src.data_generator import generate_dataset, main as generate_and_save
from src.model import train_model, DATA_PATH
from pathlib import Path


def run():
    if not DATA_PATH.exists():
        print("No historical dataset found - generating synthetic data...")
        generate_and_save()
    else:
        print(f"Using existing dataset at {DATA_PATH}")

    print("Training model...")
    model, metadata = train_model()
    print("Done.")
    print(f"  MAE : {metadata['mae']:.3f} gsm")
    print(f"  R^2 : {metadata['r2']:.3f}")
    print(f"  Trained on {metadata['n_train']} rows, tested on {metadata['n_test']} rows")
    print("\nModel saved to models/bw_deviation_model.pkl")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    run()
