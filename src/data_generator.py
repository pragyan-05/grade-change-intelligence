"""
data_generator.py
------------------
Generates SYNTHETIC historical process data that mimics what a real paper
machine's DCS/PI historian would log during grade changes.

Why synthetic data?
This project ships without access to a real mill's historian, so we
simulate physically-plausible relationships between process variables and
Basis Weight (BW) deviation. The relationships (speed-change rate, stock
consistency error, steam pressure lag, headbox pressure, slice opening,
time-since-transition) are based on well-known paper-making cause/effect
patterns described in process-control literature. Replace this generator
with a loader for your real historian export (CSV/SQL) when available -
the rest of the pipeline (model.py, app.py) does not need to change as
long as the column names match.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
GRADES = ["Grade_A_Light", "Grade_B_Medium", "Grade_C_Heavy", "Grade_D_Specialty"]

# Target basis weight (gsm) per grade
GRADE_TARGET_BW = {
    "Grade_A_Light": 45,
    "Grade_B_Medium": 70,
    "Grade_C_Heavy": 120,
    "Grade_D_Specialty": 90,
}


def _simulate_transition(rng, transition_id, from_grade, to_grade, n_minutes=60):
    """Simulate one grade-change event, minute by minute, for n_minutes."""
    target_bw = GRADE_TARGET_BW[to_grade]
    prev_bw = GRADE_TARGET_BW[from_grade]

    # Randomize the operational aggressiveness of this particular changeover
    speed_change_rate = rng.uniform(2, 25)       # m/min per minute ramp rate
    consistency_setpoint = rng.uniform(2.8, 4.2)  # % stock consistency
    steam_pressure_setpoint = rng.uniform(2.0, 5.5)  # bar
    headbox_pressure_setpoint = rng.uniform(15, 45)  # kPa
    slice_opening_setpoint = rng.uniform(8, 22)      # mm
    stock_flow_setpoint = rng.uniform(2500, 6000)    # L/min

    rows = []
    machine_speed = rng.uniform(600, 1000)
    consistency = consistency_setpoint + rng.normal(0, 0.15)
    steam_pressure = steam_pressure_setpoint + rng.normal(0, 0.2)
    headbox_pressure = headbox_pressure_setpoint + rng.normal(0, 1.5)
    slice_opening = slice_opening_setpoint + rng.normal(0, 0.3)
    stock_flow = stock_flow_setpoint + rng.normal(0, 80)

    for minute in range(n_minutes):
        t = minute  # minutes since transition start

        # process variables drift toward their setpoints with noise + lag
        machine_speed += rng.normal(speed_change_rate * 0.15, 1.2)
        consistency += (consistency_setpoint - consistency) * 0.12 + rng.normal(0, 0.05)
        steam_pressure += (steam_pressure_setpoint - steam_pressure) * 0.08 + rng.normal(0, 0.06)
        headbox_pressure += (headbox_pressure_setpoint - headbox_pressure) * 0.10 + rng.normal(0, 0.4)
        slice_opening += (slice_opening_setpoint - slice_opening) * 0.10 + rng.normal(0, 0.08)
        stock_flow += (stock_flow_setpoint - stock_flow) * 0.10 + rng.normal(0, 25)

        # --- physically-inspired ground truth relationship for BW deviation ---
        # Faster speed ramps + consistency error + pressure lag => bigger deviation
        consistency_error = consistency - consistency_setpoint
        pressure_lag = (steam_pressure_setpoint - steam_pressure)
        stability_factor = np.exp(-t / 18.0)  # deviation shrinks as process stabilizes

        deviation = (
            0.85 * speed_change_rate * stability_factor / 10.0
            + 4.5 * consistency_error
            - 1.6 * pressure_lag
            + 0.035 * (headbox_pressure - headbox_pressure_setpoint)
            + 0.09 * (slice_opening - slice_opening_setpoint)
            + 0.0022 * (stock_flow - stock_flow_setpoint)
            + rng.normal(0, 0.35)
        )

        basis_weight_actual = target_bw + deviation
        basis_weight_target = target_bw

        rows.append({
            "transition_id": transition_id,
            "from_grade": from_grade,
            "to_grade": to_grade,
            "minutes_since_transition": t,
            "machine_speed_mpm": round(machine_speed, 2),
            "speed_change_rate": round(speed_change_rate, 2),
            "stock_consistency_pct": round(consistency, 3),
            "consistency_setpoint": round(consistency_setpoint, 3),
            "steam_pressure_bar": round(steam_pressure, 3),
            "steam_pressure_setpoint": round(steam_pressure_setpoint, 3),
            "headbox_pressure_kpa": round(headbox_pressure, 2),
            "headbox_pressure_setpoint": round(headbox_pressure_setpoint, 2),
            "slice_opening_mm": round(slice_opening, 2),
            "slice_opening_setpoint": round(slice_opening_setpoint, 2),
            "stock_flow_lpm": round(stock_flow, 1),
            "stock_flow_setpoint": round(stock_flow_setpoint, 1),
            "basis_weight_target_gsm": basis_weight_target,
            "basis_weight_actual_gsm": round(basis_weight_actual, 2),
            "basis_weight_deviation_gsm": round(deviation, 3),
        })

    return rows


def generate_dataset(n_transitions=180, n_minutes=60, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    all_rows = []
    for i in range(n_transitions):
        from_grade, to_grade = rng.choice(GRADES, size=2, replace=False)
        rows = _simulate_transition(rng, transition_id=i + 1,
                                     from_grade=from_grade, to_grade=to_grade,
                                     n_minutes=n_minutes)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    return df


def main():
    out_path = Path(__file__).resolve().parent.parent / "data" / "historical_grade_change_data.csv"
    df = generate_dataset()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows across {df['transition_id'].nunique()} grade transitions")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
