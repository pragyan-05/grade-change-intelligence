# 🧻 AI-Powered Grade Change Intelligence for Paper Manufacturing

An AI-driven decision-support system that predicts and prevents **Basis
Weight (BW) quality deviations** during paper grade transitions. It analyzes
historical process data to surface known and hidden correlations, predicts
BW deviations before they exceed spec, explains *why* in plain language, and
recommends setpoint changes to stabilize the process faster — all through an
interactive Streamlit dashboard, with operator feedback captured for
continuous learning.

Built for beginners: every file is commented, and on Windows the two
included batch files (`setup.bat` then `start.bat`) are all you need to see
it running.

---

## 1. What's inside

| Capability (from the brief)              | Where it lives |
|-------------------------------------------|----------------|
| Historical data analysis / correlations   | `Overview` tab — correlation heatmap, trend charts |
| Predict BW deviations before limits break | `Live Prediction` tab — RandomForest model + forecast |
| Recommend setpoints, reduce stabilization | `Recommendations` tab — greedy surrogate optimizer |
| Explainable recommendations               | `Explainability` tab — global + local (SHAP) attributions |
| Interactive dashboard                     | Full Streamlit app (`app.py`) |
| Operator feedback / continuous learning   | `Operator Feedback` tab — CSV log + retrain button |

---

## 2. Project structure

```
paper-grade-intelligence/
├── app.py                     # Streamlit dashboard (run this)
├── train.py                   # One-command setup: generate data + train model
├── setup.bat                  # Windows: double-click to install + train
├── start.bat                  # Windows: double-click to launch the dashboard
├── requirements.txt
├── .streamlit/config.toml     # Dashboard theme
├── data/
│   └── historical_grade_change_data.csv   # created by train.py
├── models/
│   ├── bw_deviation_model.pkl              # created by train.py
│   └── model_metadata.pkl
├── feedback/
│   └── feedback.csv                        # created when an operator submits feedback
└── src/
    ├── data_generator.py      # synthetic historical data (physics-inspired)
    ├── model.py                # RandomForest training / loading / predicting
    ├── explainability.py       # global importance + SHAP local explanations
    └── recommender.py          # greedy setpoint optimizer
```

> **Using real mill data?** Replace `data/historical_grade_change_data.csv`
> with your historian export using the same column names (see
> `src/model.py: FEATURE_COLUMNS`), then run `python train.py` again. Nothing
> else in the app needs to change.

---

## 3. Setup on Windows (step-by-step)

### Step 0 — Prerequisites
1. Install **Python 3.10–3.12** from https://python.org/downloads — during
   install, tick the box that says **"Add python.exe to PATH"** (easy to
   miss, and everything below depends on it).
2. Install **VS Code** from https://code.visualstudio.com if you don't have
   it, and its **Python extension** (Extensions panel → search "Python").
3. Open PowerShell (or the VS Code terminal — see Step 1) and confirm:
   ```powershell
   python --version
   ```
   You should see `Python 3.10.x` or similar. If you get an error, Python
   isn't on PATH — reinstall and tick that checkbox.

### Step 1 — Unzip and open in VS Code
Right-click `paper-grade-intelligence.zip` in File Explorer → **Extract
All...**, then open the extracted folder in VS Code:
```powershell
cd Downloads\paper-grade-intelligence
code .
```
From here on, use VS Code's integrated terminal (Terminal → New Terminal —
it opens PowerShell by default).

### Step 2 — Fastest path: the one-click scripts
Double-click **`setup.bat`** in File Explorer (or run `.\setup.bat` in the
terminal). It creates a virtual environment, installs everything in
`requirements.txt`, and trains the model — you'll see progress printed as it
goes. When it's done, double-click **`start.bat`** (or run `.\start.bat`)
any time you want to launch the dashboard. That's the whole setup — you can
skip straight to **Step 6** below to verify it worked.

### Step 3 — Or, the manual/step-by-step way (if you'd rather see each command)
Create a virtual environment:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
You should see `(venv)` appear at the start of your prompt. If PowerShell
blocks this with a red *"running scripts is disabled on this system"*
error, run this once (it only relaxes the policy for your user account),
then retry the activate command:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Prefer Command Prompt instead of PowerShell? Use `venv\Scripts\activate.bat`
— it doesn't have this restriction at all.

(In VS Code, also select this environment as the interpreter: Ctrl+Shift+P →
"Python: Select Interpreter" → choose the one inside `.\venv`.)

### Step 4 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 5 — Generate historical data + train the model
```powershell
python train.py
```
You should see something like:
```
MAE : 0.37 gsm
R^2 : 0.62
Model saved to models/bw_deviation_model.pkl
```

### Step 6 — Verify it worked: launch the dashboard
```powershell
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`. If it
doesn't, open that URL manually.

That's it — you now have a working AI decision-support dashboard.

### Every time you come back to work on it
Easiest: double-click **`start.bat`**.

Or manually, from the project folder:
```powershell
venv\Scripts\Activate.ps1
streamlit run app.py
```

<details>
<summary><strong>macOS / Linux instructions</strong> (click to expand)</summary>

```bash
cd ~/Downloads/paper-grade-intelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```
Next time: `cd ~/Downloads/paper-grade-intelligence && source venv/bin/activate && streamlit run app.py`
</details>

---

## 4. How to use the dashboard

1. **Overview** — pick a historical grade transition in the sidebar, see its
   BW deviation trend against spec limits, and explore the correlation
   heatmap across all transitions.
2. **Live Prediction** — move the sliders to describe a current/planned
   process state and get an instant deviation prediction, risk level, and a
   45-minute forecast.
3. **Explainability** — see which variables matter most overall (global) and
   which ones are driving *this specific* prediction (local, SHAP-based).
4. **Recommendations** — click "Generate Recommendation" to get concrete
   setpoint changes that reduce the predicted deviation, with before/after
   numbers.
5. **Operator Feedback** — rate how useful a recommendation was, log what
   actually happened, and (optionally) click "Retrain Model Now" to simulate
   the continuous-learning loop.

---

## 5. Push this project to your GitHub profile

If `git --version` doesn't work in your terminal, install Git for Windows
from https://git-scm.com/download/win first (defaults are fine — it adds
Git to PATH and gives you Git Bash as a bonus terminal).

Run these from inside the `paper-grade-intelligence` folder in VS Code's
terminal (make sure `venv/` is excluded — it already is, via `.gitignore`):

```powershell
git init
git add .
git commit -m "Initial commit: AI-powered grade change intelligence dashboard"
```

Then create a new **empty** repository on GitHub (no README/license, so it
doesn't conflict): go to https://github.com/new, name it e.g.
`paper-grade-intelligence`, and click **Create repository**. GitHub will
show you a remote URL — use it below:

```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/paper-grade-intelligence.git
git push -u origin main
```

If prompted for credentials, use a GitHub **Personal Access Token** (not your
password) — generate one at GitHub → Settings → Developer settings →
Personal access tokens.

> Tip: the `.gitignore` currently excludes the generated `data/*.csv` and
> `models/*.pkl` files to keep the repo light and reproducible (anyone who
> clones it just runs `train.py`). If you'd rather commit the trained
> artifacts too (so the app works immediately after cloning, no training
> step), open `.gitignore` and remove/comment those two lines before your
> first commit.

---

## 6. Architecture

### 6.1 Diagram (Mermaid — renders automatically on GitHub)

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        A1[("Mill Historian / DCS\n(or synthetic generator)")]
    end

    subgraph Pipeline["Data & ML Pipeline (src/)"]
        B1["data_generator.py\nHistorical grade-change data"]
        B2["model.py\nRandomForest Regressor\n(train / predict)"]
        B3["explainability.py\nGlobal importance + SHAP\nlocal explanations"]
        B4["recommender.py\nGreedy surrogate\nsetpoint optimizer"]
    end

    subgraph Storage["Persisted Artifacts"]
        C1[("data/historical_grade_change_data.csv")]
        C2[("models/bw_deviation_model.pkl")]
        C3[("feedback/feedback.csv")]
    end

    subgraph App["Streamlit Dashboard (app.py)"]
        D1["Overview tab\nTrends & correlations"]
        D2["Live Prediction tab\nDeviation forecast"]
        D3["Explainability tab\nWhy this prediction?"]
        D4["Recommendations tab\nSuggested setpoints"]
        D5["Operator Feedback tab\nRate & log outcomes"]
    end

    E1(["Operator / Process Engineer"])

    A1 --> B1 --> C1
    C1 --> B2 --> C2
    C2 --> B3
    C2 --> B4
    C1 --> D1
    C2 --> D2
    B3 --> D3
    B4 --> D4
    D1 & D2 & D3 & D4 --> E1
    E1 --> D5 --> C3
    C3 -.retrain loop.-> B2
```

### 6.2 Prompt for Napkin AI (if you'd like a stylized illustration)

Paste this into [napkin.ai](https://www.napkin.ai) to auto-generate a
polished architecture diagram:

```
Create a system architecture diagram titled "AI-Powered Grade Change
Intelligence for Paper Manufacturing" with these components and flow,
left to right:

1. Data Source: "Mill Historian / DCS Data" feeding into
2. Data Layer: "Historical Process Data (CSV)" which feeds into
3. ML Layer with three boxes side by side: "Prediction Model
   (Random Forest)", "Explainability Engine (SHAP)", and
   "Setpoint Recommender (Optimizer)" — all three connect to
4. Application Layer: "Interactive Streamlit Dashboard" containing four
   tabs shown as small sub-nodes: "Trend & Correlation View",
   "Deviation Prediction & Risk Alerts", "Explainable Recommendations",
   "Operator Feedback Capture"
5. End user: "Process Engineer / Machine Operator" receiving the
   dashboard output and sending feedback
6. A feedback loop arrow going from "Operator Feedback Capture" back to
   "Prediction Model (Random Forest)" labeled "Continuous Learning /
   Retraining"

Use an industrial, clean tech style with blue and teal tones, rounded
rectangles, and clear directional arrows. Group items 2-3 in a labeled
container called "AI / ML Pipeline" and group item 4 in a container called
"Decision Support Dashboard".
```

---

## 7. How the AI actually works (plain-language summary)

- **Prediction**: A Random Forest model learns, from thousands of historical
  minute-by-minute grade-change records, how variables like speed ramp rate,
  stock consistency error, steam pressure lag, headbox pressure, slice
  opening, and stock flow relate to Basis Weight deviation. Given a new
  process snapshot, it predicts the expected deviation in grams per square
  metre (gsm).
- **Explainability**: *Global* importance (from the forest's structure)
  shows which variables matter most in general. *Local* explanation (SHAP,
  with a dependency-free fallback if SHAP isn't installed) shows, for one
  specific prediction, exactly how much each variable pushed the number up
  or down — this is what makes recommendations trustworthy instead of a
  black box.
- **Recommendation**: A greedy search nudges controllable setpoints (speed,
  consistency, steam pressure, headbox pressure, slice opening, stock flow)
  within safe operating ranges, using the trained model as a fast stand-in
  ("surrogate") for the real process, to find changes that reduce the
  predicted deviation.
- **Continuous learning**: Operator feedback (was the recommendation useful?
  what actually happened?) is logged to `feedback/feedback.csv`. In a real
  deployment this would be merged into the training set on a schedule; the
  "Retrain Model Now" button demonstrates that loop.

---

## 8. Troubleshooting

- **`'python' is not recognized as an internal or external command`** →
  Python isn't on PATH. Reinstall from python.org and tick **"Add python.exe
  to PATH"**, or search "Edit the system environment variables" → Environment
  Variables → add your Python install folder + its `Scripts` subfolder to PATH.
- **PowerShell: `running scripts is disabled on this system`** when
  activating the venv → run
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
  once, or just use `venv\Scripts\activate.bat` in Command Prompt instead.
- **`ModuleNotFoundError`** → you forgot to activate the venv
  (`venv\Scripts\Activate.ps1`) or run `pip install -r requirements.txt`.
- **"No trained model found"** in the app → run `python train.py` first (or
  double-click `setup.bat`).
- **`shap` fails to install/build** → this is common on Windows since it can
  need a C++ compiler. The app still works fine without it — it
  automatically falls back to a built-in explanation method (see
  `src/explainability.py`). Just delete the `shap` line from
  `requirements.txt` and reinstall, or ignore the pip error and continue.
- **Port already in use** → `streamlit run app.py --server.port 8502`.
- **Antivirus/Windows Defender flags `venv\Scripts\activate.ps1`** → this is
  a common false positive for freshly-created venv scripts; you can safely
  allow it, or use `activate.bat` in Command Prompt instead.
- **Double-clicking `setup.bat` closes instantly with no visible error** →
  open PowerShell, `cd` into the folder, and run `.\setup.bat` instead so
  the window stays open and you can read any error message.

---

## 9. Extending this project

- Swap the synthetic CSV for a real historian export.
- Replace RandomForest with XGBoost/LightGBM in `src/model.py` (same
  `predict_deviation()` interface, nothing else changes).
- Add more quality variables beyond Basis Weight (moisture, caliper).
- Schedule automatic retraining (e.g. a cron job calling `train.py`).
- Deploy with [Streamlit Community Cloud](https://streamlit.io/cloud) by
  connecting your GitHub repo directly — free hosting for this kind of app.
