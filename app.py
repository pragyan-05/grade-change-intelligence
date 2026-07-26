"""
app.py
------
AI-Powered Grade Change Intelligence Dashboard for Paper Manufacturing.

Run with:
    streamlit run app.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model import (
    FEATURE_COLUMNS,
    CONTROLLABLE_FEATURES,
    load_data,
    load_model,
    predict_deviation,
)
from src.explainability import (
    global_feature_importance,
    local_explanation,
    humanize_feature_name,
)
from src.recommender import recommend_setpoints

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BASE_DIR / "feedback" / "feedback.csv"
BW_LIMIT_GSM = 3.0  # +/- acceptable basis weight deviation before it's "out of spec"

st.set_page_config(
    page_title="Grade Change Intelligence | Paper Mill AI",
    page_icon="🧻",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.2rem; border-radius: 12px; color: white;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 10px 18px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return load_model()


def model_is_ready():
    from src.model import MODEL_PATH
    return MODEL_PATH.exists()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧻 AI-Powered Grade Change Intelligence")
st.caption(
    "Decision support for paper machine grade transitions — predicts Basis "
    "Weight deviations before they happen, explains why, and recommends "
    "setpoints to stabilize faster."
)

if not model_is_ready():
    st.error(
        "No trained model found. Open a terminal in this project folder and run:\n\n"
        "`python train.py`\n\nthen refresh this page."
    )
    st.stop()

df = get_data()
model, metadata = get_model()

# ----------------------------------------------------------------------------
# Sidebar: pick a transition to work with
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Select Grade Transition")
transition_ids = sorted(df["transition_id"].unique())
selected_transition = st.sidebar.selectbox(
    "Historical transition record", transition_ids,
    format_func=lambda x: f"Transition #{x}"
)

trans_df = df[df["transition_id"] == selected_transition].reset_index(drop=True)
from_grade = trans_df["from_grade"].iloc[0]
to_grade = trans_df["to_grade"].iloc[0]

st.sidebar.markdown(f"**From:** {from_grade}")
st.sidebar.markdown(f"**To:** {to_grade}")

minute = st.sidebar.slider(
    "Minutes since transition start", 0, int(trans_df["minutes_since_transition"].max()), 5
)

st.sidebar.divider()
st.sidebar.header("📊 Model Info")
st.sidebar.metric("Mean Abs. Error", f"{metadata.get('mae', 0):.2f} gsm")
st.sidebar.metric("R² Score", f"{metadata.get('r2', 0):.3f}")
st.sidebar.caption(
    "Trained on synthetic historical data reproducing typical grade-change "
    "dynamics. Swap in your mill's historian export (same column names) "
    "in data/historical_grade_change_data.csv and re-run train.py."
)

current_row = trans_df[trans_df["minutes_since_transition"] == minute]
if current_row.empty:
    current_row = trans_df.iloc[[0]]
current_row = current_row.iloc[0]
current_state = {f: current_row[f] for f in FEATURE_COLUMNS}

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_predict, tab_explain, tab_recommend, tab_feedback = st.tabs(
    ["📈 Overview", "🔮 Live Prediction", "🧠 Explainability", "🎯 Recommendations", "📝 Operator Feedback"]
)

# =============================== OVERVIEW ====================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    total_transitions = df["transition_id"].nunique()
    avg_dev = df["basis_weight_deviation_gsm"].abs().mean()
    out_of_spec_pct = (df["basis_weight_deviation_gsm"].abs() > BW_LIMIT_GSM).mean() * 100
    avg_stabilize = (
        df[df["basis_weight_deviation_gsm"].abs() <= BW_LIMIT_GSM]
        .groupby("transition_id")["minutes_since_transition"].min().mean()
    )

    c1.metric("Historical Transitions", total_transitions)
    c2.metric("Avg |BW Deviation|", f"{avg_dev:.2f} gsm")
    c3.metric("Time Out-of-Spec", f"{out_of_spec_pct:.1f}%")
    c4.metric("Avg. Stabilization Time", f"{avg_stabilize:.1f} min")

    st.divider()
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader(f"Basis Weight Deviation — Transition #{selected_transition}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trans_df["minutes_since_transition"], y=trans_df["basis_weight_deviation_gsm"],
            mode="lines+markers", name="BW Deviation", line=dict(color="#2c5282", width=3)
        ))
        fig.add_hline(y=BW_LIMIT_GSM, line_dash="dash", line_color="red", annotation_text="Upper limit")
        fig.add_hline(y=-BW_LIMIT_GSM, line_dash="dash", line_color="red", annotation_text="Lower limit")
        fig.add_vline(x=minute, line_dash="dot", line_color="orange", annotation_text="Selected time")
        fig.update_layout(xaxis_title="Minutes since transition start",
                           yaxis_title="Deviation (gsm)", height=420,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Grade Mix (historical)")
        grade_counts = df.drop_duplicates("transition_id")["to_grade"].value_counts()
        fig2 = px.pie(values=grade_counts.values, names=grade_counts.index, hole=0.5)
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Between Process Variables & BW Deviation")
    corr_cols = FEATURE_COLUMNS + ["basis_weight_deviation_gsm"]
    corr = df[corr_cols].corr()
    fig3 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    fig3.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "This heatmap reveals both **known** correlations (e.g. speed change "
        "rate vs deviation) and potentially **hidden** ones an operator might "
        "not track by eye — a key goal of this system."
    )

# ============================= LIVE PREDICTION ===============================
with tab_predict:
    st.subheader("Predict Basis Weight Deviation for a Custom Process State")
    st.caption("Adjust the sliders to reflect current/planned process conditions and get an instant prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        machine_speed = st.slider("Machine speed (m/min)", 400, 1200, int(current_state["machine_speed_mpm"]))
        speed_change_rate = st.slider("Speed change rate", 0, 30, int(current_state["speed_change_rate"]))
        stock_consistency = st.slider("Stock consistency (%)", 1.5, 5.0, float(current_state["stock_consistency_pct"]), 0.05)
        consistency_setpoint = st.slider("Consistency setpoint (%)", 1.5, 5.0, float(current_state["consistency_setpoint"]), 0.05)
    with col2:
        steam_pressure = st.slider("Steam pressure (bar)", 1.0, 7.0, float(current_state["steam_pressure_bar"]), 0.1)
        steam_setpoint = st.slider("Steam pressure setpoint (bar)", 1.0, 7.0, float(current_state["steam_pressure_setpoint"]), 0.1)
        headbox_pressure = st.slider("Headbox pressure (kPa)", 5, 60, int(current_state["headbox_pressure_kpa"]))
        headbox_setpoint = st.slider("Headbox pressure setpoint (kPa)", 5, 60, int(current_state["headbox_pressure_setpoint"]))
    with col3:
        slice_opening = st.slider("Slice opening (mm)", 5.0, 26.0, float(current_state["slice_opening_mm"]), 0.1)
        slice_setpoint = st.slider("Slice opening setpoint (mm)", 5.0, 26.0, float(current_state["slice_opening_setpoint"]), 0.1)
        stock_flow = st.slider("Stock flow (L/min)", 1500, 7000, int(current_state["stock_flow_lpm"]))
        stock_flow_setpoint = st.slider("Stock flow setpoint (L/min)", 1500, 7000, int(current_state["stock_flow_setpoint"]))

    minutes_input = st.slider("Minutes since transition start", 0, 90, int(minute))

    custom_state = {
        "minutes_since_transition": minutes_input,
        "machine_speed_mpm": machine_speed,
        "speed_change_rate": speed_change_rate,
        "stock_consistency_pct": stock_consistency,
        "consistency_setpoint": consistency_setpoint,
        "steam_pressure_bar": steam_pressure,
        "steam_pressure_setpoint": steam_setpoint,
        "headbox_pressure_kpa": headbox_pressure,
        "headbox_pressure_setpoint": headbox_setpoint,
        "slice_opening_mm": slice_opening,
        "slice_opening_setpoint": slice_setpoint,
        "stock_flow_lpm": stock_flow,
        "stock_flow_setpoint": stock_flow_setpoint,
    }

    predicted_dev = predict_deviation(model, custom_state)

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        risk_color = "🟢" if abs(predicted_dev) <= BW_LIMIT_GSM * 0.5 else (
            "🟠" if abs(predicted_dev) <= BW_LIMIT_GSM else "🔴")
        st.metric("Predicted BW Deviation", f"{predicted_dev:+.2f} gsm", delta=None)
        st.markdown(f"### {risk_color} Risk Level: "
                     f"{'Low' if abs(predicted_dev) <= BW_LIMIT_GSM*0.5 else ('Moderate' if abs(predicted_dev) <= BW_LIMIT_GSM else 'HIGH — action recommended')}")
        if abs(predicted_dev) > BW_LIMIT_GSM:
            st.warning(
                f"Predicted deviation exceeds the ±{BW_LIMIT_GSM} gsm spec limit. "
                "Check the **Recommendations** tab for suggested setpoint changes."
            )
        st.session_state["custom_state"] = custom_state
        st.session_state["predicted_dev"] = predicted_dev

    with r2:
        # forecast over the next 45 minutes assuming the model's learned stabilization trend
        forecast_minutes = list(range(minutes_input, minutes_input + 45))
        forecast_devs = []
        for m in forecast_minutes:
            s = dict(custom_state)
            s["minutes_since_transition"] = m
            forecast_devs.append(predict_deviation(model, s))

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=forecast_minutes, y=forecast_devs, mode="lines",
                                   line=dict(color="#2c5282", width=3), name="Forecast"))
        fig4.add_hline(y=BW_LIMIT_GSM, line_dash="dash", line_color="red")
        fig4.add_hline(y=-BW_LIMIT_GSM, line_dash="dash", line_color="red")
        fig4.update_layout(title="Forecasted Stabilization Trend (next 45 min at current setpoints)",
                            xaxis_title="Minutes since transition start", yaxis_title="Predicted deviation (gsm)",
                            height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)

# ============================= EXPLAINABILITY =================================
with tab_explain:
    st.subheader("Why does the model predict this?")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 🌍 Global Feature Importance")
        st.caption("Which variables matter most across ALL historical grade changes.")
        gi = global_feature_importance(model, FEATURE_COLUMNS)
        gi["feature_label"] = gi["feature"].apply(humanize_feature_name)
        fig5 = px.bar(gi, x="importance", y="feature_label", orientation="h",
                      color="importance", color_continuous_scale="Blues")
        fig5.update_layout(height=480, yaxis=dict(autorange="reversed"),
                            margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with right:
        st.markdown("#### 🔍 Local Explanation (this prediction)")
        state_to_explain = st.session_state.get("custom_state", current_state)
        pred_to_explain = st.session_state.get("predicted_dev", predict_deviation(model, state_to_explain))
        st.caption(f"Breaking down the **{pred_to_explain:+.2f} gsm** prediction from the Live Prediction tab.")

        local_df = local_explanation(model, state_to_explain, FEATURE_COLUMNS, df)
        local_df["feature_label"] = local_df["feature"].apply(humanize_feature_name)
        local_df["direction"] = np.where(local_df["contribution_gsm"] >= 0, "Increases deviation", "Decreases deviation")

        fig6 = px.bar(local_df.head(10), x="contribution_gsm", y="feature_label", orientation="h",
                      color="direction", color_discrete_map={
                          "Increases deviation": "#e53e3e", "Decreases deviation": "#2f855a"
                      })
        fig6.update_layout(height=480, yaxis=dict(autorange="reversed"),
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Contribution to prediction (gsm)")
        st.plotly_chart(fig6, use_container_width=True)

    st.info(
        "💡 **How to read this:** Global importance tells you which knobs matter "
        "in general. The local explanation tells you, for THIS specific process "
        "snapshot, which variables are pushing the prediction up or down and by "
        "how much — this is what makes the recommendation explainable rather "
        "than a black box."
    )

# ============================== RECOMMENDATIONS ================================
with tab_recommend:
    st.subheader("🎯 Recommended Setpoint Adjustments")
    state_for_rec = st.session_state.get("custom_state", current_state)

    st.caption(
        "The recommender searches for small, safe adjustments to controllable "
        "variables (speed, consistency, steam pressure, headbox pressure, slice "
        "opening, stock flow) that minimize the model's predicted |deviation|, "
        "using the trained model as a fast stand-in for the real process."
    )

    if st.button("🔎 Generate Recommendation", type="primary"):
        with st.spinner("Searching for optimal setpoints..."):
            rec_state, changes_df, history_df, pred_before, pred_after = recommend_setpoints(
                model, state_for_rec
            )
        st.session_state["rec_state"] = rec_state
        st.session_state["changes_df"] = changes_df
        st.session_state["pred_before"] = pred_before
        st.session_state["pred_after"] = pred_after
        st.session_state["history_df"] = history_df

    if "changes_df" in st.session_state:
        pred_before = st.session_state["pred_before"]
        pred_after = st.session_state["pred_after"]
        changes_df = st.session_state["changes_df"]
        history_df = st.session_state["history_df"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted deviation BEFORE", f"{pred_before:+.2f} gsm")
        c2.metric("Predicted deviation AFTER", f"{pred_after:+.2f} gsm",
                   delta=f"{pred_after - pred_before:+.2f} gsm")
        improvement_pct = (1 - abs(pred_after) / max(abs(pred_before), 1e-6)) * 100
        c3.metric("Improvement", f"{improvement_pct:.0f}%")

        st.divider()
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("#### Suggested Changes")
            if changes_df.empty:
                st.success("Current setpoints are already near-optimal — no changes needed.")
            else:
                display_df = changes_df.copy()
                display_df["variable"] = display_df["variable"].apply(humanize_feature_name)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### Optimization Path")
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(x=history_df["round"], y=history_df["predicted_deviation"],
                                       mode="lines+markers", line=dict(color="#2c5282")))
            fig7.add_hline(y=BW_LIMIT_GSM, line_dash="dash", line_color="red")
            fig7.add_hline(y=-BW_LIMIT_GSM, line_dash="dash", line_color="red")
            fig7.update_layout(xaxis_title="Optimization round", yaxis_title="Predicted deviation (gsm)",
                                height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig7, use_container_width=True)

        st.info(
            "✅ These recommendations are generated from the same model shown in "
            "the Explainability tab, so every suggested change can be traced back "
            "to a feature contribution — nothing here is a black-box output."
        )
    else:
        st.write("Click **Generate Recommendation** to see suggested setpoint changes.")

# ================================ FEEDBACK =====================================
with tab_feedback:
    st.subheader("📝 Operator Feedback — Continuous Learning Loop")
    st.caption(
        "Feedback captured here is stored and can be used to retrain / recalibrate "
        "the model over time, closing the loop between AI recommendations and "
        "real operator judgment."
    )

    with st.form("feedback_form", clear_on_submit=True):
        f1, f2 = st.columns(2)
        with f1:
            operator_name = st.text_input("Operator name / ID")
            transition_ref = st.text_input("Transition reference", value=f"Transition #{selected_transition}")
            usefulness = st.select_slider(
                "Was the recommendation useful?",
                options=["Not useful", "Somewhat useful", "Useful", "Very useful"],
                value="Useful",
            )
        with f2:
            was_applied = st.radio("Was the recommendation applied?", ["Yes", "No", "Partially"], horizontal=True)
            actual_deviation = st.number_input("Actual BW deviation observed (gsm)", value=0.0, step=0.1)
            comments = st.text_area("Comments / observations")

        submitted = st.form_submit_button("Submit Feedback", type="primary")

        if submitted:
            FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            new_row = pd.DataFrame([{
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "operator_name": operator_name,
                "transition_reference": transition_ref,
                "predicted_deviation": st.session_state.get("predicted_dev", np.nan),
                "actual_deviation": actual_deviation,
                "usefulness": usefulness,
                "was_applied": was_applied,
                "comments": comments,
            }])
            if FEEDBACK_PATH.exists():
                new_row.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
            else:
                new_row.to_csv(FEEDBACK_PATH, index=False)
            st.success("Feedback saved. Thank you!")

    st.divider()
    st.markdown("#### Feedback History")
    if FEEDBACK_PATH.exists():
        fb_df = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(fb_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Usefulness distribution**")
            fig8 = px.histogram(fb_df, x="usefulness",
                                 category_orders={"usefulness": ["Not useful", "Somewhat useful", "Useful", "Very useful"]})
            fig8.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig8, use_container_width=True)
        with c2:
            st.markdown("**Predicted vs. Actual Deviation**")
            if fb_df["actual_deviation"].notna().any():
                fig9 = px.scatter(fb_df, x="predicted_deviation", y="actual_deviation",
                                   trendline=None)
                fig9.add_shape(type="line", x0=-10, y0=-10, x1=10, y1=10,
                                line=dict(color="gray", dash="dash"))
                fig9.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig9, use_container_width=True)
    else:
        st.write("No feedback submitted yet.")

    st.divider()
    st.markdown("#### 🔁 Retrain Model")
    st.caption(
        "In a production deployment this would merge new historian + feedback data "
        "and retrain automatically on a schedule. For this demo, click below to "
        "retrain on the current synthetic dataset (simulates the continuous "
        "learning loop)."
    )
    if st.button("Retrain Model Now"):
        from src.model import train_model
        with st.spinner("Retraining..."):
            _, new_meta = train_model()
        st.cache_resource.clear()
        st.success(f"Retrained. New MAE: {new_meta['mae']:.3f} gsm, R²: {new_meta['r2']:.3f}")
        st.rerun()

st.divider()
st.caption(
    "AI-Powered Grade Change Intelligence · Built with Streamlit, scikit-learn, "
    "and Plotly · This is a decision-support prototype — always verify against "
    "plant safety systems before acting on recommendations."
)
