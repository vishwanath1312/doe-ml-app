import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_curve, auc

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="DOE Full ML Dashboard", layout="wide")
st.title("🔬 DOE + ML Full Dashboard")
st.write("Forward/Backward Prediction, Model Performance (MSE & EER), and Optimal Formulation")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("doe.xlsx")
    df.columns = ["GMO", "Poloxamer", "ProbeTime", "ParticleSize", "Entrapment", "CDR"]
    return df

df = load_data()

# -----------------------------
# TRAIN MODELS
# -----------------------------
@st.cache_resource
def train_models():
    # Forward: Formulation → Responses
    X_fwd = df[["GMO", "Poloxamer", "ProbeTime"]]
    Y_fwd = df[["ParticleSize", "Entrapment", "CDR"]]
    fwd_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    fwd_model.fit(X_fwd, Y_fwd)

    # Backward: Responses → Formulation
    X_bwd = df[["ParticleSize", "Entrapment", "CDR"]]
    Y_bwd = df[["GMO", "Poloxamer", "ProbeTime"]]
    bwd_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    bwd_model.fit(X_bwd, Y_bwd)

    # Train/test split for performance evaluation (forward model)
    X_train, X_test, Y_train, Y_test = train_test_split(X_fwd, Y_fwd, test_size=0.2, random_state=42)

    return fwd_model, bwd_model, X_test, Y_test

fwd_model, bwd_model, X_test, Y_test = train_models()

# -----------------------------
# CREATE TABS
# -----------------------------
tabs = st.tabs(["Forward Prediction", "Backward Prediction", "Model Performance", "Optimization"])

# -----------------------------
# TAB 0: Forward Prediction
# -----------------------------
with tabs[0]:
    st.header("➡ Forward Prediction: Formulation → Responses")
    gmo = st.number_input("GMO (%)", float(df.GMO.min()), float(df.GMO.max()), float(df.GMO.mean()), key="fwd_gmo")
    poloxamer = st.number_input("Poloxamer 407 (%)", float(df.Poloxamer.min()), float(df.Poloxamer.max()), float(df.Poloxamer.mean()), key="fwd_polox")
    probe_time = st.number_input("Probe Time (min)", float(df.ProbeTime.min()), float(df.ProbeTime.max()), float(df.ProbeTime.mean()), key="fwd_probe")

    if st.button("Predict Responses", key="fwd_btn"):
        input_df = pd.DataFrame([[gmo, poloxamer, probe_time]], columns=["GMO", "Poloxamer", "ProbeTime"])
        pred = fwd_model.predict(input_df)

        st.write("**User Inputs (Formulation)**")
        st.table(input_df)

        st.write("**Computed Responses**")
        computed_df = pd.DataFrame({
            "Particle Size (nm)": [pred[0][0]],
            "Entrapment Efficiency (%)": [pred[0][1]],
            "CDR (%)": [pred[0][2]]
        })
        st.table(computed_df)

# -----------------------------
# TAB 1: Backward Prediction
# -----------------------------
with tabs[1]:
    st.header("🔄 Backward Prediction: Responses → Formulation")
    particle_size = st.number_input("Particle Size (nm)", float(df.ParticleSize.min()), float(df.ParticleSize.max()), float(df.ParticleSize.mean()), key="bwd_ps")
    entrapment = st.number_input("Entrapment Efficiency (%)", float(df.Entrapment.min()), float(df.Entrapment.max()), float(df.Entrapment.mean()), key="bwd_ent")
    cdr = st.number_input("CDR (%)", float(df.CDR.min()), float(df.CDR.max()), float(df.CDR.mean()), key="bwd_cdr")

    if st.button("Predict Formulation", key="bwd_btn"):
        input_df = pd.DataFrame([[particle_size, entrapment, cdr]], columns=["ParticleSize", "Entrapment", "CDR"])
        pred = bwd_model.predict(input_df)

        st.write("**User Inputs (Target Responses)**")
        st.table(input_df)

        st.write("**Computed Formulation**")
        computed_df = pd.DataFrame({
            "GMO (%)": [pred[0][0]],
            "Poloxamer 407 (%)": [pred[0][1]],
            "Probe Time (min)": [pred[0][2]]
        })
        st.table(computed_df)

# -----------------------------
# TAB 2: Model Performance Dashboard
# -----------------------------
with tabs[2]:
    st.header("📊 Model Performance Dashboard")

    preds_test = fwd_model.predict(X_test)

    col1, col2, col3 = st.columns([1, 1, 2])

    # ---- Column 1: MSE ----
    with col1:
        st.subheader("Mean Squared Error (MSE)")
        mse_dict = {col: mean_squared_error(Y_test[col], preds_test[:, idx])
                    for idx, col in enumerate(Y_test.columns)}
        mse_df = pd.DataFrame(mse_dict, index=["MSE"]).T
        st.table(mse_df)

    # ---- Column 2: ROC & EER ----
    with col2:
        st.subheader("ROC Curve & EER")
        response_choice = st.selectbox("Select Response for ROC/EER", ["ParticleSize", "Entrapment", "CDR"], key="roc_side")
        idx = Y_test.columns.get_loc(response_choice)

        y_true = (Y_test[response_choice] > Y_test[response_choice].median()).astype(int)
        y_score = preds_test[:, idx]

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        st.write(f"**{response_choice}:** AUC = {roc_auc:.2f}, EER = {eer:.2f}")

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve ({response_choice})")
        ax.legend()
        st.pyplot(fig)

    # ---- Column 3: Sample Forward Predictions ----
    with col3:
        st.subheader("Sample Predictions vs Actual")
        # Allow user to select a row for forward prediction
        row_idx = st.number_input("Select Test Sample Row Index", min_value=0, max_value=len(X_test)-1, value=0, step=1)
        sample_input = X_test.iloc[[row_idx]]
        sample_actual = Y_test.iloc[[row_idx]]
        sample_pred = fwd_model.predict(sample_input)

        st.write("**User Inputs (Formulation)**")
        st.table(sample_input)

        st.write("**Actual Responses**")
        st.table(sample_actual)

        st.write("**Predicted Responses**")
        sample_pred_df = pd.DataFrame({
            "Particle Size (nm)": [sample_pred[0][0]],
            "Entrapment Efficiency (%)": [sample_pred[0][1]],
            "CDR (%)": [sample_pred[0][2]]
        })
        st.table(sample_pred_df)

# -----------------------------
# TAB 3: Optimization
# -----------------------------
with tabs[3]:
    st.header("🎯 Optimal Formulation")
    GMO_grid = np.linspace(df.GMO.min(), df.GMO.max(), 10)
    Poloxamer_grid = np.linspace(df.Poloxamer.min(), df.Poloxamer.max(), 10)
    ProbeTime_grid = np.linspace(df.ProbeTime.min(), df.ProbeTime.max(), 10)

    candidates = pd.DataFrame([[g, p, t] for g in GMO_grid for p in Poloxamer_grid for t in ProbeTime_grid],
                              columns=["GMO", "Poloxamer", "ProbeTime"])
    preds_grid = fwd_model.predict(candidates)
    candidates["ParticleSize"] = preds_grid[:, 0]
    candidates["Entrapment"] = preds_grid[:, 1]
    candidates["CDR"] = preds_grid[:, 2]
    candidates["Score"] = -candidates["ParticleSize"] + candidates["Entrapment"] + candidates["CDR"]

    best = candidates.loc[candidates.Score.idxmax()]
    st.success("Optimal Formulation Found")
    st.table(best)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("DOE + Machine Learning Full Dashboard | Academic Project")
