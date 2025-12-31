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
st.set_page_config(page_title="DOE ML Dashboard", layout="wide")
st.title("🔬 DOE + ML Based Formulation Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("doe.xlsx")
    df.columns = [
        "GMO", "Poloxamer", "ProbeTime",
        "ParticleSize", "Entrapment", "CDR"
    ]
    return df

df = load_data()

X = df[["GMO", "Poloxamer", "ProbeTime"]]
Y = df[["ParticleSize", "Entrapment", "CDR"]]

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model():
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    model.fit(X_train, Y_train)

    return model, X_test, Y_test

model, X_test, Y_test = train_model()

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs(["Forward Prediction: Formulation", "Optimization"])

# =====================================================
# TAB 1: FORWARD PREDICTION + MODEL PERFORMANCE BELOW
# =====================================================
with tabs[0]:

    st.header("➡ Forward Prediction: Formulation → Responses")

    col1, col2, col3 = st.columns(3)

    with col1:
        gmo = st.number_input(
            "GMO (%)",
            float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean())
        )
    with col2:
        poloxamer = st.number_input(
            "Poloxamer 407 (%)",
            float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean())
        )
    with col3:
        probe_time = st.number_input(
            "Probe Time (min)",
            float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean())
        )

    if st.button("🔍 Predict Responses"):

        user_df = pd.DataFrame(
            [[gmo, poloxamer, probe_time]],
            columns=["GMO", "Poloxamer", "ProbeTime"]
        )

        preds = model.predict(user_df)

        st.subheader("📥 User Inputs")
        st.table(user_df)

        st.subheader("📤 Computed Outputs")
        output_df = pd.DataFrame({
            "Particle Size (nm)": [preds[0][0]],
            "Entrapment Efficiency (%)": [preds[0][1]],
            "CDR (%)": [preds[0][2]]
        })
        st.table(output_df)

        # =============================================
        # MODEL PERFORMANCE (BELOW FORWARD PREDICTION)
        # =============================================
        st.markdown("---")
        st.subheader("📊 Model Performance (Forward Model)")

        preds_test = model.predict(X_test)

        mp_col1, mp_col2 = st.columns([1, 2])

        # -------- MSE --------
        with mp_col1:
            st.write("### Mean Squared Error (MSE)")

            mse_vals = {
                col: mean_squared_error(Y_test[col], preds_test[:, idx])
                for idx, col in enumerate(Y_test.columns)
            }

            mse_df = pd.DataFrame(mse_vals, index=["MSE"]).T
            st.table(mse_df)

        # -------- ROC + EER --------
        with mp_col2:
            st.write("### ROC Curve & Equal Error Rate (EER)")

            response = st.selectbox(
                "Select Response Variable",
                ["ParticleSize", "Entrapment", "CDR"]
            )

            idx = Y_test.columns.get_loc(response)

            # Robust binary conversion
            y_true = (Y_test[response] >= Y_test[response].median()).astype(int)
            y_score = preds_test[:, idx]

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            fnr = 1 - tpr
            eer_index = np.argmin(np.abs(fpr - fnr))
            eer = (fpr[eer_index] + fnr[eer_index]) / 2

            st.write(f"**AUC:** {roc_auc:.3f}")
            st.write(f"**EER:** {eer:.3f}")

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.scatter(fpr[eer_index], tpr[eer_index], color="red", label="EER Point")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve ({response})")
            ax.legend()
            st.pyplot(fig)

# =====================================================
# TAB 2: OPTIMIZATION
# =====================================================
with tabs[1]:
    st.header("🎯 Optimal Formulation")

    GMO = np.linspace(X.GMO.min(), X.GMO.max(), 10)
    Poloxamer = np.linspace(X.Poloxamer.min(), X.Poloxamer.max(), 10)
    ProbeTime = np.linspace(X.ProbeTime.min(), X.ProbeTime.max(), 10)

    grid = pd.DataFrame(
        [[g, p, t] for g in GMO for p in Poloxamer for t in ProbeTime],
        columns=["GMO", "Poloxamer", "ProbeTime"]
    )

    preds = model.predict(grid)

    grid["ParticleSize"] = preds[:, 0]
    grid["Entrapment"] = preds[:, 1]
    grid["CDR"] = preds[:, 2]

    grid["Score"] = (
        -grid["ParticleSize"] +
        grid["Entrapment"] +
        grid["CDR"]
    )

    best = grid.loc[grid.Score.idxmax()]

    st.success("Optimal Formulation Found")
    st.table(best)
