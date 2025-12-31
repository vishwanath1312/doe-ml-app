import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_curve, auc

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="DOE + ML Formulation Dashboard",
    layout="wide"
)

st.title("🔬 DOE + Machine Learning Formulation Dashboard")
st.caption("Forward & Backward Prediction with MSE, ROC and EER Evaluation")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("doe.xlsx")
    df.columns = [
        "GMO", "Poloxamer", "ProbeTime",
        "ParticleSize", "Entrapment", "CDR"
    ]
    return df

df = load_data()

# =====================================================
# SPLIT DATA
# =====================================================
X_fwd = df[["GMO", "Poloxamer", "ProbeTime"]]
Y_fwd = df[["ParticleSize", "Entrapment", "CDR"]]

X_bwd = Y_fwd
Y_bwd = X_fwd

# =====================================================
# TRAIN MODELS
# =====================================================
@st.cache_resource
def train_models():
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_fwd, Y_fwd, test_size=0.2, random_state=42
    )

    fwd_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    fwd_model.fit(X_train, Y_train)

    bwd_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    bwd_model.fit(X_bwd, Y_bwd)

    return fwd_model, bwd_model, X_test, Y_test

fwd_model, bwd_model, X_test, Y_test = train_models()

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "➡ Forward Prediction (Formulation → Responses)",
    "🔄 Backward Prediction (Responses → Formulation)"
])

# =====================================================
# TAB 1: FORWARD PREDICTION
# =====================================================
with tabs[0]:

    st.header("➡ Forward Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        gmo = st.number_input("GMO (%)", float(df.GMO.min()), float(df.GMO.max()), float(df.GMO.mean()))
    with c2:
        poloxamer = st.number_input("Poloxamer 407 (%)", float(df.Poloxamer.min()), float(df.Poloxamer.max()), float(df.Poloxamer.mean()))
    with c3:
        probe = st.number_input("Probe Time (min)", float(df.ProbeTime.min()), float(df.ProbeTime.max()), float(df.ProbeTime.mean()))

    if st.button("🔍 Predict Responses"):

        user_input = pd.DataFrame([[gmo, poloxamer, probe]],
                                  columns=["GMO", "Poloxamer", "ProbeTime"])
        preds = fwd_model.predict(user_input)

        st.subheader("📥 User Inputs")
        st.table(user_input)

        st.subheader("📤 Computed Outputs")
        output_df = pd.DataFrame({
            "Particle Size (nm)": [preds[0][0]],
            "Entrapment Efficiency (%)": [preds[0][1]],
            "CDR (%)": [preds[0][2]]
        })
        st.table(output_df)

        # ================= MODEL PERFORMANCE =================
        st.markdown("---")
        st.subheader("📊 Forward Model Performance")

        preds_test = fwd_model.predict(X_test)

        mp1, mp2 = st.columns([1, 2])

        # ----- MSE -----
        with mp1:
            st.write("### Mean Squared Error (MSE)")
            mse_vals = {
                col: mean_squared_error(Y_test[col], preds_test[:, i])
                for i, col in enumerate(Y_test.columns)
            }
            st.table(pd.DataFrame(mse_vals, index=["MSE"]).T)

        # ----- ROC & EER -----
        with mp2:
            st.write("### ROC Curve & EER")

            response = st.selectbox(
                "Select Output Variable",
                ["ParticleSize", "Entrapment", "CDR"],
                key="fwd_roc"
            )

            idx = Y_test.columns.get_loc(response)
            y_true = (Y_test[response] >= Y_test[response].median()).astype(int)
            y_score = preds_test[:, idx]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fnr = 1 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

            st.write(f"**AUC:** {roc_auc:.3f}")
            st.write(f"**EER:** {eer:.3f}")

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.scatter(fpr[eer_idx], tpr[eer_idx], color="red", label="EER")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve ({response})")
            ax.legend()
            st.pyplot(fig)

# =====================================================
# TAB 2: BACKWARD PREDICTION
# =====================================================
with tabs[1]:

    st.header("🔄 Backward Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        ps = st.number_input("Particle Size (nm)", float(df.ParticleSize.min()), float(df.ParticleSize.max()), float(df.ParticleSize.mean()))
    with c2:
        ent = st.number_input("Entrapment Efficiency (%)", float(df.Entrapment.min()), float(df.Entrapment.max()), float(df.Entrapment.mean()))
    with c3:
        cdr = st.number_input("CDR (%)", float(df.CDR.min()), float(df.CDR.max()), float(df.CDR.mean()))

    if st.button("🔄 Predict Formulation"):

        user_input = pd.DataFrame([[ps, ent, cdr]],
                                  columns=["ParticleSize", "Entrapment", "CDR"])
        preds = bwd_model.predict(user_input)

        st.subheader("📥 User Inputs")
        st.table(user_input)

        st.subheader("📤 Computed Formulation")
        output_df = pd.DataFrame({
            "GMO (%)": [preds[0][0]],
            "Poloxamer 407 (%)": [preds[0][1]],
            "Probe Time (min)": [preds[0][2]]
        })
        st.table(output_df)

        # ================= MODEL PERFORMANCE =================
        st.markdown("---")
        st.subheader("📊 Backward Model Performance")

        preds_test = bwd_model.predict(Y_test)

        mp1, mp2 = st.columns([1, 2])

        # ----- MSE -----
        with mp1:
            st.write("### Mean Squared Error (MSE)")
            mse_vals = {
                col: mean_squared_error(X_test[col], preds_test[:, i])
                for i, col in enumerate(X_test.columns)
            }
            st.table(pd.DataFrame(mse_vals, index=["MSE"]).T)

        # ----- ROC & EER -----
        with mp2:
            st.write("### ROC Curve & EER")

            param = st.selectbox(
                "Select Formulation Parameter",
                ["GMO", "Poloxamer", "ProbeTime"],
                key="bwd_roc"
            )

            idx = X_test.columns.get_loc(param)
            y_true = (X_test[param] >= X_test[param].median()).astype(int)
            y_score = preds_test[:, idx]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fnr = 1 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

            st.write(f"**AUC:** {roc_auc:.3f}")
            st.write(f"**EER:** {eer:.3f}")

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.scatter(fpr[eer_idx], tpr[eer_idx], color="red", label="EER")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve ({param})")
            ax.legend()
            st.pyplot(fig)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("DOE + ML Dashboard | Forward & Backward Prediction with Robust Model Evaluation")
