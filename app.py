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
st.set_page_config(page_title="DOE ML Optimization App", layout="wide")
st.title("🔬 DOE + ML Based Formulation Optimization")
st.write("Predict & optimize formulation parameters using Machine Learning")

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
        RandomForestRegressor(n_estimators=200, random_state=42)
    )
    model.fit(X_train, Y_train)
    return model, X_test, Y_test

model, X_test, Y_test = train_model()

# -----------------------------
# RESULTS TAB
# -----------------------------
st.header("📊 Results Dashboard: Prediction, Model Performance & Optimization")

# Create three columns
col1, col2, col3 = st.columns(3)

# -----------------------------
# COLUMN 1: Prediction
# -----------------------------
with col1:
    st.subheader("🧪 Prediction")
    gmo = st.number_input("GMO (%)", float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean()))
    poloxamer = st.number_input("Poloxamer 407 (%)", float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean()))
    probe_time = st.number_input("Probe Time (min)", float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean()))

    if st.button("🔍 Predict & Evaluate"):
        user_input = pd.DataFrame([[gmo, poloxamer, probe_time]],
                                  columns=["GMO", "Poloxamer", "ProbeTime"])
        pred = model.predict(user_input)

        st.success("Prediction Successful")
        st.metric("Particle Size (nm)", f"{pred[0][0]:.2f}")
        st.metric("Entrapment Efficiency (%)", f"{pred[0][1]:.2f}")
        st.metric("CDR (%)", f"{pred[0][2]:.2f}")

        # -----------------------------
        # COLUMN 2: Model Performance (MSE + ROC/EER)
        # -----------------------------
        with col2:
            st.subheader("📈 Model Performance")
            preds_test = model.predict(X_test)

            # MSE
            st.write("**Mean Squared Error (MSE)**")
            mse_dict = {}
            for idx, col_name in enumerate(Y_test.columns):
                mse_dict[col_name] = mean_squared_error(Y_test[col_name], preds_test[:, idx])
                st.write(f"{col_name}: {mse_dict[col_name]:.2f}")

            # Interactive ROC/EER
            st.write("**ROC Curve & Equal Error Rate (EER)**")
            response_choice = st.selectbox("Select Response for ROC/EER", ["ParticleSize", "Entrapment", "CDR"], key="roc_response")
            idx = Y_test.columns.get_loc(response_choice)

            y_true = (Y_test[response_choice] > Y_test[response_choice].median()).astype(int)
            y_score = preds_test[:, idx]

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
            eer = fpr[eer_idx]

            st.write(f"**{response_choice}:** ROC AUC = {roc_auc:.2f}, EER = {eer:.2f}")

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"ROC Curve ({response_choice})")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # -----------------------------
        # COLUMN 3: Optimization
        # -----------------------------
        with col3:
            st.subheader("🎯 Optimization")
            GMO_grid = np.linspace(X.GMO.min(), X.GMO.max(), 10)
            Poloxamer_grid = np.linspace(X.Poloxamer.min(), X.Poloxamer.max(), 10)
            ProbeTime_grid = np.linspace(X.ProbeTime.min(), X.ProbeTime.max(), 10)

            candidates = pd.DataFrame([[g, p, t] for g in GMO_grid for p in Poloxamer_grid for t in ProbeTime_grid],
                                      columns=["GMO", "Poloxamer", "ProbeTime"])
            preds_grid = model.predict(candidates)
            candidates["ParticleSize"] = preds_grid[:, 0]
            candidates["Entrapment"] = preds_grid[:, 1]
            candidates["CDR"] = preds_grid[:, 2]
            candidates["Score"] = -candidates["ParticleSize"] + candidates["Entrapment"] + candidates["CDR"]

            best = candidates.loc[candidates.Score.idxmax()]
            st.success("Optimal Formulation Found")
            st.write(best)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("DOE + Machine Learning Based Optimization | Academic Project")
