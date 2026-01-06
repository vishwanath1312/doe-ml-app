import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import Binarizer

from skopt import gp_minimize
from skopt.space import Real

import shap

st.set_page_config(page_title="DOE–ML Formulation Dashboard", layout="wide")
st.title("🧪 DOE–ML Formulation Intelligence Dashboard")

# ======================================================
# DATA LOADING
# ======================================================
@st.cache_data
def load_data():
    # Replace with your actual dataset
    return pd.read_csv("formulation_data.csv")

df = load_data()

X = df[["GMO", "Poloxamer", "ProbeTime"]]
Y = df[["ParticleSize", "EntrapmentEfficiency", "CDR"]]

# ======================================================
# MODELS
# ======================================================
@st.cache_resource
def train_models():
    fwd = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=42)
    fwd.fit(X, Y)

    bwd = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=42)
    bwd.fit(Y, X)

    return fwd, bwd

fwd_model, bwd_model = train_models()

# ======================================================
# PERFORMANCE METRICS
# ======================================================
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")

    binarizer = Binarizer(threshold=np.mean(y_true))
    y_true_bin = binarizer.fit_transform(y_true)
    y_pred_bin = binarizer.transform(y_pred)

    precision = precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

    cm = confusion_matrix(y_true_bin.ravel(), y_pred_bin.ravel())

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc = auc(fpr, tpr)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return mse, precision, recall, f1, cm, fpr, tpr, roc_auc, eer

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Forward Prediction",
    "🔄 Backward Prediction",
    "⚙ Optimization",
    "📈 Visualization"
])

# ======================================================
# TAB 1 — FORWARD
# ======================================================
with tab1:
    st.subheader("Forward Prediction (Formulation → Responses)")

    col1, col2, col3 = st.columns(3)
    with col1:
        gmo = st.number_input("GMO (%)", 0.0, 100.0, 5.0)
    with col2:
        pol = st.number_input("Poloxamer 407 (%)", 0.0, 100.0, 5.0)
    with col3:
        pt = st.number_input("Probe Time (min)", 1.0, 60.0, 10.0)

    inp = pd.DataFrame([[gmo, pol, pt]], columns=X.columns)
    pred = fwd_model.predict(inp)[0]

    st.markdown("### 🔢 Predicted Outputs")
    st.dataframe(pd.DataFrame({
        "Output": ["Particle Size (nm)", "Entrapment Efficiency (%)", "CDR (%)"],
        "Predicted Value": pred
    }))

    st.markdown("### 📊 Model Performance")
    Y_pred_all = fwd_model.predict(X)
    mse, p, r, f1, cm, fpr, tpr, aucv, eer = compute_metrics(Y.values, Y_pred_all)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("MSE (Particle Size)", f"{mse[0]:.3f}")
        st.metric("MSE (EE)", f"{mse[1]:.3f}")
        st.metric("MSE (CDR)", f"{mse[2]:.3f}")
    with c2:
        st.metric("Precision", f"{p:.3f}")
        st.metric("Recall", f"{r:.3f}")
        st.metric("F1 Score", f"{f1:.3f}")
        st.metric("EER", f"{eer:.3f}")
    with c3:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        st.pyplot(fig)

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ======================================================
# TAB 2 — BACKWARD
# ======================================================
with tab2:
    st.subheader("Backward Prediction (Responses → Formulation)")

    c1, c2, c3 = st.columns(3)
    with c1:
        ps = st.number_input("Particle Size (nm)", 50.0, 1000.0, 200.0)
    with c2:
        ee = st.number_input("Entrapment Efficiency (%)", 0.0, 100.0, 70.0)
    with c3:
        cdr = st.number_input("CDR (%)", 0.0, 100.0, 80.0)

    inp_b = pd.DataFrame([[ps, ee, cdr]], columns=Y.columns)
    pred_b = bwd_model.predict(inp_b)[0]

    st.markdown("### 🔢 Computed Inputs")
    st.dataframe(pd.DataFrame({
        "Input": ["GMO (%)", "Poloxamer (%)", "Probe Time (min)"],
        "Computed Value": pred_b
    }))

    st.markdown("### 📊 Model Performance")
    X_pred_all = bwd_model.predict(Y)
    mse, p, r, f1, cm, fpr, tpr, aucv, eer = compute_metrics(X.values, X_pred_all)

    st.metric("EER", f"{eer:.3f}")

# ======================================================
# TAB 3 — BAYESIAN OPTIMIZATION
# ======================================================
with tab3:
    st.subheader("Bayesian Optimization")

    def objective(x):
        g, p, t = x
        Xr = pd.DataFrame([[g, p, t]], columns=X.columns)
        ps, ee, cdr = fwd_model.predict(Xr)[0]
        return ps * 0.4 - ee * 0.3 - cdr * 0.3

    if st.button("Run Optimization"):
        res = gp_minimize(
            objective,
            [
                Real(X.GMO.min(), X.GMO.max()),
                Real(X.Poloxamer.min(), X.Poloxamer.max()),
                Real(X.ProbeTime.min(), X.ProbeTime.max())
            ],
            n_calls=30,
            random_state=42
        )

        st.success("Optimal Formulation Found")
        st.dataframe(pd.DataFrame({
            "Parameter": ["GMO", "Poloxamer", "Probe Time"],
            "Optimal Value": res.x
        }))

# ======================================================
# TAB 4 — VISUALIZATION
# ======================================================
with tab4:
    st.subheader("DOE Visualizations")

    fig = px.scatter_3d(
        df,
        x="GMO",
        y="Poloxamer",
        z="ProbeTime",
        color="CDR",
        title="3D DOE Space (CDR)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SHAP Sensitivity")
    explainer = shap.TreeExplainer(fwd_model)
    shap_vals = explainer.shap_values(X)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X, show=False)
    st.pyplot(fig)
