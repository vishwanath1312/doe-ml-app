import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

from skopt import gp_minimize
from skopt.space import Real

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="DOE–ML Dashboard", layout="wide")
st.title("🧪 DOE–ML Formulation Intelligence Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
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

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
@st.cache_resource
def train_models():
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    fwd_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    fwd_model.fit(X_tr, Y_tr)

    bwd_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    bwd_model.fit(Y_tr, X_tr)

    return fwd_model, bwd_model, X_te, Y_te

fwd_model, bwd_model, X_test, Y_test = train_models()

# -------------------------------------------------
# METRICS
# -------------------------------------------------
def classification_metrics(y_true, y_pred):
    thr = np.median(y_true)
    yt = (y_true >= thr).astype(int)
    yp = (y_pred >= thr).astype(int)
    return (
        precision_score(yt, yp, zero_division=0),
        recall_score(yt, yp, zero_division=0),
        f1_score(yt, yp, zero_division=0),
        confusion_matrix(yt, yp),
        roc_curve(yt, yp)
    )

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Forward Prediction",
    "🔄 Backward Prediction",
    "⚙ Optimization",
    "📈 Visualization"
])

# =================================================
# TAB 1 — FORWARD
# =================================================
with tab1:
    st.subheader("Formulation → Responses")

    c1, c2, c3 = st.columns(3)
    with c1:
        gmo = st.slider("GMO (%)", float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean()), 0.1)
    with c2:
        pol = st.slider("Poloxamer 407 (%)", float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean()), 0.1)
    with c3:
        pt = st.slider("Probe Time (min)", float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean()), 0.1)

    user_X = pd.DataFrame([[gmo, pol, pt]], columns=X.columns)
    preds = fwd_model.predict(user_X)[0]

    st.markdown("### 🔢 Predicted Outputs")
    st.dataframe(pd.DataFrame({
        "Response": Y.columns,
        "Predicted Value": preds
    }), use_container_width=True)

    # -------- PERFORMANCE --------
    st.markdown("---")
    st.subheader("📊 Model Performance")

    Y_pred_test = fwd_model.predict(X_test)

    rows = []
    for i, col in enumerate(Y.columns):
        mse = mean_squared_error(Y_test[col], Y_pred_test[:, i])
        p, r, f1, cm, (fpr, tpr, _) = classification_metrics(
            Y_test[col], Y_pred_test[:, i]
        )
        rows.append([col, mse, p, r, f1])

    st.dataframe(pd.DataFrame(
        rows, columns=["Output", "MSE", "Precision", "Recall", "F1"]
    ), use_container_width=True)

    # ROC + Confusion Matrix
    target = st.selectbox("Select Output for ROC", Y.columns)
    idx = Y.columns.get_loc(target)

    p, r, f1, cm, (fpr, tpr, _) = classification_metrics(
        Y_test[target], Y_pred_test[:, idx]
    )
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.legend()
    ax.set_title("ROC Curve")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

# =================================================
# TAB 2 — BACKWARD
# =================================================
with tab2:
    st.subheader("Responses → Formulation")

    c1, c2, c3 = st.columns(3)
    with c1:
        ps = st.number_input("Particle Size (nm)", float(Y.ParticleSize.min()), float(Y.ParticleSize.max()), float(Y.ParticleSize.mean()))
    with c2:
        ent = st.number_input("Entrapment (%)", float(Y.Entrapment.min()), float(Y.Entrapment.max()), float(Y.Entrapment.mean()))
    with c3:
        cdr = st.number_input("CDR (%)", float(Y.CDR.min()), float(Y.CDR.max()), float(Y.CDR.mean()))

    user_Y = pd.DataFrame([[ps, ent, cdr]], columns=Y.columns)
    pred_X = bwd_model.predict(user_Y)[0]

    st.markdown("### 🔢 Suggested Formulation")
    st.dataframe(pd.DataFrame({
        "Parameter": X.columns,
        "Predicted Value": pred_X
    }), use_container_width=True)

# =================================================
# TAB 3 — OPTIMIZATION
# =================================================
with tab3:
    st.subheader("🎯 Bayesian Optimization")

    def objective(x):
        g, p, t = x
        Xr = pd.DataFrame([[g, p, t]], columns=X.columns)
        ps, ee, cdr = fwd_model.predict(Xr)[0]
        return ps - ee - cdr  # minimize PS, maximize EE & CDR

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
            "Parameter": X.columns,
            "Optimal Value": res.x
        }), use_container_width=True)

# =================================================
# TAB 4 — VISUALIZATION
# =================================================
with tab4:
    st.subheader("📈 DOE Visualizations")

    fig = px.scatter_3d(
        df,
        x="GMO",
        y="Poloxamer",
        z="ProbeTime",
        color="CDR",
        title="3D DOE Space (CDR)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Data Preview")
    st.dataframe(df.head(), use_container_width=True)
