import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="DOE ML Dashboard", layout="wide")
st.title("🔬 DOE + Machine Learning Dashboard")

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
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    fwd = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    bwd = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )

    fwd.fit(X_train, Y_train)
    bwd.fit(Y_train, X_train)

    return fwd, bwd, X_train, X_test, Y_train, Y_test

fwd_model, bwd_model, X_train, X_test, Y_train, Y_test = train_models()

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def classification_metrics(y_true_cont, y_pred_cont):
    thr = np.median(y_true_cont)
    y_true = (y_true_cont >= thr).astype(int)
    y_pred = (y_pred_cont >= thr).astype(int)

    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    )

def compute_eer(y_true_cont, y_score):
    y_true = (y_true_cont >= np.median(y_true_cont)).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return fpr, tpr, eer, eer_idx, auc(fpr, tpr)

def compute_confusion(y_true_cont, y_pred_cont):
    thr = np.median(y_true_cont)
    y_true = (y_true_cont >= thr).astype(int)
    y_pred = (y_pred_cont >= thr).astype(int)
    return confusion_matrix(y_true, y_pred)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔁 Forward Prediction: Formulation → Responses",
    "🔄 Backward Prediction: Responses → Formulation",
    "⚙ Optimization"
])

# =================================================
# TAB 1 – FORWARD
# =================================================
with tab1:
    st.subheader("User Inputs → Computed Outputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        gmo = st.number_input("GMO (%)", float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean()))
    with c2:
        pol = st.number_input("Poloxamer 407 (%)", float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean()))
    with c3:
        pt = st.number_input("Probe Time (min)", float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean()))

    user_X = pd.DataFrame([[gmo, pol, pt]], columns=X.columns)
    pred_Y = fwd_model.predict(user_X)

    st.write("### User Inputs")
    st.dataframe(user_X, use_container_width=True)

    st.write("### Computed Outputs")
    out_df = pd.DataFrame(pred_Y, columns=Y.columns)
    st.dataframe(out_df, use_container_width=True)

    # ---------- MODEL PERFORMANCE ----------
    st.markdown("---")
    st.subheader("📊 Model Performance")

    preds_test = fwd_model.predict(X_test)

    mp1, mp2 = st.columns([1, 1])

    with mp1:
        rows = []
        for i, col in enumerate(Y.columns):
            mse = mean_squared_error(Y_test[col], preds_test[:, i])
            p, r, f1 = classification_metrics(Y_test[col], preds_test[:, i])
            rows.append([col, mse, p, r, f1])

        st.dataframe(
            pd.DataFrame(rows, columns=["Output", "MSE", "Precision", "Recall", "F1"]),
            use_container_width=True
        )

    with mp2:
        resp = st.selectbox("Select Output", Y.columns, key="fwd_roc")
        idx = Y.columns.get_loc(resp)

        fpr, tpr, eer, eer_idx, roc_auc = compute_eer(Y_test[resp], preds_test[:, idx])

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.scatter(fpr[eer_idx], tpr[eer_idx], color="red", label=f"EER={eer:.2f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        st.pyplot(fig)

        cm = compute_confusion(Y_test[resp], preds_test[:, idx])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["Low", "High"]).plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

# =================================================
# TAB 2 – BACKWARD
# =================================================
with tab2:
    st.subheader("User Outputs → Computed Inputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        ps = st.number_input("Particle Size (nm)", float(Y.ParticleSize.min()), float(Y.ParticleSize.max()), float(Y.ParticleSize.mean()))
    with c2:
        ent = st.number_input("Entrapment Efficiency (%)", float(Y.Entrapment.min()), float(Y.Entrapment.max()), float(Y.Entrapment.mean()))
    with c3:
        cdr = st.number_input("CDR (%)", float(Y.CDR.min()), float(Y.CDR.max()), float(Y.CDR.mean()))

    user_Y = pd.DataFrame([[ps, ent, cdr]], columns=Y.columns)
    pred_X = bwd_model.predict(user_Y)

    st.write("### User Outputs")
    st.dataframe(user_Y, use_container_width=True)

    st.write("### Computed Inputs")
    st.dataframe(pd.DataFrame(pred_X, columns=X.columns), use_container_width=True)

    # ---------- MODEL PERFORMANCE ----------
    st.markdown("---")
    st.subheader("📊 Model Performance")

    preds_test = bwd_model.predict(Y_test)

    mp1, mp2 = st.columns([1, 1])

    with mp1:
        rows = []
        for i, col in enumerate(X.columns):
            mse = mean_squared_error(X_test[col], preds_test[:, i])
            p, r, f1 = classification_metrics(X_test[col], preds_test[:, i])
            rows.append([col, mse, p, r, f1])

        st.dataframe(
            pd.DataFrame(rows, columns=["Input", "MSE", "Precision", "Recall", "F1"]),
            use_container_width=True
        )

    with mp2:
        param = st.selectbox("Select Parameter", X.columns, key="bwd_roc")
        idx = X.columns.get_loc(param)

        fpr, tpr, eer, eer_idx, roc_auc = compute_eer(X_test[param], preds_test[:, idx])

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.scatter(fpr[eer_idx], tpr[eer_idx], color="red", label=f"EER={eer:.2f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        st.pyplot(fig)

        cm = compute_confusion(X_test[param], preds_test[:, idx])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["Low", "High"]).plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

# =================================================
# TAB 3 – OPTIMIZATION
# =================================================
with tab3:
    st.subheader("🎯 Optimal Formulation")

    GMO = np.linspace(X.GMO.min(), X.GMO.max(), 10)
    POL = np.linspace(X.Poloxamer.min(), X.Poloxamer.max(), 10)
    PT = np.linspace(X.ProbeTime.min(), X.ProbeTime.max(), 10)

    grid = pd.DataFrame(
        [[g, p, t] for g in GMO for p in POL for t in PT],
        columns=X.columns
    )

    preds = fwd_model.predict(grid)
    grid[Y.columns] = preds

    grid["Score"] = -grid["ParticleSize"] + grid["Entrapment"] + grid["CDR"]
    best = grid.loc[grid.Score.idxmax()]

    st.dataframe(best.to_frame("Optimal Value"), use_container_width=True)
