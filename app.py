import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="DOE ML Optimization App", layout="centered")

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
    return model

model = train_model()

# -----------------------------
# USER INPUT
# -----------------------------
st.header("🧪 Enter Parameters")

gmo = st.number_input("GMO (%)", float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean()))
poloxamer = st.number_input("Poloxamer 407 (%)", float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean()))
probe_time = st.number_input("Probe Time (min)", float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean()))

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Predict Outputs"):
    user_input = pd.DataFrame([[gmo, poloxamer, probe_time]],
                              columns=["GMO", "Poloxamer", "ProbeTime"])
    pred = model.predict(user_input)

    st.success("Prediction Successful")

    st.metric("Particle Size (nm)", f"{pred[0][0]:.2f}")
    st.metric("Entrapment Efficiency (%)", f"{pred[0][1]:.2f}")
    st.metric("CDR (%)", f"{pred[0][2]:.2f}")

# -----------------------------
# OPTIMIZATION BUTTON
# -----------------------------
st.header("⚙ Optimization")

if st.button("🎯 Find Optimal Formulation"):
    GMO = np.linspace(X.GMO.min(), X.GMO.max(), 10)
    Poloxamer = np.linspace(X.Poloxamer.min(), X.Poloxamer.max(), 10)
    ProbeTime = np.linspace(X.ProbeTime.min(), X.ProbeTime.max(), 10)

    candidates = []
    for g in GMO:
        for p in Poloxamer:
            for t in ProbeTime:
                candidates.append([g, p, t])

    candidates = pd.DataFrame(candidates, columns=["GMO", "Poloxamer", "ProbeTime"])
    preds = model.predict(candidates)

    candidates["ParticleSize"] = preds[:, 0]
    candidates["Entrapment"] = preds[:, 1]
    candidates["CDR"] = preds[:, 2]

    candidates["Score"] = (
        -candidates["ParticleSize"]
        + candidates["Entrapment"]
        + candidates["CDR"]
    )

    best = candidates.loc[candidates.Score.idxmax()]

    st.success("Optimal Formulation Found")

    st.write(best)

# -----------------------------
# PLOTS SECTION
# -----------------------------
st.header("📊 Visualization")

response = st.selectbox("Select Response", ["ParticleSize", "Entrapment", "CDR"])
factor = st.selectbox("Select Factor", ["GMO", "Poloxamer", "ProbeTime"])

fig, ax = plt.subplots()
ax.scatter(df[factor], df[response])
ax.set_xlabel(factor)
ax.set_ylabel(response)
ax.set_title(f"{factor} vs {response}")
st.pyplot(fig)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("DOE + Machine Learning Based Optimization | Academic Project")
