# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Estimador de RUL — Rodamientos", layout="centered")
st.title("🔧 Estimador de RUL — Rodamientos XJTU-SY")
st.caption("Predicción de vida útil remanente a partir de features de vibración")

paquete = joblib.load("model.pkl")
modelo = paquete["modelo"]
cols = paquete["features"]
rul_cap = paquete.get("rul_cap", 60)

datos = pd.read_csv("datos_muestra.csv")

st.subheader("1. Selecciona un caso a evaluar")
opciones = datos.apply(lambda r: f"{r['bearing']} — t={int(r['t'])} min ({r['split']})", axis=1)
idx_sel = st.selectbox("Caso de ejemplo:", options=datos.index, format_func=lambda i: opciones[i])

fila = datos.loc[[idx_sel]]

st.subheader("2. Features de entrada (vibración)")
st.dataframe(fila[cols].T.rename(columns={fila.index[0]: "valor"}))

if st.button("🔍 Predecir RUL"):
    pred = modelo.predict(fila[cols])[0]
    real = fila['rul_capped'].values[0]

    col1, col2 = st.columns(2)
    col1.metric("RUL estimado", f"{pred:.1f} min")
    col2.metric("RUL real (referencia histórica)", f"{real:.0f} min")

    error = abs(pred - real)
    st.write(f"Error absoluto: {error:.1f} min ({error/rul_cap*100:.1f}% sobre el tope de {rul_cap} min)")

    if fila['split'].values[0] == 'test':
        st.info("✅ Este caso NO fue visto por el modelo durante el entrenamiento (fuera de muestra).")
    else:
        st.warning("⚠️ Este caso fue parte del entrenamiento del modelo (in-sample).")