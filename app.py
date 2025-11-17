import streamlit as st
import pandas as pd
import gspread
import json
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------
# CONFIG BÁSICA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Buscador de Proyectos Inmobiliarios",
    page_icon="🏙️",
    layout="wide",
)

# 👉 RELLENA ESTO CON TU INFO
SHEET_ID = "1PLYS284AMaw8XukpR1107BAkbYDHvO8ARzhllzmwEjY"   # ej: "1SaXuzhY_sJ9Tk9MOLDL..."
SHEET_NAME = "Hoja1"            # nombre de la hoja dentro del archivo

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


# ---------------------------------------------------------
# FUNCIÓN QUE LEE TODA LA HOJA
# ---------------------------------------------------------
@st.cache_data(ttl=60)  # se actualiza solo cada 60 segundos
def load_data():
    """
    Lee TODOS los datos de la hoja de Google Sheets.
    Fila 1 = encabezados.
    """

    # 🔑 Aquí usamos tu secreto GOOGLE_CREDENTIALS_JSON
    # st.secrets["GOOGLE_CREDENTIALS_JSON"] es un STRING con el JSON adentro
    creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"])

    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=SCOPES
    )

    client = gspread.authorize(creds)
    ws = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

    # Esto trae todas las filas y columnas con datos
    data = ws.get_all_values()

    if not data or len(data) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(data[1:], columns=data[0])

    # Intentar convertir algunas columnas a número (si existen)
    numeric_cols = [
        "dormitorios",
        "banos",
        "superficie_util_m2",
        "superficie_terraza_m2",
        "superficie_total_m2",
        "precio_uf",
        "precio_clp_aprox",
        "precio_uf_m2_aprox",
        "gastos_comunes_clp_aprox",
        "contribuciones_trimestrales_clp_aprox",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
st.title("🏙️ Buscador de Proyectos Inmobiliarios")

# Ver rápido qué claves hay en secrets (puedes borrar esta línea luego)
# st.write("Secrets cargados:", list(st.secrets.keys()))

df = load_data()

if df.empty:
    st.warning("No se encontraron datos en la hoja. Revisa SHEET_ID / SHEET_NAME o que haya información.")
    st.stop()

# ----------------- FILTROS SIMPLES -------------------
st.sidebar.header("🔍 Filtros")

df_filtrado = df.copy()

# Filtro por comuna
if "comuna" in df_filtrado.columns:
    comunas = sorted(df_filtrado["comuna"].dropna().unique().tolist())
    comuna_sel = st.sidebar.multiselect("Comuna", comunas, default=comunas)
    df_filtrado = df_filtrado[df_filtrado["comuna"].isin(comuna_sel)]

# Filtro por tipo_unidad
if "tipo_unidad" in df_filtrado.columns:
    tipos = sorted(df_filtrado["tipo_unidad"].dropna().unique().tolist())
    tipo_sel = st.sidebar.multiselect("Tipo unidad", tipos, default=tipos)
    df_filtrado = df_filtrado[df_filtrado["tipo_unidad"].isin(tipo_sel)]

# Filtro por rango de precio UF
if "precio_uf" in df_filtrado.columns and df_filtrado["precio_uf"].notna().any():
    min_uf = float(df_filtrado["precio_uf"].min())
    max_uf = float(df_filtrado["precio_uf"].max())
    rango_uf = st.sidebar.slider(
        "Rango precio UF",
        min_value=min_uf,
        max_value=max_uf,
        value=(min_uf, max_uf),
    )
    df_filtrado = df_filtrado[
        (df_filtrado["precio_uf"] >= rango_uf[0]) &
        (df_filtrado["precio_uf"] <= rango_uf[1])
    ]

# ----------------- RESULTADOS -------------------
st.markdown("## 📊 Resultados filtrados")
st.write(f"Propiedades encontradas: **{len(df_filtrado)}**")

# 👇 IMPORTANTE: NO usamos .head(), mostramos TODO el DataFrame filtrado
st.dataframe(df_filtrado, use_container_width=True)
