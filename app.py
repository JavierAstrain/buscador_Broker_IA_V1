import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# --------------------------------------------------------------------
# CONFIG BÁSICA
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Buscador de Proyectos Inmobiliarios",
    page_icon="🏙️",
    layout="wide",
)

# 👇 Cambia esto por tu ID real de Google Sheet y nombre de hoja
SHEET_ID = "TU_SHEET_ID_AQUI"      # ej: 1SaXuzhY_sJ9Tk9MOLDL...
SHEET_NAME = "Hoja1"               # o el nombre real de tu hoja

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


# --------------------------------------------------------------------
# FUNCIÓN PARA LEER TODA LA PLANILLA DESDE GOOGLE SHEETS
# --------------------------------------------------------------------
@st.cache_data(ttl=60)  # vuelve a leer cada 60 segundos
def load_data():
    """
    Lee TODOS los datos de la hoja de Google Sheets.
    Fila 1 = encabezados. Resto = datos.
    """
    # El JSON del service account debe ir en st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

    # 👇 Esto trae TODAS las filas y columnas con datos
    data = sheet.get_all_values()

    if not data or len(data) < 2:
        return pd.DataFrame()

    # Primera fila = encabezados
    df = pd.DataFrame(data[1:], columns=data[0])

    # Opcional: intentar convertir algunas columnas a números
    numeric_cols = [
        "dormitorios", "banos",
        "superficie_util_m2", "superficie_terraza_m2", "superficie_total_m2",
        "precio_uf", "precio_clp_aprox", "precio_uf_m2_aprox",
        "gastos_comunes_clp_aprox", "contribuciones_trimestrales_clp_aprox"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------------------------
# UI PRINCIPAL
# --------------------------------------------------------------------
st.title("🏙️ Buscador de Proyectos Inmobiliarios")
st.markdown(
    "Esta app lee **directamente desde tu Google Sheet** y muestra "
    "todas las propiedades filtradas en tiempo real."
)

# Botón para refrescar manualmente
col_refresh, _ = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Actualizar datos desde Google Sheets"):
        load_data.clear()          # limpiamos cache
        st.experimental_rerun()    # recargamos app

# Cargar datos
df = load_data()

if df.empty:
    st.warning("No se encontraron datos en la hoja. Revisa el SHEET_ID, SHEET_NAME o que haya información.")
    st.stop()

# --------------------------------------------------------------------
# SIDEBAR DE FILTROS
# --------------------------------------------------------------------
st.sidebar.header("🔍 Filtros")

# Filtro comuna
if "comuna" in df.columns:
    comunas = sorted(df["comuna"].dropna().unique().tolist())
    comuna_sel = st.sidebar.multiselect("Comuna", options=comunas, default=comunas)
else:
    comuna_sel = None

# Filtro tipo_unidad
if "tipo_unidad" in df.columns:
    tipos = sorted(df["tipo_unidad"].dropna().unique().tolist())
    tipo_sel = st.sidebar.multiselect("Tipo de unidad", options=tipos, default=tipos)
else:
    tipo_sel = None

# Filtro dormitorios
if "dormitorios" in df.columns and df["dormitorios"].notna().any():
    min_dorm = int(df["dormitorios"].min())
    max_dorm = int(df["dormitorios"].max())
    dorm_sel = st.sidebar.slider("Dormitorios", min_dorm, max_dorm, (min_dorm, max_dorm))
else:
    dorm_sel = None

# Filtro precio UF
if "precio_uf" in df.columns and df["precio_uf"].notna().any():
    min_uf = float(df["precio_uf"].min())
    max_uf = float(df["precio_uf"].max())
    uf_sel = st.sidebar.slider("Rango precio UF", float(min_uf), float(max_uf), (float(min_uf), float(max_uf)))
else:
    uf_sel = None

# --------------------------------------------------------------------
# APLICAR FILTROS
# --------------------------------------------------------------------
df_filtrado = df.copy()

if comuna_sel is not None:
    df_filtrado = df_filtrado[df_filtrado["comuna"].isin(comuna_sel)]

if tipo_sel is not None:
    df_filtrado = df_filtrado[df_filtrado["tipo_unidad"].isin(tipo_sel)]

if dorm_sel is not None and "dormitorios" in df_filtrado.columns:
    df_filtrado = df_filtrado[
        (df_filtrado["dormitorios"] >= dorm_sel[0]) &
        (df_filtrado["dormitorios"] <= dorm_sel[1])
    ]

if uf_sel is not None and "precio_uf" in df_filtrado.columns:
    df_filtrado = df_filtrado[
        (df_filtrado["precio_uf"] >= uf_sel[0]) &
        (df_filtrado["precio_uf"] <= uf_sel[1])
    ]

# --------------------------------------------------------------------
# RESULTADOS
# --------------------------------------------------------------------
st.markdown("## 📊 Resultados filtrados")
st.write(f"Propiedades encontradas: **{len(df_filtrado)}**")

# ⚠️ IMPORTANTE: NO usamos .head() ni cortamos el DataFrame
st.dataframe(df_filtrado, use_container_width=True)

# Opcional: columnas clave para ver rápidamente detalles
if "url_proyecto" in df_filtrado.columns or "url_mapa_google" in df_filtrado.columns:
    st.markdown("### 🔗 Links rápidos (fila seleccionada)")

    # Seleccionar índice de fila
    idx = st.number_input(
        "Índice de fila (posición en la tabla filtrada)",
        min_value=0,
        max_value=max(len(df_filtrado) - 1, 0),
        value=0,
        step=1
    )

    fila = df_filtrado.iloc[int(idx)]

    if "url_proyecto" in df_filtrado.columns:
        st.write("**URL proyecto:**", fila.get("url_proyecto", "N/A"))
    if "url_mapa_google" in df_filtrado.columns:
        st.write("**URL mapa Google:**", fila.get("url_mapa_google", "N/A"))
