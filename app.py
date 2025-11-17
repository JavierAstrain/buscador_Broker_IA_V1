import streamlit as st
import pandas as pd
import textwrap
import json
import requests

# Si usas openai oficial nuevo:
from openai import OpenAI

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(
    page_title="Asesor de Propiedades IA para Brokers",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ Asesor IA de Propiedades Nuevas para Brokers")
st.caption("MVP inspirado en la lógica del Controller Fénix Nexa, pero aplicado a proyectos inmobiliarios nuevos.")

# =========================
# FUNCIÓN: CARGAR DATOS
# =========================

@st.cache_data
def load_sheet_from_google(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    # Normalizamos nombres de columnas por si acaso
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def get_default_sheet() -> pd.DataFrame:
    """
    Usa tu planilla de ejemplo como fuente por defecto.
    """
    sheet_base = "https://docs.google.com/spreadsheets/d/1PLYS284AMaw8XukpR1107BAkbYDHvO8ARzhllzmwEjY"
    # Export CSV de la primera hoja (gid=0)
    csv_url = f"{sheet_base}/export?format=csv&gid=0"
    df = load_sheet_from_google(csv_url)
    return df

# =========================
# CARGA DE DATOS
# =========================

st.sidebar.header("📂 Datos")

fuente_datos = st.sidebar.radio(
    "¿De dónde cargamos las propiedades?",
    ["Planilla de ejemplo (Google Sheets)", "Subir archivo (Excel/CSV)"],
)

if fuente_datos == "Planilla de ejemplo (Google Sheets)":
    df = get_default_sheet()
else:
    archivo = st.sidebar.file_uploader("Sube un archivo Excel o CSV", type=["xlsx", "xls", "csv"])
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        st.warning("Por favor sube un archivo para continuar.")
        st.stop()

if df.empty:
    st.error("La tabla de propiedades está vacía.")
    st.stop()

# =========================
# PERFIL DEL CLIENTE
# =========================

st.sidebar.header("👤 Perfil del cliente")

nombre_cliente = st.sidebar.text_input("Nombre del cliente (opcional)", "")
objetivo = st.sidebar.selectbox(
    "Objetivo principal",
    [
        "Primera vivienda",
        "Cambio a una vivienda mejor",
        "Inversión para renta",
        "Inversión para plusvalía",
        "Otro"
    ],
)

monto_uf_min = float(df["precio_uf_desde"].min()) if "precio_uf_desde" in df.columns else 0
monto_uf_max = float(df["precio_uf_hasta"].max()) if "precio_uf_hasta" in df.columns else 10000

rango_uf = st.sidebar.slider(
    "Rango de precio (UF)",
    min_value=int(monto_uf_min),
    max_value=int(monto_uf_max),
    value=(int(monto_uf_min), int(monto_uf_max))
)

# Dormitorios / baños
dorms_min = st.sidebar.selectbox("Dormitorios mínimos", [0, 1, 2, 3, 4, 5], index=2)
banos_min = st.sidebar.selectbox("Baños mínimos", [0, 1, 2, 3, 4], index=1)

# Superficie
if "superficie_total_m2" in df.columns:
    sup_min = float(df["superficie_total_m2"].min())
    sup_max = float(df["superficie_total_m2"].max())
else:
    sup_min, sup_max = 20.0, 150.0

rango_m2 = st.sidebar.slider(
    "Rango de superficie total (m²)",
    min_value=int(sup_min),
    max_value=int(sup_max),
    value=(int(sup_min), int(sup_max))
)

# Filtros por ubicación
comunas = sorted(df["comuna"].dropna().unique()) if "comuna" in df.columns else []
comuna_pref = st.sidebar.multiselect("Comunas deseadas", comunas, default=comunas[:3] if len(comunas) >= 3 else comunas)

# Etapa / entrega
etapas = sorted(df["etapa"].dropna().unique()) if "etapa" in df.columns else []
etapas_selec = st.sidebar.multiselect("Etapa del proyecto", etapas, default=etapas)

anos_entrega = sorted(df["ano_entrega_estimada"].dropna().unique()) if "ano_entrega_estimada" in df.columns else []
anos_selec = st.sidebar.multiselect("Año de entrega estimada", anos_entrega, default=anos_entrega)

estado_comercial = sorted(df["estado_comercial"].dropna().unique()) if "estado_comercial" in df.columns else []
estado_selec = st.sidebar.multiselect("Estado comercial", estado_comercial, default=estado_comercial)

# Comentarios extra
comentarios_extra = st.sidebar.text_area(
    "Notas sobre el cliente (ej. tolerancia al riesgo, preferencia de barrio, bancos, etc.)",
    ""
)

# =========================
# FILTRO DE PROPIEDADES
# =========================

df_filtrado = df.copy()

def aplicar_filtros(df_in):
    df_out = df_in.copy()

    # Precio UF
    if "precio_uf_desde" in df_out.columns:
        df_out = df_out[df_out["precio_uf_desde"].between(rango_uf[0], rango_uf[1])]

    # Dormitorios
    if "dormitorios" in df_out.columns:
        df_out = df_out[df_out["dormitorios"] >= dorms_min]

    # Baños
    if "banos" in df_out.columns:
        df_out = df_out[df_out["banos"] >= banos_min]

    # Superficie
    if "superficie_total_m2" in df_out.columns:
        df_out = df_out[df_out["superficie_total_m2"].between(rango_m2[0], rango_m2[1])]

    # Comuna
    if "comuna" in df_out.columns and comuna_pref:
        df_out = df_out[df_out["comuna"].isin(comuna_pref)]

    # Etapa
    if "etapa" in df_out.columns and etapas_selec:
        df_out = df_out[df_out["etapa"].isin(etapas_selec)]

    # Año entrega
    if "ano_entrega_estimada" in df_out.columns and anos_selec:
        df_out = df_out[df_out["ano_entrega_estimada"].isin(anos_selec)]

    # Estado comercial
    if "estado_comercial" in df_out.columns and estado_selec:
        df_out = df_out[df_out["estado_comercial"].isin(estado_selec)]

    return df_out

df_filtrado = aplicar_filtros(df_filtrado)

st.subheader("📊 Resultados filtrados")

st.write(f"Propiedades encontradas: **{len(df_filtrado)}**")

if len(df_filtrado) == 0:
    st.warning("No hay propiedades que calcen con los filtros. Ajusta los rangos o comunas.")
else:
    # Mostramos algunas columnas relevantes
    columnas_mostrar = [
        c for c in [
            "nombre_proyecto",
            "comuna",
            "tipo_unidad",
            "dormitorios",
            "banos",
            "superficie_total_m2",
            "precio_uf_desde",
            "precio_uf_hasta",
            "etapa",
            "ano_entrega_estimada",
            "estado_comercial",
            "url_portal"
        ] if c in df_filtrado.columns
    ]
    st.dataframe(df_filtrado[columnas_mostrar].reset_index(drop=True))

# =========================
# FUNCIÓN: LLAMADA A LA IA
# =========================

def build_client_profile_text() -> str:
    texto = f"Objetivo principal del cliente: {objetivo}.\n"
    if nombre_cliente:
        texto += f"Nombre del cliente: {nombre_cliente}.\n"
    texto += f"Rango de precio en UF: {rango_uf[0]} - {rango_uf[1]}.\n"
    texto += f"Dormitorios mínimos: {dorms_min}. Baños mínimos: {banos_min}.\n"
    texto += f"Rango de superficie total (m2): {rango_m2[0]} - {rango_m2[1]}.\n"
    if comuna_pref:
        texto += f"Comunas preferidas: {', '.join(comuna_pref)}.\n"
    if etapas_selec:
        texto += f"Etapas aceptables: {', '.join(map(str, etapas_selec))}.\n"
    if anos_selec:
        texto += f"Años de entrega estimada preferidos: {', '.join(map(str, anos_selec))}.\n"
    if estado_selec:
        texto += f"Estados comerciales deseados: {', '.join(map(str, estado_selec))}.\n"
    if comentarios_extra:
        texto += f"Notas adicionales del broker sobre el cliente: {comentarios_extra}\n"
    return texto

def prepare_properties_for_ai(df_props: pd.DataFrame, max_props: int = 25):
    """
    Prepara una lista resumida de propiedades para enviar al modelo.
    """
    df_short = df_props.copy().reset_index(drop=True).head(max_props)
    props = []
    for idx, row in df_short.iterrows():
        prop = {
            "id_interno": idx,
            "nombre_proyecto": row.get("nombre_proyecto", ""),
            "comuna": row.get("comuna", ""),
            "tipo_unidad": row.get("tipo_unidad", ""),
            "dormitorios": int(row.get("dormitorios", 0)) if not pd.isna(row.get("dormitorios", 0)) else 0,
            "banos": int(row.get("banos", 0)) if not pd.isna(row.get("banos", 0)) else 0,
            "superficie_total_m2": float(row.get("superficie_total_m2", 0)) if not pd.isna(row.get("superficie_total_m2", 0)) else 0,
            "precio_uf_desde": float(row.get("precio_uf_desde", 0)) if not pd.isna(row.get("precio_uf_desde", 0)) else 0,
            "precio_uf_hasta": float(row.get("precio_uf_hasta", 0)) if not pd.isna(row.get("precio_uf_hasta", 0)) else 0,
            "etapa": row.get("etapa", ""),
            "ano_entrega_estimada": row.get("ano_entrega_estimada", ""),
            "estado_comercial": row.get("estado_comercial", ""),
            "url_portal": row.get("url_portal", "")
        }
        props.append(prop)
    return props

def call_ai_recommendations(client_profile: str, properties: list, top_k: int = 5):
    """
    Llama a OpenAI (o modelo que uses) para que recomiende propiedades.
    Devuelve una lista de dicts con 'id_interno', 'score', 'motivo', 'argumentos_venta'.
    """

    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("Falta configurar OPENAI_API_KEY en st.secrets.")
        return None

    client = OpenAI(api_key=api_key)

    system_prompt = """
Eres un asistente experto en corretaje inmobiliario en Chile, especializado en proyectos nuevos.
Tu tarea es leer el perfil del cliente y una lista de propiedades, y devolver un ranking de las mejores
opciones para ese cliente.

Debes considerar:
- Objetivo (vivir / inversión / plusvalía).
- Presupuesto en UF.
- Dormitorios, baños y superficie.
- Comuna y entorno.
- Etapa del proyecto y año de entrega.
- Estado comercial (disponible, últimas unidades, etc.).

Responde SIEMPRE en formato JSON con una lista llamada "recomendaciones".
Cada recomendación debe tener:
- id_interno (número de la propiedad que te paso).
- score (1 a 10, 10 es la mejor).
- motivo_principal (texto corto).
- argumentos_venta (lista de 3 a 5 bullets pensados para que el broker se los diga al cliente).
"""

    user_prompt = f"""
PERFIL DEL CLIENTE:
{client_profile}

PROPIEDADES DISPONIBLES (lista en JSON):
{json.dumps(properties, ensure_ascii=False)}

TAREA:
Elige las {top_k} mejores propiedades para este cliente.
Devuélvelas en JSON estrictamente con la forma:

{{
  "recomendaciones": [
    {{
      "id_interno": <numero>,
      "score": <numero>,
      "motivo_principal": "<texto>",
      "argumentos_venta": ["<bullet1>", "<bullet2>", "<bullet3>"]
    }},
    ...
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # o el modelo que prefieras
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content

    # Intentamos parsear JSON
    try:
        data = json.loads(content)
        return data.get("recomendaciones", [])
    except Exception:
        # Intento de rescate si el modelo devuelve texto + JSON mezclado
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            return data.get("recomendaciones", [])
        except Exception:
            st.error("No pude interpretar la respuesta de la IA como JSON.")
            st.text(content)
            return None

# =========================
# UI: BOTÓN DE RECOMENDACIÓN
# =========================

st.subheader("🧠 Recomendación IA")

col1, col2 = st.columns([1, 2])

with col1:
    top_k = st.number_input("Cantidad de propiedades a recomendar", min_value=1, max_value=10, value=5, step=1)

recomendar = st.button("🔍 Pedir recomendación IA")

if recomendar:
    if len(df_filtrado) == 0:
        st.warning("No hay propiedades filtradas para recomendar.")
    else:
        with st.spinner("Analizando propiedades con IA..."):
            perfil_texto = build_client_profile_text()
            props_list = prepare_properties_for_ai(df_filtrado, max_props=50)
            recomendaciones = call_ai_recommendations(perfil_texto, props_list, top_k=int(top_k))

        if recomendaciones:
            st.success("La IA generó recomendaciones para tu cliente:")

            for rec in recomendaciones:
                idx = rec.get("id_interno")
                score = rec.get("score")
                motivo = rec.get("motivo_principal", "")
                argumentos = rec.get("argumentos_venta", [])
                if idx is None or idx >= len(df_filtrado):
                    continue

                fila = df_filtrado.reset_index(drop=True).iloc[idx]

                st.markdown("---")
                st.markdown(f"### ⭐ Recomendación (score {score}/10): {fila.get('nombre_proyecto', 'Proyecto sin nombre')}")
                st.markdown(
                    f"**Comuna:** {fila.get('comuna', '')}  \n"
                    f"**Tipo unidad:** {fila.get('tipo_unidad', '')}  \n"
                    f"**Programa:** {fila.get('dormitorios', '')}D / {fila.get('banos', '')}B  \n"
                    f"**Superficie total aprox.:** {fila.get('superficie_total_m2', '')} m²  \n"
                    f"**Precio desde:** {fila.get('precio_uf_desde', '')} UF  \n"
                    f"**Etapa:** {fila.get('etapa', '')}  \n"
                    f"**Entrega estimada:** {fila.get('ano_entrega_estimada', '')} (T{fila.get('trimestre_entrega_estimada', '')})  \n"
                    f"**Estado comercial:** {fila.get('estado_comercial', '')}"
                )

                st.markdown(f"**Motivo principal:** {motivo}")

                if argumentos:
                    st.markdown("**Argumentos de venta sugeridos:**")
                    for a in argumentos:
                        st.markdown(f"- {a}")

                url_portal = fila.get("url_portal", "")
                if isinstance(url_portal, str) and url_portal.strip():
                    st.markdown(f"[Ver ficha en portal]({url_portal})")

        else:
            st.info("No se recibieron recomendaciones desde la IA. Revisa la configuración o la respuesta mostrada arriba.")
