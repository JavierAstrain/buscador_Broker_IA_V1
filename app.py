import streamlit as st
import pandas as pd
import json

from openai import OpenAI

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Asesor IA para Brokers Inmobiliarios",
    page_icon="🏙️",
    layout="wide",
)

st.title("🏙️ Asesor IA de Propiedades Nuevas para Brokers")
st.caption("Buscador + Asesor Inmobiliario Potenciado con IA (MVP Nexa).")

# =========================
# UTILIDADES
# =========================

@st.cache_data
def load_sheet_from_google(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    return df

def get_default_sheet() -> pd.DataFrame:
    """Planilla de ejemplo (tu Google Sheet)."""
    sheet_base = "https://docs.google.com/spreadsheets/d/1PLYS284AMaw8XukpR1107BAkbYDHvO8ARzhllzmwEjY"
    csv_url = f"{sheet_base}/export?format=csv&gid=0"
    df = load_sheet_from_google(csv_url)
    return df

def normalize_name(name: str) -> str:
    """Normaliza un nombre de columna para poder matchearlo."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )

def find_column(df: pd.DataFrame, logical_names) -> str | None:
    """
    Busca una columna en el DF matcheando contra una lista de nombres lógicos
    (normalizados). Devuelve el nombre REAL de la columna del DF.
    """
    norm_to_real = {normalize_name(c): c for c in df.columns}
    for ln in logical_names:
        if ln in norm_to_real:
            return norm_to_real[ln]
    return None

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("⚠️ Falta configurar OPENAI_API_KEY en st.secrets.")
        return None
    return OpenAI(api_key=api_key)

# =========================
# CARGA DE DATOS
# =========================

st.sidebar.header("📂 Datos")

fuente_datos = st.sidebar.radio(
    "¿Fuente de propiedades?",
    ["Planilla de ejemplo (Google Sheets)", "Subir archivo (Excel/CSV)"],
)

if fuente_datos == "Planilla de ejemplo (Google Sheets)":
    df = get_default_sheet()
else:
    archivo = st.sidebar.file_uploader("Sube archivo Excel o CSV", type=["xlsx", "xls", "csv"])
    if archivo is None:
        st.warning("Sube un archivo para continuar.")
        st.stop()
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)

if df.empty:
    st.error("La tabla de propiedades está vacía.")
    st.stop()

# Mostrar columnas para debug rápido
with st.expander("Ver columnas detectadas de la planilla"):
    st.write(list(df.columns))

# =========================
# MAPEO DE COLUMNAS
# =========================

col_nombre_proyecto = find_column(df, ["nombre_proyecto", "proyecto", "nombre_del_proyecto"])
col_comuna = find_column(df, ["comuna"])
col_tipo_unidad = find_column(df, ["tipo_unidad", "tipo_de_unidad", "tipo"])
col_dorms = find_column(df, ["dormitorios", "dorms"])
col_banos = find_column(df, ["banos", "baños"])
col_sup_total = find_column(df, ["superficie_total_m2", "sup_total_m2", "superficie_total"])
col_precio_desde = find_column(df, ["precio_uf_desde", "precio_desde_uf", "precio_desde_en_uf"])
col_precio_hasta = find_column(df, ["precio_uf_hasta", "precio_hasta_uf", "precio_hasta_en_uf"])
col_etapa = find_column(df, ["etapa"])
col_ano_entrega = find_column(df, ["ano_entrega_estimada", "anio_entrega", "ano_entrega"])
col_trim_entrega = find_column(df, ["trimestre_entrega_estimada", "trimestre_entrega"])
col_estado_com = find_column(df, ["estado_comercial", "estado"])
col_url = find_column(df, ["url_portal", "url", "link_portal"])

# =========================
# PERFIL DEL CLIENTE (SIDEBAR)
# =========================

st.sidebar.header("👤 Perfil del cliente")

nombre_cliente = st.sidebar.text_input("Nombre del cliente (opcional)")

objetivo = st.sidebar.selectbox(
    "Objetivo del cliente",
    [
        "Primera vivienda",
        "Cambio de vivienda",
        "Inversión para renta",
        "Inversión para plusvalía",
        "Portafolio de inversión (varias propiedades)",
        "Otro",
    ],
)

# Rango de precio en UF (si no tenemos columnas, usamos rango dummy)
if col_precio_desde and col_precio_hasta:
    uf_min = float(df[col_precio_desde].min())
    uf_max = float(df[col_precio_hasta].max())
else:
    uf_min, uf_max = 1500, 15000

rango_uf = st.sidebar.slider(
    "Rango de precio (UF)",
    min_value=int(uf_min),
    max_value=int(uf_max),
    value=(int(uf_min), int(uf_max)),
)

# Dorms y baños
dorms_min = st.sidebar.selectbox("Dormitorios mínimos", [0, 1, 2, 3, 4, 5], index=2)
banos_min = st.sidebar.selectbox("Baños mínimos", [0, 1, 2, 3, 4], index=1)

# Superficie
if col_sup_total:
    sup_min = float(df[col_sup_total].min())
    sup_max = float(df[col_sup_total].max())
else:
    sup_min, sup_max = 20, 150

rango_m2 = st.sidebar.slider(
    "Rango de superficie total (m²)",
    min_value=int(sup_min),
    max_value=int(sup_max),
    value=(int(sup_min), int(sup_max)),
)

# Comunas
if col_comuna:
    comunas = sorted(df[col_comuna].dropna().unique())
    comuna_pref = st.sidebar.multiselect(
        "Comunas preferidas",
        comunas,
        default=comunas[:3] if len(comunas) >= 3 else comunas,
    )
else:
    comuna_pref = []

# Etapa y año entrega
if col_etapa:
    etapas = sorted(df[col_etapa].dropna().unique())
    etapas_selec = st.sidebar.multiselect("Etapa del proyecto", etapas, default=etapas)
else:
    etapas_selec = []

if col_ano_entrega:
    anos = sorted(df[col_ano_entrega].dropna().unique())
    anos_selec = st.sidebar.multiselect("Año entrega estimada", anos, default=anos)
else:
    anos_selec = []

if col_estado_com:
    estados = sorted(df[col_estado_com].dropna().unique())
    estados_selec = st.sidebar.multiselect("Estado comercial", estados, default=estados)
else:
    estados_selec = []

comentarios_extra = st.sidebar.text_area(
    "Notas del broker sobre el cliente",
    placeholder="Ej: prefiere bancos específicos, nivel de riesgo, barrios preferidos, etc.",
)

# =========================
# FILTRADO DE PROPIEDADES
# =========================

def aplicar_filtros(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()

    # Precio UF
    if col_precio_desde:
        df_out = df_out[
            df_out[col_precio_desde].between(rango_uf[0], rango_uf[1])
        ]

    # Dorms
    if col_dorms:
        df_out = df_out[df_out[col_dorms] >= dorms_min]

    # Baños
    if col_banos:
        df_out = df_out[df_out[col_banos] >= banos_min]

    # Superficie
    if col_sup_total:
        df_out = df_out[
            df_out[col_sup_total].between(rango_m2[0], rango_m2[1])
        ]

    # Comuna
    if col_comuna and comuna_pref:
        df_out = df_out[df_out[col_comuna].isin(comuna_pref)]

    # Etapa
    if col_etapa and etapas_selec:
        df_out = df_out[df_out[col_etapa].isin(etapas_selec)]

    # Año entrega
    if col_ano_entrega and anos_selec:
        df_out = df_out[df_out[col_ano_entrega].isin(anos_selec)]

    # Estado
    if col_estado_com and estados_selec:
        df_out = df_out[df_out[col_estado_com].isin(estados_selec)]

    return df_out

df_filtrado = aplicar_filtros(df)

# =========================
# TEXTO PERFIL CLIENTE PARA IA
# =========================

def build_client_profile_text() -> str:
    texto = f"Objetivo del cliente: {objetivo}.\n"
    if nombre_cliente:
        texto += f"Nombre del cliente: {nombre_cliente}.\n"
    texto += f"Rango de precio en UF: {rango_uf[0]} - {rango_uf[1]}.\n"
    texto += f"Dormitorios mínimos: {dorms_min}. Baños mínimos: {banos_min}.\n"
    texto += f"Rango de superficie total: {rango_m2[0]} - {rango_m2[1]} m2.\n"
    if comuna_pref:
        texto += "Comunas preferidas: " + ", ".join(map(str, comuna_pref)) + ".\n"
    if etapas_selec:
        texto += "Etapas aceptables: " + ", ".join(map(str, etapas_selec)) + ".\n"
    if anos_selec:
        texto += "Años de entrega estimada: " + ", ".join(map(str, anos_selec)) + ".\n"
    if estados_selec:
        texto += "Estados comerciales preferidos: " + ", ".join(map(str, estados_selec)) + ".\n"
    if comentarios_extra:
        texto += f"Notas adicionales del broker: {comentarios_extra}\n"
    return texto

def prepare_properties_for_ai(df_props: pd.DataFrame, max_props: int = 40):
    """Convierte el DF a lista simple para enviar a la IA."""
    df_short = df_props.reset_index(drop=True).head(max_props)
    props = []
    for idx, row in df_short.iterrows():
        prop = {"id_interno": int(idx)}
        if col_nombre_proyecto: prop["nombre_proyecto"] = row.get(col_nombre_proyecto, "")
        if col_comuna: prop["comuna"] = row.get(col_comuna, "")
        if col_tipo_unidad: prop["tipo_unidad"] = row.get(col_tipo_unidad, "")
        if col_dorms: prop["dormitorios"] = int(row.get(col_dorms, 0)) if pd.notna(row.get(col_dorms, 0)) else 0
        if col_banos: prop["banos"] = int(row.get(col_banos, 0)) if pd.notna(row.get(col_banos, 0)) else 0
        if col_sup_total: prop["superficie_total_m2"] = float(row.get(col_sup_total, 0)) if pd.notna(row.get(col_sup_total, 0)) else 0.0
        if col_precio_desde: prop["precio_uf_desde"] = float(row.get(col_precio_desde, 0)) if pd.notna(row.get(col_precio_desde, 0)) else 0.0
        if col_precio_hasta: prop["precio_uf_hasta"] = float(row.get(col_precio_hasta, 0)) if pd.notna(row.get(col_precio_hasta, 0)) else 0.0
        if col_etapa: prop["etapa"] = row.get(col_etapa, "")
        if col_ano_entrega: prop["ano_entrega_estimada"] = row.get(col_ano_entrega, "")
        if col_trim_entrega: prop["trimestre_entrega_estimada"] = row.get(col_trim_entrega, "")
        if col_estado_com: prop["estado_comercial"] = row.get(col_estado_com, "")
        if col_url: prop["url_portal"] = row.get(col_url, "")
        props.append(prop)
    return props

# =========================
# LLAMADAS A IA
# =========================

def ia_recomendaciones(client_profile: str, properties: list, top_k: int = 5):
    client = get_openai_client()
    if client is None:
        return None

    system_prompt = """
Eres un asesor inmobiliario senior en Chile, experto en inversión en departamentos y casas nuevas.
Analizas el perfil del cliente y una lista de propiedades disponibles.
Tu objetivo es:

1. Seleccionar las mejores propiedades para ese cliente.
2. Explicar por qué son buenas opciones.
3. Dar argumentos comerciales que pueda usar un broker para cerrar la venta.
4. Si aplica, sugerir estrategias de inversión (aprovechar plusvalía, arriendo, diversificación, etc.).

RESPUESTA OBLIGATORIA EN FORMATO JSON:

{
  "recomendaciones": [
    {
      "id_interno": <numero>,
      "score": <numero 1-10>,
      "motivo_principal": "<texto>",
      "argumentos_venta": ["<bullet1>", "<bullet2>", "<bullet3>"]
    }
  ],
  "estrategia_general": "<texto explicando la estrategia inmobiliaria para este cliente>"
}
"""
    user_prompt = f"""
PERFIL DEL CLIENTE:
{client_profile}

PROPIEDADES DISPONIBLES (JSON):
{json.dumps(properties, ensure_ascii=False)}

TAREA:
- Elige las {top_k} mejores propiedades.
- Devuelve el JSON EXACTAMENTE con la forma indicada.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",  # puedes cambiar a gpt-4.1-mini si quieres
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.4,
    )

    content = resp.choices[0].message.content

    try:
        data = json.loads(content)
    except Exception:
        # Intento rescatar el JSON dentro del texto
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
        except Exception:
            st.error("No pude interpretar la respuesta de la IA como JSON.")
            st.code(content)
            return None

    return data

def ia_chat_libre(client_question: str, client_profile: str, properties: list):
    """
    Chat general de estrategia: puede hablar de inversión,
    qué tipo de propiedades convienen, etc., usando las propiedades filtradas como contexto.
    """
    client = get_openai_client()
    if client is None:
        return None

    system_prompt = """
Eres un asesor inmobiliario y de inversiones experto en Chile.
Te pasan:

- Perfil de un cliente.
- Lista de propiedades nuevas (contexto).
- Una pregunta de un broker.

Tu objetivo:
- Responder de forma clara y accionable.
- Recomendar tipos de proyectos, comunas, rangos de UF, y estrategias de inversión (plazo, arriendo, reventa).
- Cuando tenga sentido, referenciar ejemplos de propiedades del contexto (por nombre de proyecto o comuna).
- Siempre responder para un público chileno (UF, CLP, bancos, realidad local).
"""

    context = f"""
PERFIL DEL CLIENTE:
{client_profile}

PROPIEDADES DISPONIBLES (RESUMEN JSON):
{json.dumps(properties, ensure_ascii=False)[:12000]}  # cortesía para no pasar demasiado
"""

    user_prompt = f"""
CONTEXT0 (NO RESPONDAS AÚN, SOLO USA COMO REFERENCIA):
{context}

PREGUNTA DEL BROKER:
{client_question}

Por favor responde como un asesor experto que quiere ayudar al broker a cerrar una buena operación para el cliente.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.5,
    )

    return resp.choices[0].message.content

# =========================
# UI PRINCIPAL: TABS
# =========================

tab_explorer, tab_recom, tab_chat = st.tabs(
    ["🔍 Explorador de propiedades", "🧠 Recomendador IA", "💬 Chat IA (estrategias)"]
)

# ---- TAB 1: EXPLORADOR ----
with tab_explorer:
    st.subheader("🔍 Propiedades filtradas")
    st.write(f"Propiedades encontradas con los filtros: **{len(df_filtrado)}**")

    if len(df_filtrado) == 0:
        st.warning("No hay propiedades que calcen. Ajusta los filtros en la barra lateral.")
    else:
        cols_mostrar = [c for c in [
            col_nombre_proyecto,
            col_comuna,
            col_tipo_unidad,
            col_dorms,
            col_banos,
            col_sup_total,
            col_precio_desde,
            col_precio_hasta,
            col_etapa,
            col_ano_entrega,
            col_estado_com,
            col_url,
        ] if c is not None]

        st.dataframe(df_filtrado[cols_mostrar].reset_index(drop=True))

# ---- TAB 2: RECOMENDADOR IA ----
with tab_recom:
    st.subheader("🧠 Recomendación IA de propiedades concretas")

    if len(df_filtrado) == 0:
        st.warning("Primero ajusta filtros para tener propiedades candidatas.")
    else:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            top_k = st.number_input(
                "Cantidad de propiedades a recomendar",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
            )

            pedir_recom = st.button("🔍 Pedir recomendación IA")

        if pedir_recom:
            with st.spinner("Analizando propiedades para este cliente..."):
                perfil_texto = build_client_profile_text()
                props_list = prepare_properties_for_ai(df_filtrado, max_props=60)
                data = ia_recomendaciones(perfil_texto, props_list, top_k=int(top_k))

            if data:
                recomendaciones = data.get("recomendaciones", [])
                estrategia = data.get("estrategia_general", "")

                if estrategia:
                    st.markdown("### 🧭 Estrategia general sugerida")
                    st.write(estrategia)

                st.markdown("### ⭐ Propiedades recomendadas")

                for rec in recomendaciones:
                    idx = rec.get("id_interno")
                    score = rec.get("score")
                    motivo = rec.get("motivo_principal", "")
                    argumentos = rec.get("argumentos_venta", [])

                    if idx is None or idx >= len(df_filtrado):
                        continue

                    fila = df_filtrado.reset_index(drop=True).iloc[idx]

                    st.markdown("---")
                    titulo = (
                        fila.get(col_nombre_proyecto, "Proyecto sin nombre")
                        if col_nombre_proyecto
                        else f"Proyecto #{idx}"
                    )
                    st.markdown(f"#### ⭐ Score {score}/10 – {titulo}")

                    info_lineas = []
                    if col_comuna: info_lineas.append(f"**Comuna:** {fila.get(col_comuna, '')}")
                    if col_tipo_unidad: info_lineas.append(f"**Tipo unidad:** {fila.get(col_tipo_unidad, '')}")
                    if col_dorms and col_banos:
                        info_lineas.append(
                            f"**Programa:** {fila.get(col_dorms, '')}D / {fila.get(col_banos, '')}B"
                        )
                    if col_sup_total:
                        info_lineas.append(f"**Sup. total aprox.:** {fila.get(col_sup_total, '')} m²")
                    if col_precio_desde:
                        info_lineas.append(f"**Precio desde:** {fila.get(col_precio_desde, '')} UF")
                    if col_etapa:
                        info_lineas.append(f"**Etapa:** {fila.get(col_etapa, '')}")
                    if col_ano_entrega:
                        info_lineas.append(f"**Entrega estimada:** {fila.get(col_ano_entrega, '')}")
                    if col_estado_com:
                        info_lineas.append(f"**Estado comercial:** {fila.get(col_estado_com, '')}")

                    st.markdown("  \n".join(info_lineas))

                    st.markdown(f"**Motivo principal:** {motivo}")

                    if argumentos:
                        st.markdown("**Argumentos de venta sugeridos:**")
                        for a in argumentos:
                            st.markdown(f"- {a}")

                    if col_url:
                        url = fila.get(col_url, "")
                        if isinstance(url, str) and url.strip():
                            st.markdown(f"[Ver ficha en portal]({url})")

# ---- TAB 3: CHAT IA ----
with tab_chat:
    st.subheader("💬 Chat con Asesor IA (estrategias y dudas)")

    if len(df_filtrado) == 0:
        st.warning("Primero filtra propiedades en la barra lateral para dar contexto al asesor IA.")
    else:
        pregunta = st.text_area(
            "Escribe tu pregunta para el asesor IA",
            placeholder=(
                "Ejemplos:\n"
                "- ¿Qué estrategia de inversión recomiendas para este cliente con este presupuesto?\n"
                "- ¿Conviene más un 2D2B o un 1D1B en estas comunas?\n"
                "- ¿Cómo armarías una propuesta comparando 3 proyectos para este cliente?\n"
            ),
            height=150,
        )

        if st.button("💬 Consultar IA"):
            if not pregunta.strip():
                st.warning("Escribe una pregunta primero.")
            else:
                with st.spinner("Pensando respuesta estratégica..."):
                    perfil_texto = build_client_profile_text()
                    props_list = prepare_properties_for_ai(df_filtrado, max_props=50)
                    respuesta = ia_chat_libre(pregunta, perfil_texto, props_list)

                st.markdown("### Respuesta del asesor IA")
                st.write(respuesta)
