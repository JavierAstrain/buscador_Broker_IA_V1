import streamlit as st
import pandas as pd
import json
from io import StringIO

from openai import OpenAI

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Broker IA",
    page_icon="🏙️",
    layout="wide",
)

# ---------- Helpers de sesión ----------
def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if "column_map" not in st.session_state:
        st.session_state.column_map = {}

    if "client_profile" not in st.session_state:
        st.session_state.client_profile = {
            "nombre_cliente": "",
            "objetivo": "Primera vivienda",
            "rango_uf": (1500, 15000),
            "dorms_min": 2,
            "banos_min": 1,
            "rango_m2": (30, 120),
            "comunas": [],
            "etapas": [],
            "anos": [],
            "estados": [],
            "comentarios": "",
        }

    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # lista de {"role": "...", "content": "..."}


init_session_state()

# =========================
# UTILIDADES
# =========================

@st.cache_data
def load_sheet_from_google(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    return df


def get_default_sheet() -> pd.DataFrame:
    sheet_base = "https://docs.google.com/spreadsheets/d/1PLYS284AMaw8XukpR1107BAkbYDHvO8ARzhllzmwEjY"
    csv_url = f"{sheet_base}/export?format=csv&gid=0"
    return load_sheet_from_google(csv_url)


def normalize_name(name: str) -> str:
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
    norm_to_real = {normalize_name(c): c for c in df.columns}
    for ln in logical_names:
        if ln in norm_to_real:
            return norm_to_real[ln]
    return None


def map_columns(df: pd.DataFrame) -> dict:
    """Detecta las columnas más importantes de la planilla."""
    column_map = {
        "nombre_proyecto": find_column(df, ["nombre_proyecto", "proyecto", "nombre_del_proyecto"]),
        "comuna": find_column(df, ["comuna"]),
        "tipo_unidad": find_column(df, ["tipo_unidad", "tipo_de_unidad", "tipo"]),
        "dormitorios": find_column(df, ["dormitorios", "dorms"]),
        "banos": find_column(df, ["banos", "baños"]),
        "superficie_total_m2": find_column(df, ["superficie_total_m2", "sup_total_m2", "superficie_total"]),
        "precio_uf_desde": find_column(df, ["precio_uf_desde", "precio_desde_uf", "precio_desde_en_uf"]),
        "precio_uf_hasta": find_column(df, ["precio_uf_hasta", "precio_hasta_uf", "precio_hasta_en_uf"]),
        "etapa": find_column(df, ["etapa"]),
        "ano_entrega_estimada": find_column(df, ["ano_entrega_estimada", "anio_entrega", "ano_entrega"]),
        "trimestre_entrega_estimada": find_column(df, ["trimestre_entrega_estimada", "trimestre_entrega"]),
        "estado_comercial": find_column(df, ["estado_comercial", "estado"]),
        "url_portal": find_column(df, ["url_portal", "url", "link_portal"]),
    }
    return column_map


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("⚠️ Falta configurar OPENAI_API_KEY en st.secrets.")
        return None
    return OpenAI(api_key=api_key)


def get_filtered_df() -> pd.DataFrame:
    """Aplica filtros basados en el perfil del cliente."""
    df = st.session_state.df
    cm = st.session_state.column_map
    cp = st.session_state.client_profile

    if df.empty:
        return df

    df_out = df.copy()

    # Precio UF
    col_precio_desde = cm.get("precio_uf_desde")
    if col_precio_desde and col_precio_desde in df_out.columns:
        df_out = df_out[
            df_out[col_precio_desde].between(cp["rango_uf"][0], cp["rango_uf"][1])
        ]

    # Dormitorios
    col_dorms = cm.get("dormitorios")
    if col_dorms and col_dorms in df_out.columns:
        df_out = df_out[df_out[col_dorms] >= cp["dorms_min"]]

    # Baños
    col_banos = cm.get("banos")
    if col_banos and col_banos in df_out.columns:
        df_out = df_out[df_out[col_banos] >= cp["banos_min"]]

    # Superficie
    col_sup_total = cm.get("superficie_total_m2")
    if col_sup_total and col_sup_total in df_out.columns:
        df_out = df_out[
            df_out[col_sup_total].between(cp["rango_m2"][0], cp["rango_m2"][1])
        ]

    # Comunas
    col_comuna = cm.get("comuna")
    if col_comuna and cp["comunas"]:
        df_out = df_out[df_out[col_comuna].isin(cp["comunas"])]

    # Etapas
    col_etapa = cm.get("etapa")
    if col_etapa and cp["etapas"]:
        df_out = df_out[df_out[col_etapa].isin(cp["etapas"])]

    # Años entrega
    col_ano = cm.get("ano_entrega_estimada")
    if col_ano and cp["anos"]:
        df_out = df_out[df_out[col_ano].isin(cp["anos"])]

    # Estado comercial
    col_estado = cm.get("estado_comercial")
    if col_estado and cp["estados"]:
        df_out = df_out[df_out[col_estado].isin(cp["estados"])]

    return df_out


def build_client_profile_text() -> str:
    cp = st.session_state.client_profile
    texto = f"Objetivo del cliente: {cp['objetivo']}.\n"
    if cp["nombre_cliente"]:
        texto += f"Nombre del cliente: {cp['nombre_cliente']}.\n"
    texto += f"Rango de precio en UF: {cp['rango_uf'][0]} - {cp['rango_uf'][1]}.\n"
    texto += f"Dormitorios mínimos: {cp['dorms_min']}. Baños mínimos: {cp['banos_min']}.\n"
    texto += f"Rango de superficie total: {cp['rango_m2'][0]} - {cp['rango_m2'][1]} m2.\n"
    if cp["comunas"]:
        texto += "Comunas preferidas: " + ", ".join(map(str, cp["comunas"])) + ".\n"
    if cp["etapas"]:
        texto += "Etapas preferidas: " + ", ".join(map(str, cp["etapas"])) + ".\n"
    if cp["anos"]:
        texto += "Años de entrega estimados: " + ", ".join(map(str, cp["anos"])) + ".\n"
    if cp["estados"]:
        texto += "Estados comerciales preferidos: " + ", ".join(map(str, cp["estados"])) + ".\n"
    if cp["comentarios"]:
        texto += f"Notas adicionales del broker: {cp['comentarios']}\n"
    return texto


def prepare_properties_for_ai(df_props: pd.DataFrame, max_props: int = 50):
    cm = st.session_state.column_map
    df_short = df_props.reset_index(drop=True).head(max_props)
    props = []

    for idx, row in df_short.iterrows():
        prop = {"id_interno": int(idx)}
        for key, col in cm.items():
            if col and col in row:
                value = row[col]
                if pd.isna(value):
                    value = ""
                prop[key] = value
        props.append(prop)

    return props


# =========================
# IA: RECOMENDACIONES Y CHAT
# =========================

def ia_recomendaciones(client_profile: str, properties: list, top_k: int = 5):
    client = get_openai_client()
    if client is None:
        return None

    system_prompt = """
Eres un asesor inmobiliario senior en Chile, experto en proyectos nuevos.
Analizas el perfil del cliente y una lista de propiedades.
Objetivo:

1. Seleccionar las mejores propiedades para ese cliente.
2. Explicar por qué son buenas opciones.
3. Dar argumentos comerciales concretos para que el broker cierre la venta.
4. Proponer una estrategia general de inversión (renta, plusvalía, portafolio, etc.).

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
        model="gpt-4.1",
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
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
        except Exception:
            st.error("No pude interpretar la respuesta de la IA como JSON.")
            st.code(content)
            return None

    return data


def ia_chat_libre(mensaje_usuario: str):
    client = get_openai_client()
    if client is None:
        return None

    df_filtrado = get_filtered_df()
    props_list = prepare_properties_for_ai(df_filtrado, max_props=40)
    perfil_texto = build_client_profile_text()

    system_prompt = """
Eres un asesor inmobiliario y de inversiones experto en Chile.
Tienes:
- Perfil del cliente.
- Lista de propiedades nuevas filtradas.
- Historial de conversación con un broker.

Objetivo:
- Dar respuestas claras, accionables y profesionales.
- Recomendar comunas, tipos de unidades, estrategias (renta, plusvalía, portafolio).
- Usar UF y CLP, realidad chilena y lenguaje cercano pero profesional.
"""

    context = {
        "perfil_cliente": perfil_texto,
        "propiedades_resumen": props_list,
    }

    messages = [{"role": "system", "content": system_prompt.strip()}]

    # Contexto
    messages.append({
        "role": "user",
        "content": "Contexto inicial (no responder todavía, solo incorpora): " + json.dumps(context, ensure_ascii=False)[:11000]
    })
    messages.append({"role": "assistant", "content": "Contexto recibido. Ahora esperaré las preguntas del broker."})

    # Historial de chat
    for m in st.session_state.chat_history:
        messages.append({"role": m["role"], "content": m["content"]})

    # Último mensaje del usuario
    messages.append({"role": "user", "content": mensaje_usuario})

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.5,
    )

    return resp.choices[0].message.content


# =========================
# VISTAS (DERECHA)
# =========================

def show_fuente_propiedades():
    st.header("📂 Fuente de propiedades")

    st.markdown(
        "Configura desde dónde se cargan las propiedades que Broker IA utilizará "
        "para analizar y recomendar proyectos."
    )

    fuente = st.radio(
        "Selecciona la fuente de datos",
        ["Planilla de ejemplo (Google Sheets)", "Subir archivo (Excel/CSV)"],
    )

    if fuente == "Planilla de ejemplo (Google Sheets)":
        if st.button("Cargar planilla de ejemplo"):
            df = get_default_sheet()
            st.session_state.df = df
            st.session_state.column_map = map_columns(df)
            st.success("Planilla de ejemplo cargada correctamente.")
    else:
        archivo = st.file_uploader("Sube un archivo Excel o CSV", type=["xlsx", "xls", "csv"])
        if archivo is not None:
            if archivo.name.endswith(".csv"):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)
            st.session_state.df = df
            st.session_state.column_map = map_columns(df)
            st.success("Archivo cargado correctamente.")

    if not st.session_state.df.empty:
        st.subheader("Vista rápida de la planilla")
        st.dataframe(st.session_state.df.head(10))

        st.subheader("Columnas detectadas")
        st.json(st.session_state.column_map)
    else:
        st.info("Aún no se han cargado propiedades. Usa una de las opciones de arriba.")


def show_dashboard():
    st.header("📊 Dashboard de propiedades")

    df = st.session_state.df
    cm = st.session_state.column_map

    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    df_filtrado = get_filtered_df()

    col1, col2, col3, col4 = st.columns(4)

    total_props = len(df_filtrado)
    col1.metric("Propiedades filtradas", total_props)

    col_precio_desde = cm.get("precio_uf_desde")
    col_precio_hasta = cm.get("precio_uf_hasta")
    col_dorms = cm.get("dormitorios")
    col_comuna = cm.get("comuna")
    col_ano = cm.get("ano_entrega_estimada")

    if col_precio_desde and col_precio_desde in df_filtrado.columns:
        col2.metric("UF promedio (desde)", f"{df_filtrado[col_precio_desde].mean():.0f}")
    if col_precio_hasta and col_precio_hasta in df_filtrado.columns:
        col3.metric("UF promedio (hasta)", f"{df_filtrado[col_precio_hasta].mean():.0f}")
    if col_dorms and col_dorms in df_filtrado.columns:
        col4.metric("Dormitorios más frecuente", int(df_filtrado[col_dorms].mode()[0]))

    st.markdown("---")

    # Gráfico por comuna
    if col_comuna and col_comuna in df_filtrado.columns:
        st.subheader("Distribución de propiedades por comuna")
        comuna_counts = df_filtrado[col_comuna].value_counts().reset_index()
        comuna_counts.columns = [col_comuna, "cantidad"]
        st.bar_chart(comuna_counts.set_index(col_comuna))

    # Gráfico por tipo de unidad
    col_tipo = cm.get("tipo_unidad")
    if col_tipo and col_tipo in df_filtrado.columns:
        st.subheader("Distribución por tipo de unidad")
        tipo_counts = df_filtrado[col_tipo].value_counts().reset_index()
        tipo_counts.columns = [col_tipo, "cantidad"]
        st.bar_chart(tipo_counts.set_index(col_tipo))

    # Gráfico por año de entrega
    if col_ano and col_ano in df_filtrado.columns:
        st.subheader("Unidades por año de entrega estimada")
        ano_counts = df_filtrado[col_ano].value_counts().sort_index().reset_index()
        ano_counts.columns = [col_ano, "cantidad"]
        st.line_chart(ano_counts.set_index(col_ano))

    # Relación precio vs superficie
    col_sup = cm.get("superficie_total_m2")
    if col_precio_desde and col_sup and col_precio_desde in df_filtrado.columns and col_sup in df_filtrado.columns:
        st.subheader("Relación precio UF (desde) vs superficie total")
        scatter_df = df_filtrado[[col_precio_desde, col_sup]].dropna()
        scatter_df = scatter_df.rename(columns={col_precio_desde: "UF_desde", col_sup: "m2"})
        st.scatter_chart(scatter_df, x="m2", y="UF_desde")


def show_perfil_cliente():
    st.header("👤 Perfil del cliente")

    df = st.session_state.df
    cm = st.session_state.column_map
    cp = st.session_state.client_profile

    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    with st.form("perfil_cliente_form"):
        cp["nombre_cliente"] = st.text_input("Nombre del cliente (opcional)", value=cp["nombre_cliente"])

        cp["objetivo"] = st.selectbox(
            "Objetivo principal",
            [
                "Primera vivienda",
                "Cambio de vivienda",
                "Inversión para renta",
                "Inversión para plusvalía",
                "Portafolio de inversión (varias propiedades)",
                "Otro",
            ],
            index=[
                "Primera vivienda",
                "Cambio de vivienda",
                "Inversión para renta",
                "Inversión para plusvalía",
                "Portafolio de inversión (varias propiedades)",
                "Otro",
            ].index(cp["objetivo"]),
        )

        col_precio_desde = cm.get("precio_uf_desde")
        if col_precio_desde and col_precio_desde in df.columns:
            uf_min = int(df[col_precio_desde].min())
            uf_max = int(df[col_precio_desde].max())
        else:
            uf_min, uf_max = 1500, 15000

        cp["rango_uf"] = st.slider(
            "Rango de precio en UF",
            min_value=uf_min,
            max_value=uf_max,
            value=(cp["rango_uf"][0], cp["rango_uf"][1]),
        )

        col1, col2 = st.columns(2)
        with col1:
            cp["dorms_min"] = st.selectbox(
                "Dormitorios mínimos", [0, 1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4, 5].index(cp["dorms_min"])
            )
        with col2:
            cp["banos_min"] = st.selectbox(
                "Baños mínimos", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(cp["banos_min"])
            )

        col_sup = cm.get("superficie_total_m2")
        if col_sup and col_sup in df.columns:
            sup_min = int(df[col_sup].min())
            sup_max = int(df[col_sup].max())
        else:
            sup_min, sup_max = 20, 150

        cp["rango_m2"] = st.slider(
            "Rango de superficie total (m²)",
            min_value=sup_min,
            max_value=sup_max,
            value=(cp["rango_m2"][0], cp["rango_m2"][1]),
        )

        col_comuna = cm.get("comuna")
        if col_comuna and col_comuna in df.columns:
            comunas = sorted(df[col_comuna].dropna().unique())
            cp["comunas"] = st.multiselect(
                "Comunas preferidas",
                comunas,
                default=cp["comunas"] if cp["comunas"] else comunas[:3],
            )

        col_etapa = cm.get("etapa")
        if col_etapa and col_etapa in df.columns:
            etapas = sorted(df[col_etapa].dropna().unique())
            cp["etapas"] = st.multiselect(
                "Etapas aceptables",
                etapas,
                default=cp["etapas"] if cp["etapas"] else etapas,
            )

        col_ano = cm.get("ano_entrega_estimada")
        if col_ano and col_ano in df.columns:
            anos = sorted(df[col_ano].dropna().unique())
            cp["anos"] = st.multiselect(
                "Años de entrega estimada",
                anos,
                default=cp["anos"] if cp["anos"] else anos,
            )

        col_estado = cm.get("estado_comercial")
        if col_estado and col_estado in df.columns:
            estados = sorted(df[col_estado].dropna().unique())
            cp["estados"] = st.multiselect(
                "Estado comercial",
                estados,
                default=cp["estados"] if cp["estados"] else estados,
            )

        cp["comentarios"] = st.text_area(
            "Notas adicionales del broker sobre el cliente",
            value=cp["comentarios"],
            height=120,
        )

        submitted = st.form_submit_button("💾 Guardar perfil")

    st.session_state.client_profile = cp

    if submitted:
        st.success("Perfil de cliente guardado. Se usará en explorador, recomendaciones y Agente IA.")

    st.markdown("### Resumen del perfil actual")
    st.code(build_client_profile_text())


def show_explorador():
    st.header("🔍 Explorador de propiedades")

    df = st.session_state.df
    cm = st.session_state.column_map

    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    df_filtrado = get_filtered_df()
    st.write(f"Propiedades encontradas con el perfil actual: **{len(df_filtrado)}**")

    if df_filtrado.empty:
        st.info("No hay propiedades con los filtros actuales. Ajusta el perfil del cliente.")
        return

    cols_mostrar = [c for c in [
        cm.get("nombre_proyecto"),
        cm.get("comuna"),
        cm.get("tipo_unidad"),
        cm.get("dormitorios"),
        cm.get("banos"),
        cm.get("superficie_total_m2"),
        cm.get("precio_uf_desde"),
        cm.get("precio_uf_hasta"),
        cm.get("etapa"),
        cm.get("ano_entrega_estimada"),
        cm.get("estado_comercial"),
        cm.get("url_portal"),
    ] if c is not None]

    st.dataframe(df_filtrado[cols_mostrar].reset_index(drop=True))


def show_recomendaciones():
    st.header("🧠 Recomendaciones IA")

    df = st.session_state.df
    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    df_filtrado = get_filtered_df()
    if df_filtrado.empty:
        st.info("No hay propiedades filtradas. Ajusta el perfil del cliente.")
        return

    st.markdown("El motor IA analizará el perfil del cliente y las propiedades filtradas para sugerir las mejores opciones.")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        top_k = st.number_input(
            "Cantidad de propiedades a recomendar",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )
        pedir = st.button("🔍 Generar recomendaciones IA")

    if pedir:
        with st.spinner("Analizando propiedades para este cliente..."):
            perfil_texto = build_client_profile_text()
            props_list = prepare_properties_for_ai(df_filtrado, max_props=60)
            data = ia_recomendaciones(perfil_texto, props_list, top_k=int(top_k))

        if data:
            st.session_state.last_recommendations = {
                "data": data,
                "df_filtrado": df_filtrado.reset_index(drop=True).to_dict("records"),
            }

    # Mostrar recomendaciones guardadas (última ejecución)
    if st.session_state.last_recommendations:
        data = st.session_state.last_recommendations["data"]
        df_snap = pd.DataFrame(st.session_state.last_recommendations["df_filtrado"])

        estrategia = data.get("estrategia_general", "")
        recomendaciones = data.get("recomendaciones", [])

        if estrategia:
            st.markdown("### 🧭 Estrategia general sugerida")
            st.write(estrategia)

        st.markdown("### ⭐ Propiedades recomendadas")

        cm = st.session_state.column_map

        for rec in recomendaciones:
            idx = rec.get("id_interno")
            score = rec.get("score")
            motivo = rec.get("motivo_principal", "")
            argumentos = rec.get("argumentos_venta", [])

            if idx is None or idx >= len(df_snap):
                continue

            fila = df_snap.iloc[idx]

            st.markdown("---")
            titulo = (
                fila.get(cm.get("nombre_proyecto"), "Proyecto sin nombre")
                if cm.get("nombre_proyecto")
                else f"Proyecto #{idx}"
            )
            st.markdown(f"#### ⭐ Score {score}/10 – {titulo}")

            info_lineas = []
            if cm.get("comuna"):
                info_lineas.append(f"**Comuna:** {fila.get(cm['comuna'], '')}")
            if cm.get("tipo_unidad"):
                info_lineas.append(f"**Tipo unidad:** {fila.get(cm['tipo_unidad'], '')}")
            if cm.get("dormitorios") and cm.get("banos"):
                info_lineas.append(
                    f"**Programa:** {fila.get(cm['dormitorios'], '')}D / {fila.get(cm['banos'], '')}B"
                )
            if cm.get("superficie_total_m2"):
                info_lineas.append(f"**Sup. total aprox.:** {fila.get(cm['superficie_total_m2'], '')} m²")
            if cm.get("precio_uf_desde"):
                info_lineas.append(f"**Precio desde:** {fila.get(cm['precio_uf_desde'], '')} UF")
            if cm.get("etapa"):
                info_lineas.append(f"**Etapa:** {fila.get(cm['etapa'], '')}")
            if cm.get("ano_entrega_estimada"):
                info_lineas.append(f"**Entrega estimada:** {fila.get(cm['ano_entrega_estimada'], '')}")
            if cm.get("estado_comercial"):
                info_lineas.append(f"**Estado comercial:** {fila.get(cm['estado_comercial'], '')}")

            st.markdown("  \n".join(info_lineas))

            st.markdown(f"**Motivo principal:** {motivo}")

            if argumentos:
                st.markdown("**Argumentos de venta sugeridos:**")
                for a in argumentos:
                    st.markdown(f"- {a}")

            if cm.get("url_portal"):
                url = fila.get(cm["url_portal"], "")
                if isinstance(url, str) and url.strip():
                    st.markdown(f"[Ver ficha en portal]({url})")
    else:
        st.info("Aún no se han generado recomendaciones. Usa el botón de arriba.")


def show_agente_chat():
    st.header("🤖 Agente IA (chat)")

    df = st.session_state.df
    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    df_filtrado = get_filtered_df()
    if df_filtrado.empty:
        st.info("No hay propiedades filtradas. Ajusta el perfil del cliente antes de hablar con el agente.")
        return

    # Mostrar historial
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(m["content"])

    # Input de chat
    user_input = st.chat_input("Haz una pregunta al Agente IA (estrategias, comparaciones, etc.)")

    if user_input:
        # Guardar mensaje usuario
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Pensando respuesta..."):
                respuesta = ia_chat_libre(user_input)
                st.markdown(respuesta)
        st.session_state.chat_history.append({"role": "assistant", "content": respuesta})

    st.markdown("---")
    if st.button("🧹 Limpiar conversación"):
        st.session_state.chat_history = []
        st.experimental_rerun()


def show_exportar():
    st.header("📤 Exportar propuesta")

    if st.session_state.df.empty:
        st.warning("Primero carga propiedades en **Fuente de propiedades**.")
        return

    if not st.session_state.last_recommendations:
        st.info("Aún no hay recomendaciones guardadas. Genera recomendaciones en la sección **Recomendaciones IA**.")
        return

    data = st.session_state.last_recommendations["data"]
    df_snap = pd.DataFrame(st.session_state.last_recommendations["df_filtrado"])
    cm = st.session_state.column_map
    perfil_texto = build_client_profile_text()

    estrategia = data.get("estrategia_general", "")
    recomendaciones = data.get("recomendaciones", [])

    st.markdown("Esta sección genera un resumen en texto (formato Markdown) con el perfil del cliente y el plan sugerido.")
    st.markdown("Luego puedes descargar el archivo y usarlo como base para enviar una propuesta al cliente.")

    # Construir contenido markdown
    md = StringIO()
    md.write("# Propuesta Broker IA\n\n")
    md.write("## Perfil del cliente\n\n")
    md.write(f"```\n{perfil_texto}\n```\n\n")

    md.write("## Estrategia general sugerida\n\n")
    md.write(f"{estrategia}\n\n")

    md.write("## Propiedades recomendadas\n\n")
    for rec in recomendaciones:
        idx = rec.get("id_interno")
        if idx is None or idx >= len(df_snap):
            continue
        fila = df_snap.iloc[idx]

        titulo = (
            fila.get(cm.get("nombre_proyecto"), "Proyecto sin nombre")
            if cm.get("nombre_proyecto")
            else f"Proyecto #{idx}"
        )

        md.write(f"### {titulo}\n\n")

        # Datos básicos
        if cm.get("comuna"):
            md.write(f"- **Comuna:** {fila.get(cm['comuna'], '')}\n")
        if cm.get("tipo_unidad"):
            md.write(f"- **Tipo unidad:** {fila.get(cm['tipo_unidad'], '')}\n")
        if cm.get("dormitorios") and cm.get("banos"):
            md.write(f"- **Programa:** {fila.get(cm['dormitorios'], '')}D / {fila.get(cm['banos'], '')}B\n")
        if cm.get("superficie_total_m2"):
            md.write(f"- **Superficie total aprox.:** {fila.get(cm['superficie_total_m2'], '')} m²\n")
        if cm.get("precio_uf_desde"):
            md.write(f"- **Precio desde:** {fila.get(cm['precio_uf_desde'], '')} UF\n")
        if cm.get("etapa"):
            md.write(f"- **Etapa:** {fila.get(cm['etapa'], '')}\n")
        if cm.get("ano_entrega_estimada"):
            md.write(f"- **Entrega estimada:** {fila.get(cm['ano_entrega_estimada'], '')}\n")
        if cm.get("estado_comercial"):
            md.write(f"- **Estado comercial:** {fila.get(cm['estado_comercial'], '')}\n")
        if cm.get("url_portal"):
            url = fila.get(cm["url_portal"], "")
            if isinstance(url, str) and url.strip():
                md.write(f"- **Link portal:** {url}\n")

        # IA
        md.write(f"- **Score IA:** {rec.get('score')}/10\n")
        md.write(f"- **Motivo principal:** {rec.get('motivo_principal', '')}\n")
        argumentos = rec.get("argumentos_venta", [])
        if argumentos:
            md.write("- **Argumentos de venta sugeridos:**\n")
            for a in argumentos:
                md.write(f"  - {a}\n")

        md.write("\n")

    contenido_md = md.getvalue()

    st.markdown("### Vista previa")
    st.markdown(contenido_md)

    st.download_button(
        label="⬇️ Descargar propuesta en Markdown",
        data=contenido_md,
        file_name="propuesta_broker_ia.md",
        mime="text/markdown",
    )


# =========================
# MENÚ LATERAL (IZQUIERDA)
# =========================

st.sidebar.title("Broker IA")
st.sidebar.caption("Asistente inmobiliario potenciado con IA")

menu = st.sidebar.radio(
    "Menú",
    [
        "Fuente de propiedades",
        "Dashboard",
        "Perfil del cliente",
        "Explorador",
        "Recomendaciones IA",
        "Agente IA",
        "Exportar propuesta",
    ],
)

# =========================
# ROUTER DE VISTAS
# =========================

if menu == "Fuente de propiedades":
    show_fuente_propiedades()
elif menu == "Dashboard":
    show_dashboard()
elif menu == "Perfil del cliente":
    show_perfil_cliente()
elif menu == "Explorador":
    show_explorador()
elif menu == "Recomendaciones IA":
    show_recomendaciones()
elif menu == "Agente IA":
    show_agente_chat()
elif menu == "Exportar propuesta":
    show_exportar()
