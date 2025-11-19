import streamlit as st
import pandas as pd
import json
from io import StringIO
import altair as alt
import requests
from datetime import datetime
from openai import OpenAI
import base64

# -------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------

st.set_page_config(
    page_title="Broker IA",
    page_icon="🏙️",
    layout="wide",
)

# -------------------------------------------------------
# CONTEXTO EXPERTO CHILE (para IA)
# -------------------------------------------------------

CHILE_CONTEXT = """
Eres un analista inmobiliario chileno experto (años 2023–2025). Usa este conocimiento SIEMPRE:

🔵 Comunas con plusvalía estable o creciente:
- Ñuñoa, Providencia, Macul, San Miguel, La Florida (zonas conectadas)
- Las Condes, Vitacura, Lo Barnechea (alto estándar)
- Independencia y Recoleta (ejes conectados)

🔴 Comunas de riesgo:
- Estación Central: sobreoferta, micro-unidades, baja plusvalía.
- Santiago Centro: sectores saturados, vacancia alta.
- Evitar micro-unidades < 25–28 m2.

📈 Tendencias Chile 2023–2025:
- Tasas hipotecarias bajando levemente.
- Fuerte demanda de arriendo.
- Renta estable en zonas conectadas (Metro).
- Entrega inmediata muy atractiva para inversionistas.
- Unidades 2D/2B → mejor equilibrio para renta y reventa.

📉 Evitar recomendar:
- Proyectos densos, mala habitabilidad.
- Zonas de riesgo delictual fuerte.
- Micro-unidades poco revendibles.

Siempre responde con datos reales, tono profesional y lógica del mercado chileno.
"""

# -------------------------------------------------------
# ESTADO DE SESIÓN
# -------------------------------------------------------

def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if "column_map" not in st.session_state:
        st.session_state.column_map = {}

    if "client_profile" not in st.session_state:
        st.session_state.client_profile = {
            "nombre_cliente": "",
            "objetivo": "Inversión",
            "rango_uf": (1500, 15000),
            "dorms_min": 1,
            "banos_min": 1,
            "rango_m2": (25, 120),
            "comunas": [],
            "etapas": [],
            "anos": [],
            "estados": [],
            "comentarios": "",
        }

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = None

    # Tema UI
    if "theme" not in st.session_state:
        st.session_state.theme = {
            "primary_color": "#2563EB",
            "secondary_color": "#0F172A",
            "bg_color": "#F8FAFC",
        }

    # Fuentes de noticias Chile
    if "news_sources" not in st.session_state:
        st.session_state.news_sources = [
            "https://www.df.cl/rss",
            "https://www.latercera.com/rss/",
            "https://www.elmostrador.cl/feed/",
        ]

init_session_state()

# -------------------------------------------------------
# CSS PERSONALIZADO
# -------------------------------------------------------

def inject_custom_css():
    theme = st.session_state.theme
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme['bg_color']};
        }}
        section[data-testid="stSidebar"] > div {{
            background-color: {theme['secondary_color']};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E5E7EB !important;
        }}
        .stButton > button {{
            background-color: {theme['primary_color']};
            color: white;
            border-radius: 8px;
        }}
        .stButton > button:hover {{
            opacity: 0.9;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# -------------------------------------------------------
# UTILIDADES DE COLUMNAS Y PLANILLAS
# -------------------------------------------------------

def normalize_name(name):
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

def find_column(df, names):
    norm_map = {normalize_name(c): c for c in df.columns}
    for name in names:
        if name in norm_map:
            return norm_map[name]
    return None

def map_columns(df):
    return {
        "nombre_proyecto": find_column(df, ["proyecto", "nombre_proyecto"]),
        "comuna": find_column(df, ["comuna"]),
        "tipo_unidad": find_column(df, ["tipo_unidad", "tipologia", "tipo"]),
        "dormitorios": find_column(df, ["dormitorios", "n_dormitorios"]),
        "banos": find_column(df, ["banos", "baños", "n_banos"]),
        "superficie_total_m2": find_column(df, ["superficie_total_m2", "m2", "superficie"]),
        "precio_uf_desde": find_column(df, ["precio_uf_desde", "precio_desde_uf"]),
        "precio_uf_hasta": find_column(df, ["precio_uf_hasta", "precio_hasta_uf"]),
        "etapa": find_column(df, ["etapa"]),
        "ano_entrega_estimada": find_column(df, ["ano_entrega", "anio_entrega"]),
        "estado_comercial": find_column(df, ["estado", "estado_comercial"]),
        "url_portal": find_column(df, ["url", "link", "url_proyecto"]),
    }

@st.cache_data
def load_sheet(url):
    return pd.read_csv(url)

def get_example_sheet():
    base = "https://docs.google.com/spreadsheets/d/1PLYS284AMaw8XukpR1107BAkbYDHvO8ARzhllzmwEjY"
    return load_sheet(f"{base}/export?format=csv&gid=0")

def _ensure_numeric(df, col):
    if col in df.columns:
        serie = df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
        return pd.to_numeric(serie, errors="coerce")
    return pd.Series([pd.NA] * len(df))

def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        st.error("Falta OPENAI_API_KEY en st.secrets")
        return None
    return OpenAI(api_key=key)

# -------------------------------------------------------
# DASHBOARD PROFESIONAL
# -------------------------------------------------------

def show_dashboard():
    st.header("📊 Dashboard de Propiedades")

    df = st.session_state.df
    cm = st.session_state.column_map

    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    df_dash = df.copy()

    # Columnas mapeadas
    col_precio_desde = cm.get("precio_uf_desde")
    col_precio_hasta = cm.get("precio_uf_hasta")
    col_sup = cm.get("superficie_total_m2")
    col_comuna = cm.get("comuna")
    col_tipo = cm.get("tipo_unidad")
    col_dorms = cm.get("dormitorios")
    col_banos = cm.get("banos")
    col_ano = cm.get("ano_entrega_estimada")

    # Convertir a numérico
    if col_precio_desde:
        df_dash[col_precio_desde] = _ensure_numeric(df_dash, col_precio_desde)
    if col_precio_hasta:
        df_dash[col_precio_hasta] = _ensure_numeric(df_dash, col_precio_hasta)
    if col_sup:
        df_dash[col_sup] = _ensure_numeric(df_dash, col_sup)

    # Precio promedio UF
    if col_precio_desde:
        if col_precio_hasta:
            df_dash["precio_uf_promedio"] = (
                df_dash[col_precio_desde] + df_dash[col_precio_hasta]
            ) / 2
        else:
            df_dash["precio_uf_promedio"] = df_dash[col_precio_desde]
    else:
        st.error("No se encontró columna de precio en UF.")
        return

    # Precio por m2
    if col_sup:
        df_dash["precio_uf_m2"] = df_dash["precio_uf_promedio"] / df_dash[col_sup]
    else:
        df_dash["precio_uf_m2"] = None

    # ------------------------------
    # FILTROS
    # ------------------------------
    with st.expander("🎛️ Filtros del Dashboard", expanded=True):
        uf_min = int(df_dash["precio_uf_promedio"].min())
        uf_max = int(df_dash["precio_uf_promedio"].max())

        rango_uf = st.slider(
            "Rango de precio UF (promedio)",
            min_value=uf_min,
            max_value=uf_max,
            value=(uf_min, uf_max),
        )

        if col_comuna:
            comunas = sorted(df_dash[col_comuna].dropna().unique())
            comunas_sel = st.multiselect("Comunas", comunas, default=comunas)
        else:
            comunas_sel = []

        if col_tipo:
            tipos = sorted(df_dash[col_tipo].dropna().unique())
            tipos_sel = st.multiselect("Tipo Unidad", tipos, default=tipos)
        else:
            tipos_sel = []

        if col_dorms:
            df_dash[col_dorms] = _ensure_numeric(df_dash, col_dorms)
            rango_dorms = st.slider(
                "Dormitorios",
                int(df_dash[col_dorms].min()),
                int(df_dash[col_dorms].max()),
                value=(int(df_dash[col_dorms].min()), int(df_dash[col_dorms].max())),
            )
        else:
            rango_dorms = None

        if col_banos:
            df_dash[col_banos] = _ensure_numeric(df_dash, col_banos)
            rango_banos = st.slider(
                "Baños",
                int(df_dash[col_banos].min()),
                int(df_dash[col_banos].max()),
                value=(int(df_dash[col_banos].min()), int(df_dash[col_banos].max())),
            )
        else:
            rango_banos = None

        if col_ano:
            df_dash[col_ano] = _ensure_numeric(df_dash, col_ano)
            anos = sorted(df_dash[col_ano].dropna().unique())
            anos_sel = st.multiselect("Año entrega estimada", anos, default=anos)
        else:
            anos_sel = []

    # ------------------------------
    # APLICAR FILTROS
    # ------------------------------
    mask = df_dash["precio_uf_promedio"].between(rango_uf[0], rango_uf[1])

    if comunas_sel and col_comuna:
        mask &= df_dash[col_comuna].isin(comunas_sel)

    if tipos_sel and col_tipo:
        mask &= df_dash[col_tipo].isin(tipos_sel)

    if rango_dorms and col_dorms:
        mask &= df_dash[col_dorms].between(rango_dorms[0], rango_dorms[1])

    if rango_banos and col_banos:
        mask &= df_dash[col_banos].between(rango_banos[0], rango_banos[1])

    if anos_sel and col_ano:
        mask &= df_dash[col_ano].isin(anos_sel)

    df_dash = df_dash[mask]

    if df_dash.empty:
        st.warning("⚠️ No hay propiedades con los filtros actuales.")
        return

    # ----------------------------------
    # MÉTRICAS RESUMEN
    # ----------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Propiedades filtradas", len(df_dash))
    col2.metric("UF mínima", int(df_dash["precio_uf_promedio"].min()))
    col3.metric("UF mediana", int(df_dash["precio_uf_promedio"].median()))
    col4.metric("UF máxima", int(df_dash["precio_uf_promedio"].max()))

    st.markdown("---")

    # ----------------------------------
    # GRÁFICO: PRECIO MÍNIMO POR COMUNA
    # ----------------------------------
    if col_comuna and col_precio_desde:
        st.subheader("🏙️ Precio mínimo por comuna (UF)")

        gmin = (
            df_dash.groupby(col_comuna)[col_precio_desde]
            .min()
            .reset_index()
            .rename(columns={col_precio_desde: "precio_min_uf"})
        )

        graf = (
            alt.Chart(gmin)
            .mark_bar()
            .encode(
                x=alt.X("precio_min_uf:Q", title="UF mínima"),
                y=alt.Y(f"{col_comuna}:N", sort="-x", title="Comuna"),
                tooltip=[col_comuna, "precio_min_uf"],
            )
            .properties(height=330)
        )

        st.altair_chart(graf, use_container_width=True)

    st.markdown("---")

    # ----------------------------------
    # HISTOGRAMA DE PRECIOS
    # ----------------------------------
    st.subheader("📦 Distribución de precios UF")

    hist = (
        alt.Chart(df_dash)
        .mark_bar()
        .encode(
            x=alt.X("precio_uf_promedio:Q", bin=alt.Bin(maxbins=30)),
            y="count()",
            tooltip=["count()"],
        )
        .properties(height=280)
    )

    st.altair_chart(hist, use_container_width=True)

    st.markdown("---")

    # ----------------------------------
    # PRECIO PROMEDIO POR TIPO
    # ----------------------------------
    if col_tipo:
        st.subheader("🏘️ Precio promedio UF por tipo de unidad")

        gtipo = (
            df_dash.groupby(col_tipo)["precio_uf_promedio"]
            .mean()
            .reset_index()
        )

        chart_tipo = (
            alt.Chart(gtipo)
            .mark_bar()
            .encode(
                x=alt.X(f"{col_tipo}:N", sort="-y"),
                y=alt.Y("precio_uf_promedio:Q", title="UF promedio"),
                tooltip=[col_tipo, "precio_uf_promedio"],
            )
            .properties(height=260)
        )

        st.altair_chart(chart_tipo, use_container_width=True)

    st.markdown("---")

    # ----------------------------------
    # SCATTER SUPERFICIE VS PRECIO
    # ----------------------------------
    if col_sup:
        st.subheader("📐 Relación Superficie vs UF")

        scatter = (
            alt.Chart(df_dash)
            .mark_circle(size=70, opacity=0.7)
            .encode(
                x=alt.X(f"{col_sup}:Q", title="Superficie total (m²)"),
                y=alt.Y("precio_uf_promedio:Q", title="Precio UF"),
                color=alt.Color(f"{col_comuna}:N", title="Comuna") if col_comuna else alt.value("#2563EB"),
                tooltip=[col_comuna, col_sup, "precio_uf_promedio"],
            )
            .properties(height=300)
        )

        st.altair_chart(scatter, use_container_width=True)

# -------------------------------------------------------
# EXPLORADOR DE PROPIEDADES
# -------------------------------------------------------

def show_explorador():
    st.header("🔎 Explorador de Propiedades")

    df = st.session_state.df
    cm = st.session_state.column_map

    if df.empty:
        st.warning("Carga primero la planilla.")
        return

    df_exp = df.copy()

    col_proy = cm.get("nombre_proyecto")
    col_comuna = cm.get("comuna")
    col_tipo = cm.get("tipo_unidad")
    col_precio_desde = cm.get("precio_uf_desde")
    col_sup = cm.get("superficie_total_m2")
    col_url = cm.get("url_portal")

    # Convertir a numérico
    if col_precio_desde:
        df_exp[col_precio_desde] = _ensure_numeric(df_exp, col_precio_desde)
    if col_sup:
        df_exp[col_sup] = _ensure_numeric(df_exp, col_sup)

    # FILTROS
    with st.expander("Filtros", expanded=True):
        if col_comuna:
            comunas = sorted(df_exp[col_comuna].dropna().unique())
            comunas_sel = st.multiselect("Comunas", comunas, default=comunas)
        else:
            comunas_sel = []

        if col_tipo:
            tipos = sorted(df_exp[col_tipo].dropna().unique())
            tipos_sel = st.multiselect("Tipo unidad", tipos, default=tipos)
        else:
            tipos_sel = []

        if col_precio_desde:
            uf_min = int(df_exp[col_precio_desde].min())
            uf_max = int(df_exp[col_precio_desde].max())
            rango_uf = st.slider("UF", uf_min, uf_max, (uf_min, uf_max))
        else:
            rango_uf = None

    # Aplicar filtros
    mask = pd.Series([True] * len(df_exp))

    if comunas_sel and col_comuna:
        mask &= df_exp[col_comuna].isin(comunas_sel)

    if tipos_sel and col_tipo:
        mask &= df_exp[col_tipo].isin(tipos_sel)

    if rango_uf and col_precio_desde:
        mask &= df_exp[col_precio_desde].between(rango_uf[0], rango_uf[1])

    df_filtrado = df_exp[mask]

    st.subheader(f"Resultados: {len(df_filtrado)} propiedades")
    st.dataframe(df_filtrado, use_container_width=True)

    # Botón exportar CSV filtrado
    if not df_filtrado.empty:
        csv = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV filtrado", csv, "propiedades_filtradas.csv", "text/csv")

# -------------------------------------------------------
# FILTRO DE NOTICIAS INMOBILIARIAS CHILENAS
# -------------------------------------------------------

NEWS_KEYWORDS = [
    "inmobiliario", "departamento", "departamentos",
    "vivienda", "propiedad", "proyecto", "inversión",
    "arriendo", "compra", "venta", "uf", "uf.",
    "hipotecario", "hipotecaria", "crédito",
    "plusvalía", "construcción", "inmobiliaria",
    "edificio", "sucursal inmobiliaria",
]

@st.cache_data(ttl=1800)
def fetch_chile_news(sources):
    """
    Obtiene noticias desde RSS chilenas y filtra SOLO noticias del rubro inmobiliario/hipotecario.
    """
    try:
        import feedparser
    except ImportError:
        st.error("Falta instalar 'feedparser' en requirements.txt")
        return []

    noticias = []

    for url in sources:
        feed = feedparser.parse(url)

        for entry in feed.entries[:30]:
            titulo = getattr(entry, "title", "")
            resumen = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")
            fecha = getattr(entry, "published", "") or getattr(entry, "updated", "")
            fuente = feed.feed.title if "title" in feed.feed else url

            texto = (titulo + " " + resumen).lower()

            # Filtrar solo noticias inmobiliarias
            if any(k in texto for k in NEWS_KEYWORDS):
                noticias.append({
                    "titulo": titulo.strip(),
                    "resumen": resumen.strip(),
                    "link": link.strip(),
                    "fecha": fecha,
                    "fuente": fuente,
                })

    # Ordenar por fecha si es posible
    return noticias


# -------------------------------------------------------
# TASAS SBIF (API REAL)
# -------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_sbif_tasas():
    """
    Obtiene tasas hipotecarias por banco desde la API del SBIF.
    Requiere SBIF_API_KEY en st.secrets.
    """
    api_key = st.secrets.get("SBIF_API_KEY", None)
    if not api_key:
        return None, "Falta SBIF_API_KEY en st.secrets"

    year = datetime.now().year

    url = (
        f"https://api.sbif.cl/api-sbifv3/recursos_api/"
        f"tasashipotecarias/{year}?apikey={api_key}&formato=json"
    )

    try:
        resp = requests.get(url, timeout=15)

        if resp.status_code != 200:
            return None, f"Error SBIF HTTP {resp.status_code}"

        data = resp.json()

        if "TasasHipotecarias" not in data:
            return None, "La API SBIF no devolvió tasas."

        df = pd.DataFrame(data["TasasHipotecarias"])

        # Normalización
        mapping = {
            "Institucion": "Banco",
            "Tipo": "TipoCredito",
            "Tasa": "Tasa",
            "Plazo": "Plazo",
            "Pie": "PieMinimo",
        }

        df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)

        # Convertir tasa % → número
        if "Tasa" in df.columns:
            df["Tasa_float"] = (
                df["Tasa"]
                .astype(str)
                .str.replace("%", "")
                .str.replace(",", ".")
            )
            df["Tasa_float"] = pd.to_numeric(df["Tasa_float"], errors="coerce")

        return df, None

    except Exception as e:
        return None, str(e)


def compute_tasa_promedio(df):
    """Promedio simple de tasa hipotecaria."""
    if df is None or df.empty or "Tasa_float" not in df:
        return None
    serie = df["Tasa_float"].dropna()
    if serie.empty:
        return None
    return float(serie.mean())


# -------------------------------------------------------
# INSIGHTS DEL MERCADO (IA CHILE)
# -------------------------------------------------------

def ia_insights_mercado(noticias, df_tasas, uf_valor):
    """
    IA que analiza:
    - noticias inmobiliarias chilenas,
    - tasas hipotecarias,
    - valor UF,
    - contexto experto,
    y entrega insights profesionales.
    """
    client = get_openai_client()
    if client is None:
        return None

    # Preparar contexto
    titulares = [
        f"- {n['titulo']} ({n['fuente']})"
        for n in noticias[:10]
    ]

    tasas_json = []
    if df_tasas is not None and not df_tasas.empty:
        tasas_json = df_tasas.head(10)[["Banco", "TipoCredito", "Tasa"]].to_dict(orient="records")

    contexto = {
        "noticias_chile": titulares,
        "tasas_hipotecarias": tasas_json,
        "uf_hoy": uf_valor,
    }

    system_prompt = (
        "Eres un analista inmobiliario senior en Chile.\n"
        + CHILE_CONTEXT
        + """
Tu misión:
1. Explicar el estado actual del mercado inmobiliario.
2. Cómo las tasas hipotecarias afectan inversión y primera vivienda.
3. Riesgos reales que un broker debe considerar para no quedar mal con clientes.
4. Oportunidades tácticas por comuna y tipo de unidad.
5. Estrategias de venta profesionales para brokers.

Responde siempre con:
- lenguaje chileno profesional,
- análisis realista,
- sin inventar estadísticas no presentes.
"""
    )

    user_prompt = f"""
CONTEXTO (JSON):
{json.dumps(contexto, ensure_ascii=False)[:11000]}

TAREA:
- Entrega un informe completo en 5 secciones:
  a) Resumen Ejecutivo del mercado.
  b) Impacto de tasas hipotecarias hoy.
  c) Oportunidades de inversión concretas.
  d) Riesgos + comunas a evitar.
  e) Discurso comercial para brokers.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.45,
    )

    return resp.choices[0].message.content


# -------------------------------------------------------
# VISTA PRINCIPAL: NOTICIAS Y TASAS
# -------------------------------------------------------

def show_noticias_tasas():
    st.header("📰 Noticias y Tasas del Mercado")

    # ---- Noticias ----
    noticias = fetch_chile_news(st.session_state.news_sources)

    with st.expander("📰 Noticias Relevantes (Chile - Rubro Inmobiliario)", expanded=True):
        if len(noticias) == 0:
            st.warning("No se encontraron noticias inmobiliarias recientes.")
        else:
            for n in noticias[:12]:
                st.subheader(n["titulo"])
                st.caption(f"{n['fuente']} — {n['fecha']}")
                st.write(n["resumen"])
                st.markdown(f"[Leer más]({n['link']})")
                st.markdown("---")

    # ---- Tasas SBIF ----
    st.subheader("📉 Tasas Hipotecarias (SBIF)")

    df_tasas, error = fetch_sbif_tasas()

    if error:
        st.error(f"No fue posible obtener tasas SBIF: {error}")
    else:
        if df_tasas is None or df_tasas.empty:
            st.warning("SBIF no entregó datos de tasas.")
        else:
            st.dataframe(df_tasas, use_container_width=True)

            # Promedio general
            tasa_avg = compute_tasa_promedio(df_tasas)
            if tasa_avg:
                st.metric("Tasa promedio:", f"{tasa_avg:.2f}%")

    # ---- Insights IA ----
    st.markdown("---")
    st.subheader("🧠 Insights del Mercado (IA)")

    uf_valor, _ = get_uf_value()

    if st.button("Generar insights IA"):
        with st.spinner("Analizando mercado chileno..."):
            insights = ia_insights_mercado(noticias, df_tasas, uf_valor)

        if insights:
            st.write(insights)
        else:
            st.error("No fue posible generar insights.")


# -------------------------------------------------------
# PERFIL DEL CLIENTE → TEXTO PARA IA
# -------------------------------------------------------

def build_client_profile_text():
    cp = st.session_state.client_profile

    txt = f"""
Cliente: {cp["nombre_cliente"]}
Objetivo: {cp["objetivo"]}
Rango UF: {cp["rango_uf"][0]} - {cp["rango_uf"][1]}
Dormitorios mínimos: {cp["dorms_min"]}
Baños mínimos: {cp["banos_min"]}
Superficie: {cp["rango_m2"][0]} - {cp["rango_m2"][1]} m2
Comunas preferidas: {", ".join(cp["comunas"]) if cp["comunas"] else "Sin preferencia"}
Años de entrega: {", ".join(cp["anos"]) if cp["anos"] else "Cualquiera"}
Estados comerciales: {", ".join(cp["estados"]) if cp["estados"] else "Cualquiera"}
Notas extra: {cp["comentarios"]}
"""
    return txt


# -------------------------------------------------------
# IA – Recomendar Propiedades
# -------------------------------------------------------

def prepare_properties_for_ai(df_props, max_items=40):
    """Convierte el DF filtrado en JSON resumido para la IA."""
    cm = st.session_state.column_map
    props = []

    for i, row in df_props.head(max_items).iterrows():
        p = {}
        for key, col in cm.items():
            if col and col in row:
                val = row[col]
                p[key] = None if pd.isna(val) else val
        p["id_interno"] = int(i)
        props.append(p)

    return props


def ia_recomendaciones(client_profile_text, props_list, top_k=5):
    """IA selecciona las mejores propiedades para el cliente."""
    client = get_openai_client()
    if not client:
        return None

    system_prompt = (
        "Eres un asesor inmobiliario senior en Chile.\n"
        + CHILE_CONTEXT
        + """
Tu tarea:

1) Elegir las mejores propiedades para este cliente.
2) Evitar comunas riesgosas según el contexto experto.
3) Explicar POR QUÉ son una buena opción.
4) Dar argumentos comerciales listos para que el broker venda.
5) Entregar una estrategia general de inversión basada en realidad chilena.

FORMATO OBLIGATORIO:
JSON con:
{
 "recomendaciones": [
   {
     "id_interno": <id>,
     "score": <1-10>,
     "motivo_principal": "<texto>",
     "argumentos_venta": ["• punto 1", "• punto 2"]
   }
 ],
 "estrategia_general": "<texto>"
}
"""
    )

    user_prompt = f"""
PERFIL DEL CLIENTE:
{client_profile_text}

PROPIEDADES (JSON):
{json.dumps(props_list, ensure_ascii=False)[:16000]}

TAREA:
- Seleccionar top {top_k} propiedades.
- Devolver EXCLUSIVAMENTE el JSON en el formato solicitado.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    raw = resp.choices[0].message.content

    # Intentar parsear JSON de la IA
    try:
        return json.loads(raw)
    except:
        try:
            s = raw[raw.index("{") : raw.rindex("}") + 1]
            return json.loads(s)
        except:
            st.error("La IA devolvió un formato inesperado:")
            st.code(raw)
            return None


# -------------------------------------------------------
# Vista – Recomendaciones IA
# -------------------------------------------------------

def show_recomendaciones_ia():
    st.header("🤖 Recomendaciones Inteligentes (IA)")

    df = get_filtered_df()  # aplica perfil del cliente
    if df.empty:
        st.warning("No hay propiedades que coincidan con el perfil del cliente.")
        return

    st.write(f"Propiedades consideradas: **{len(df)}**")

    perfil_texto = build_client_profile_text()
    props_list = prepare_properties_for_ai(df)

    if st.button("Generar recomendaciones IA"):
        with st.spinner("Analizando opciones ideales para el cliente..."):
            data = ia_recomendaciones(perfil_texto, props_list, top_k=5)

        if data:
            st.subheader("🏆 Mejores opciones para el cliente")

            for rec in data["recomendaciones"]:
                st.markdown(f"### ⭐ Recomendación {rec['score']}/10 — ID {rec['id_interno']}")
                st.write("**Motivo principal:**", rec["motivo_principal"])
                st.write("**Argumentos de venta:**")
                for arg in rec["argumentos_venta"]:
                    st.write("•", arg)
                st.markdown("---")

            st.subheader("📘 Estrategia general sugerida por la IA")
            st.write(data["estrategia_general"])


# -------------------------------------------------------
# IA CHAT – AGENTE CONVERSACIONAL
# -------------------------------------------------------

def ia_chat(mensaje):
    """Asesor experto conversacional."""
    client = get_openai_client()
    if not client:
        return None

    perfil_texto = build_client_profile_text()
    df = get_filtered_df()
    props_json = prepare_properties_for_ai(df, max_items=25)

    system = (
        "Eres un asesor inmobiliario chileno senior.\n"
        + CHILE_CONTEXT
        + """
Tienes acceso a:
- Perfil del cliente
- Propiedades filtradas
- Conversación previa

Debes responder con lógica chilena, profesional, clara y usando UF/CLP.
"""
    )

    # Construye historial de conversación
    msgs = [
        {"role": "system", "content": system},
        {"role": "assistant",
         "content": "Contexto cargado. Estoy listo para ayudarte como broker experto."},
        {"role": "user",
         "content": f"Contexto inicial:\nPerfil:\n{perfil_texto}\n\nPropiedades:\n{json.dumps(props_json, ensure_ascii=False)[:12000]}"},
    ]

    for m in st.session_state.chat_history:
        msgs.append(m)

    msgs.append({"role": "user", "content": mensaje})

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=msgs,
        temperature=0.45,
    )

    return resp.choices[0].message.content


# -------------------------------------------------------
# Vista – Agente IA (Chat)
# -------------------------------------------------------

def show_agente_ia():
    st.header("💬 Agente IA – Asesor Inmobiliario")

    user_msg = st.text_input("Escribe tu mensaje o pregunta:")

    if st.button("Enviar"):
        if user_msg.strip() == "":
            st.warning("Escribe algo primero.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            with st.spinner("Pensando..."):
                respuesta = ia_chat(user_msg)

            if respuesta:
                st.session_state.chat_history.append({"role": "assistant", "content": respuesta})

    # Mostrar historial
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.markdown(f"**👤 Tú:** {m['content']}")
        else:
            st.markdown(f"**🤖 Asesor IA:** {m['content']}")


# -------------------------------------------------------
# VALOR UF VISUAL EN FOOTER
# -------------------------------------------------------

@st.cache_data(ttl=1800)
def get_uf_value():
    try:
        resp = requests.get(
            "https://mindicador.cl/api/uf",
            timeout=10
        )
        data = resp.json()
        valor = data["serie"][0]["valor"]
        return valor, None
    except Exception as e:
        return None, str(e)


# -------------------------------------------------------
# CONFIGURACIÓN – COLORES Y PERFIL CLIENTE
# -------------------------------------------------------

def show_configuracion():
    st.header("⚙️ Configuración del Sistema")

    st.subheader("🎨 Personalizar Colores")
    theme = st.session_state.theme

    c1, c2, c3 = st.columns(3)
    theme["primary_color"] = c1.color_picker("Color Primario", theme["primary_color"])
    theme["secondary_color"] = c2.color_picker("Sidebar", theme["secondary_color"])
    theme["bg_color"] = c3.color_picker("Fondo General", theme["bg_color"])

    if st.button("Aplicar colores"):
        st.success("Colores actualizados. Recarga la página para ver efecto.")
        inject_custom_css()

    st.markdown("---")

    st.subheader("🧑‍💼 Perfil del Cliente")

    cp = st.session_state.client_profile

    cp["nombre_cliente"] = st.text_input("Nombre Cliente")

    cp["objetivo"] = st.selectbox(
        "Objetivo del Cliente",
        ["Inversión", "Primera Vivienda", "Mejorar vivienda"],
        index=["Inversión", "Primera Vivienda", "Mejorar vivienda"].index(cp["objetivo"])
    )

    cp["rango_uf"] = st.slider(
        "Rango de Presupuesto UF",
        500, 20000,
        value=cp["rango_uf"]
    )

    cp["dorms_min"] = st.number_input("Dormitorios mínimos", 0, 6, cp["dorms_min"])
    cp["banos_min"] = st.number_input("Baños mínimos", 0, 5, cp["banos_min"])

    cp["rango_m2"] = st.slider(
        "Rango superficie total (m2)",
        20, 200,
        value=cp["rango_m2"]
    )

    df = st.session_state.df
    cm = st.session_state.column_map

    if not df.empty:
        col_comuna = cm.get("comuna")
        if col_comuna:
            comunas = sorted(df[col_comuna].dropna().unique())
            cp["comunas"] = st.multiselect("Comunas preferidas", comunas, default=cp["comunas"])

        col_ano = cm.get("ano_entrega_estimada")
        if col_ano:
            anos = sorted(df[col_ano].dropna().unique())
            cp["anos"] = st.multiselect("Años entrega estimada", anos, default=cp["anos"])

        col_est = cm.get("estado_comercial")
        if col_est:
            estados = sorted(df[col_est].dropna().unique())
            cp["estados"] = st.multiselect("Estado Comercial", estados, default=cp["estados"])

    cp["comentarios"] = st.text_area("Comentarios adicionales", cp["comentarios"])

    st.success("Perfil cliente actualizado.")


# -------------------------------------------------------
# EXPORTAR PDF
# -------------------------------------------------------

def export_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    perfil = build_client_profile_text()
    df = get_filtered_df()

    file_path = "/mnt/data/reporte_broker.pdf"

    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 40, "REPORTE BROKER IA")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, "Perfil del Cliente:")
    text_obj = c.beginText(50, height - 100)
    text_obj.textLines(perfil)
    c.drawText(text_obj)

    y = height - 250
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Propiedades Sugeridas:")
    y -= 20

    c.setFont("Helvetica", 9)

    cm = st.session_state.column_map

    for _, row in df.head(12).iterrows():
        nombre = row.get(cm.get("nombre_proyecto"), "N/A")
        comuna = row.get(cm.get("comuna"), "N/A")
        precio = row.get(cm.get("precio_uf_desde"), "N/A")

        c.drawString(50, y, f"- {nombre} | {comuna} | UF {precio}")
        y -= 15
        if y < 100:
            c.showPage()
            y = height - 100

    c.save()
    return file_path


def show_exportar_pdf():
    st.header("📄 Exportar Reporte PDF")

    if st.button("Generar PDF"):
        with st.spinner("Creando archivo..."):
            file_path = export_pdf()

        with open(file_path, "rb") as f:
            st.download_button(
                "Descargar PDF",
                f,
                "reporte_broker.pdf",
                mime="application/pdf"
            )


# -------------------------------------------------------
# FILTRADO GLOBAL PARA TODO EL SISTEMA
# -------------------------------------------------------

def get_filtered_df():
    """
    Devuelve las propiedades que coinciden con el perfil del cliente.
    Se usa en Recomendador IA y Chat IA.
    """
    df = st.session_state.df
    if df.empty:
        return df

    cm = st.session_state.column_map
    cp = st.session_state.client_profile

    col_precio = cm.get("precio_uf_desde")
    col_dorms = cm.get("dormitorios")
    col_banos = cm.get("banos")
    col_sup = cm.get("superficie_total_m2")
    col_comuna = cm.get("comuna")

    df2 = df.copy()

    # Numeric conversions
    if col_precio:
        df2[col_precio] = _ensure_numeric(df2, col_precio)
    if col_dorms:
        df2[col_dorms] = _ensure_numeric(df2, col_dorms)
    if col_banos:
        df2[col_banos] = _ensure_numeric(df2, col_banos)
    if col_sup:
        df2[col_sup] = _ensure_numeric(df2, col_sup)

    # Filtros
    mask = pd.Series([True] * len(df2))

    if col_precio:
        mask &= df2[col_precio].between(cp["rango_uf"][0], cp["rango_uf"][1])

    if col_dorms:
        mask &= df2[col_dorms] >= cp["dorms_min"]

    if col_banos:
        mask &= df2[col_banos] >= cp["banos_min"]

    if col_sup:
        mask &= df2[col_sup].between(cp["rango_m2"][0], cp["rango_m2"][1])

    if cp["comunas"] and col_comuna:
        mask &= df2[col_comuna].isin(cp["comunas"])

    return df2[mask]


# -------------------------------------------------------
# CARGAR PROPIEDADES (sheet)
# -------------------------------------------------------

def show_fuentes_propiedades():
    st.header("📁 Fuente de Propiedades")

    st.write("Carga o usa la planilla de ejemplo:")

    if st.button("Cargar planilla ejemplo"):
        df = get_example_sheet()
        st.session_state.df = df
        st.session_state.column_map = map_columns(df)
        st.success("Planilla ejemplo cargada.")

    file = st.file_uploader("Subir planilla CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.session_state.column_map = map_columns(df)
        st.success("Planilla cargada con éxito.")

    if not st.session_state.df.empty:
        st.subheader("Vista previa:")
        st.dataframe(st.session_state.df.head(20), use_container_width=True)


# -------------------------------------------------------
# ROUTER PRINCIPAL (SIDEBAR)
# -------------------------------------------------------

def main_router():
    menu = st.sidebar.selectbox(
        "Menú",
        [
            "Fuente de propiedades",
            "Dashboard",
            "Perfil del cliente",
            "Explorador de propiedades",
            "Recomendaciones IA",
            "Agente IA",
            "Noticias & Tasas",
            "Exportar PDF",
            "Configuración",
        ]
    )

    if menu == "Fuente de propiedades":
        show_fuentes_propiedades()
    elif menu == "Dashboard":
        show_dashboard()
    elif menu == "Perfil del cliente":
        show_configuracion()
    elif menu == "Explorador de propiedades":
        show_explorador()
    elif menu == "Recomendaciones IA":
        show_recomendaciones_ia()
    elif menu == "Agente IA":
        show_agente_ia()
    elif menu == "Noticias & Tasas":
        show_noticias_tasas()
    elif menu == "Exportar PDF":
        show_exportar_pdf()


# -------------------------------------------------------
# FOOTER – UF EN TIEMPO REAL
# -------------------------------------------------------

def show_footer():
    uf, err = get_uf_value()
    st.markdown("---")
    if uf:
        st.markdown(f"💱 **UF hoy:** {uf:,.2f} CLP")
    else:
        st.markdown("💱 UF no disponible en este momento.")


# -------------------------------------------------------
# RUN APP
# -------------------------------------------------------

def run_app():
    main_router()
    show_footer()


run_app()
