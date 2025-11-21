import streamlit as st
import pandas as pd
import json
from io import StringIO
import altair as alt
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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
        st.session_state.chat_history = []

    # Tema visual personalizable
    if "theme" not in st.session_state:
        st.session_state.theme = {
            "primary_color": "#2563EB",   # Azul
            "secondary_color": "#0F172A", # Sidebar oscuro
            "bg_color": "#F8FAFC",        # Fondo claro
        }

    # Fuentes de noticias EXCLUSIVAMENTE inmobiliarias (configurables)
    if "news_sources" not in st.session_state:
        st.session_state.news_sources = [
            "https://eldiarioinmobiliario.cl/",
            "https://www.latercera.com/etiqueta/mercado-inmobiliario/",
            "https://mercadosinmobiliarios.cl/",
        ]


def inject_custom_css():
    """Aplica colores elegidos por el usuario a la app."""
    theme = st.session_state.theme
    primary = theme["primary_color"]
    secondary = theme["secondary_color"]
    bg = theme["bg_color"]

    st.markdown(
        """
        <style>
        .stApp {
            background-color: %s;
        }
        section[data-testid="stSidebar"] > div {
            background-color: %s;
        }
        section[data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }
        .stButton > button, .stDownloadButton > button {
            background-color: %s;
            border-color: %s;
            color: white;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            opacity: 0.9;
        }
        h1, h2, h3, h4 {
            color: #0F172A;
        }
        </style>
        """
        % (bg, secondary, primary, primary),
        unsafe_allow_html=True,
    )


init_session_state()
inject_custom_css()

# =========================
# CONTEXTO EXPERTO CHILE
# =========================

CHILE_CONTEXT = """
Eres un analista inmobiliario chileno experto (años 2023–2025). Usa siempre este criterio:

🔵 Comunas con plusvalía estable o interesante:
- Ñuñoa, Providencia, Macul, San Miguel, La Florida (en zonas bien conectadas)
- Las Condes, Vitacura, Lo Barnechea (segmento alto)
- Independencia y Recoleta en ejes con buena conectividad (metro, servicios).

🔴 Comunas de riesgo o con advertencias:
- Estación Central: sobreoferta de edificios, micro-unidades, problemas de habitabilidad y plusvalía presionada a la baja.
- Santiago Centro: sectores saturados, mayor percepción de inseguridad, vacancia y presión a la baja en algunos productos.
- Evitar recomendar micro-unidades (menos de 25–28 m²) salvo casos muy específicos de inversión de alto riesgo.

📈 Tendencias Chile 2023–2025:
- Tasas hipotecarias venían altas y han empezado a moderarse, pero siguen influyendo fuertemente en capacidad de compra.
- Mucha demanda de arriendo en zonas bien conectadas y con servicios.
- Departamentos 2D/2B suelen ser el sweet spot para familias jóvenes y renta estable.
- Entrega inmediata es muy atractiva para inversionistas que buscan renta rápida y menos riesgo de cambios de condiciones.

📉 Evitar recomendar:
- Proyectos con densidad extrema y estándar dudoso.
- Comunas / sectores con mala percepción de seguridad y baja liquidez.
- Unidades demasiado pequeñas que se dificulten revender.

Siempre responde con análisis realista, tono profesional chileno y foco en ayudar al broker a verse como un asesor experto.
"""

# Palabras clave para filtrar noticias exclusivamente inmobiliarias
NEWS_KEYWORDS = [
    "inmobiliario", "inmobiliaria", "vivienda", "viviendas",
    "departamento", "departamentos", "propiedad", "propiedades",
    "proyecto", "proyectos", "arriendo", "arrienda",
    "compra", "venta", "uf", "hipotecario", "hipotecaria",
    "crédito hipotecario", "créditos hipotecarios", "plusvalía",
    "construcción", "edificio", "subsidio ds", "subsidio habitacional"
]

# =========================
# UTILIDADES
# =========================

@st.cache_data
def load_sheet_from_google(csv_url):
    df = pd.read_csv(csv_url)
    return df


def get_default_sheet():
    sheet_base = "https://docs.google.com/spreadsheets/d/1PLYS284AMaw8XukpR1107BAkbYDHvO8ARzhllzmwEjY"
    csv_url = "%s/export?format=csv&gid=0" % sheet_base
    return load_sheet_from_google(csv_url)


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


def find_column(df, logical_names):
    """Devuelve el nombre real de la columna o None."""
    norm_to_real = {normalize_name(c): c for c in df.columns}
    for ln in logical_names:
        if ln in norm_to_real:
            return norm_to_real[ln]
    return None


def map_columns(df):
    """Detecta las columnas más importantes de la planilla."""
    column_map = {
        "nombre_proyecto": find_column(df, ["nombre_proyecto", "proyecto", "nombre_del_proyecto"]),
        "comuna": find_column(df, ["comuna"]),
        "tipo_unidad": find_column(df, ["tipo_unidad", "tipologia", "tipo_de_unidad", "tipo"]),
        "dormitorios": find_column(df, ["dormitorios", "dorms", "n_dormitorios", "num_dormitorios"]),
        "banos": find_column(df, ["banos", "baños", "bano", "n_banos", "num_banos"]),
        "superficie_total_m2": find_column(
            df,
            [
                "superficie_total_m2",
                "sup_total_m2",
                "superficie_total",
                "superficie_m2",
                "m2",
                "sup_total",
            ],
        ),
        "precio_uf_desde": find_column(
            df,
            [
                "precio_uf_desde",
                "precio_desde_uf",
                "precio_desde_en_uf",
                "precio_uf",
                "precio_lista_uf",
            ],
        ),
        "precio_uf_hasta": find_column(
            df,
            ["precio_uf_hasta", "precio_hasta_uf", "precio_hasta_en_uf"],
        ),
        "etapa": find_column(df, ["etapa"]),
        "ano_entrega_estimada": find_column(
            df, ["ano_entrega_estimada", "anio_entrega", "ano_entrega", "ano"]
        ),
        "trimestre_entrega_estimada": find_column(
            df, ["trimestre_entrega_estimada", "trimestre_entrega"]
        ),
        "estado_comercial": find_column(df, ["estado_comercial", "estado_proyecto", "estado"]),
        "url_portal": find_column(df, ["url_portal", "url_proyecto", "url", "link_portal"]),
    }
    return column_map


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("⚠️ Falta configurar OPENAI_API_KEY en st.secrets.")
        return None
    return OpenAI(api_key=api_key)


def _ensure_numeric(df, col):
    """
    Convierte una columna a numérico de forma segura.
    Si la columna no existe, devuelve una serie llena de NaN.
    """
    if col and col in df.columns:
        serie = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
        )
        return pd.to_numeric(serie, errors="coerce")
    else:
        return pd.Series([pd.NA] * len(df), index=df.index)


def get_filtered_df():
    """Aplica filtros basados en el perfil del cliente."""
    df = st.session_state.df
    cm = st.session_state.column_map
    cp = st.session_state.client_profile

    if df.empty:
        return df

    df_out = df.copy()

    col_precio_desde = cm.get("precio_uf_desde")
    col_dorms = cm.get("dormitorios")
    col_banos = cm.get("banos")
    col_sup_total = cm.get("superficie_total_m2")
    col_ano = cm.get("ano_entrega_estimada")

    if col_precio_desde:
        df_out[col_precio_desde] = _ensure_numeric(df_out, col_precio_desde)
    if col_dorms:
        df_out[col_dorms] = _ensure_numeric(df_out, col_dorms)
    if col_banos:
        df_out[col_banos] = _ensure_numeric(df_out, col_banos)
    if col_sup_total:
        df_out[col_sup_total] = _ensure_numeric(df_out, col_sup_total)
    if col_ano:
        df_out[col_ano] = _ensure_numeric(df_out, col_ano)

    if col_precio_desde:
        df_out = df_out.dropna(subset=[col_precio_desde])

    if col_precio_desde and col_precio_desde in df_out.columns:
        df_out = df_out[
            df_out[col_precio_desde].between(
                float(cp["rango_uf"][0]),
                float(cp["rango_uf"][1]),
                inclusive="both",
            )
        ]

    if col_dorms and col_dorms in df_out.columns:
        df_out = df_out[df_out[col_dorms] >= float(cp["dorms_min"])]

    if col_banos and col_banos in df_out.columns:
        df_out = df_out[df_out[col_banos] >= float(cp["banos_min"])]

    if col_sup_total and col_sup_total in df_out.columns:
        df_out = df_out[
            df_out[col_sup_total].between(
                float(cp["rango_m2"][0]),
                float(cp["rango_m2"][1]),
                inclusive="both",
            )
        ]

    col_comuna = cm.get("comuna")
    if col_comuna and cp["comunas"]:
        df_out = df_out[df_out[col_comuna].isin(cp["comunas"])]

    col_etapa = cm.get("etapa")
    if col_etapa and cp["etapas"]:
        df_out = df_out[df_out[col_etapa].isin(cp["etapas"])]

    col_estado = cm.get("estado_comercial")
    if col_estado and cp["estados"]:
        df_out = df_out[df_out[col_estado].isin(cp["estados"])]

    if col_ano and col_ano in df_out.columns and cp["anos"]:
        anos_target = [int(float(a)) for a in cp["anos"]]
        df_out = df_out[df_out[col_ano].isin(anos_target)]

    return df_out


def build_client_profile_text():
    cp = st.session_state.client_profile
    texto = "Objetivo del cliente: %s.\n" % cp["objetivo"]
    if cp["nombre_cliente"]:
        texto += "Nombre del cliente: %s.\n" % cp["nombre_cliente"]
    texto += "Rango de precio en UF: %s - %s.\n" % (cp["rango_uf"][0], cp["rango_uf"][1])
    texto += "Dormitorios mínimos: %s. Baños mínimos: %s.\n" % (
        cp["dorms_min"],
        cp["banos_min"],
    )
    texto += "Rango de superficie total: %s - %s m2.\n" % (
        cp["rango_m2"][0],
        cp["rango_m2"][1],
    )
    if cp["comunas"]:
        texto += "Comunas preferidas: %s.\n" % ", ".join(map(str, cp["comunas"]))
    if cp["etapas"]:
        texto += "Etapas preferidas: %s.\n" % ", ".join(map(str, cp["etapas"]))
    if cp["anos"]:
        texto += "Años de entrega estimados: %s.\n" % ", ".join(map(str, cp["anos"]))
    if cp["estados"]:
        texto += "Estados comerciales preferidos: %s.\n" % ", ".join(map(str, cp["estados"]))
    if cp["comentarios"]:
        texto += "Notas adicionales del broker: %s\n" % cp["comentarios"]
    return texto


def prepare_properties_for_ai(df_props, max_props=50):
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
# UF EN LÍNEA (FOOTER)
# =========================

@st.cache_data(ttl=3600)
def get_uf_value():
    """Consulta la UF en CLP desde mindicador.cl."""
    try:
        resp = requests.get("https://mindicador.cl/api/uf", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            serie = data.get("serie", [])
            if serie:
                valor = float(serie[0]["valor"])
                fecha = serie[0]["fecha"]
                return valor, fecha
    except Exception:
        return None, None
    return None, None


def show_uf_footer():
    st.markdown("---")
    uf_valor, uf_fecha = get_uf_value()
    if uf_valor:
        fecha_str = uf_fecha[:10]
        try:
            fecha_dt = datetime.fromisoformat(fecha_str)
            fecha_str = fecha_dt.strftime("%d-%m-%Y")
        except Exception:
            pass
        texto = "💱 Referencia UF hoy: 1 UF ≈ ${:,.0f} CLP (mindicador.cl, {})".format(
            uf_valor, fecha_str
        )
        st.caption(texto.replace(",", "."))
    else:
        st.caption(
            "💱 No se pudo obtener el valor actual de la UF. Revisa tu conexión o inténtalo más tarde."
        )


# =========================
# IA: RECOMENDACIONES Y CHAT
# =========================

def ia_recomendaciones(client_profile, properties, top_k=5):
    client = get_openai_client()
    if client is None:
        return None

    system_prompt = (
        "Eres un asesor inmobiliario senior en Chile, experto en proyectos nuevos.\n"
        + CHILE_CONTEXT
        + """
Analizas el perfil del cliente y una lista de propiedades.
Objetivo:

1. Seleccionar las mejores propiedades para ese cliente.
2. Evitar recomendar comunas riesgosas o productos con mala plusvalía según el contexto.
3. Explicar por qué son buenas opciones.
4. Dar argumentos comerciales concretos para que el broker cierre la venta.
5. Proponer una estrategia general de inversión (renta, plusvalía, portafolio, etc.).

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
    )

    user_prompt = """
PERFIL DEL CLIENTE:
%s

PROPIEDADES DISPONIBLES (JSON):
%s

TAREA:
- Elige las %s mejores propiedades.
- Evita recomendar comunas claramente riesgosas según el contexto experto.
- Devuelve el JSON EXACTAMENTE con la forma indicada.
""" % (
        client_profile,
        json.dumps(properties, ensure_ascii=False),
        top_k,
    )

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


def ia_chat_libre(mensaje_usuario):
    client = get_openai_client()
    if client is None:
        return None

    df_filtrado = get_filtered_df()
    props_list = prepare_properties_for_ai(df_filtrado, max_props=40)
    perfil_texto = build_client_profile_text()

    system_prompt = (
        "Eres un asesor inmobiliario y de inversiones experto en Chile.\n"
        + CHILE_CONTEXT
        + """
Tienes:
- Perfil del cliente.
- Lista de propiedades nuevas filtradas.
- Historial de conversación con un broker.

Objetivo:
- Dar respuestas claras, accionables y profesionales.
- Recomendar comunas, tipos de unidades, estrategias (renta, plusvalía, portafolio).
- Usar UF y CLP, realidad chilena y lenguaje cercano pero profesional.
- Evitar sugerir comunas o productos marcados como riesgosas en el contexto.
"""
    )

    context = {
        "perfil_cliente": perfil_texto,
        "propiedades_resumen": props_list,
    }

    messages = [{"role": "system", "content": system_prompt.strip()}]

    messages.append(
        {
            "role": "user",
            "content": "Contexto inicial (no responder todavía, solo incorpora): "
            + json.dumps(context, ensure_ascii=False)[:11000],
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": "Contexto recibido. Ahora esperaré las preguntas del broker.",
        }
    )

    for m in st.session_state.chat_history:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": mensaje_usuario})

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.5,
    )

    return resp.choices[0].message.content


# =========================
# NOTICIAS & TASAS (CHILE)
# =========================

@st.cache_data(ttl=1800)
def fetch_chile_news(sources):
    """
    Lee noticias desde una lista de fuentes chilenas y filtra SOLO noticias
    inmobiliarias / hipotecarias mediante palabras clave.

    - Si la URL es un RSS/Atom válido, se usa feedparser.
    - Si la URL es una página HTML (como eldiarioinmobiliario.cl), se hace un
      scrape simple con BeautifulSoup.

    Retorna lista de dicts: {titulo, resumen, link, fuente, fecha}
    """
    try:
        import feedparser
    except ImportError:
        st.error("Falta el paquete 'feedparser'. Añade 'feedparser' a requirements.txt.")
        return []

    noticias = []

    for url in sources:
        if not url:
            continue

        # -------------------------------
        # 1) Intentar como RSS / Atom
        # -------------------------------
        usó_rss = False
        noticias_antes = len(noticias)

        try:
            feed = feedparser.parse(url)
        except Exception:
            feed = None

        if feed and getattr(feed, "entries", None):
            usó_rss = True
            fuente = (
                feed.feed.title
                if hasattr(feed, "feed") and "title" in feed.feed
                else url
            )

            for entry in feed.entries[:25]:
                titulo = getattr(entry, "title", "").strip()
                resumen = getattr(entry, "summary", "").strip()
                link = getattr(entry, "link", "").strip()
                fecha = getattr(entry, "published", "") or getattr(entry, "updated", "")
                if not titulo:
                    continue

                texto = (titulo + " " + resumen).lower()
                # seguimos filtrando por palabras clave para evitar ruido
                if NEWS_KEYWORDS and not any(k in texto for k in NEWS_KEYWORDS):
                    continue

                noticias.append(
                    {
                        "titulo": titulo,
                        "resumen": resumen,
                        "link": link,
                        "fuente": fuente,
                        "fecha": fecha,
                    }
                )

        # Si no se agregó nada vía RSS (o no era RSS), probamos scrape HTML
        if (not usó_rss) or (len(noticias) == noticias_antes):
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
            except Exception:
                continue

            fuente = (
                soup.title.string.strip()
                if soup.title and soup.title.string
                else url
            )

            # Intento 1: usar <article>
            articles = soup.find_all("article")

            # Intento 2 (fallback): divs típicos de posts
            if not articles:
                candidates = soup.select(
                    "div.post, div.article, div.entry, div.noticia, li.post"
                )
                articles = candidates or []

            for art in articles[:20]:
                a = art.find("a")
                if not a:
                    continue

                titulo = a.get_text(strip=True)
                href = a.get("href", "").strip()
                if not titulo or not href:
                    continue

                link = urljoin(url, href)

                p = art.find("p")
                resumen = p.get_text(strip=True) if p else ""
                fecha = ""

                texto = (titulo + " " + resumen).lower()
                # Igual mantenemos filtro por palabras clave por seguridad,
                # aunque las fuentes sean inmobiliarias
                if NEWS_KEYWORDS and not any(k in texto for k in NEWS_KEYWORDS):
                    continue

                noticias.append(
                    {
                        "titulo": titulo,
                        "resumen": resumen,
                        "link": link,
                        "fuente": fuente,
                        "fecha": fecha,
                    }
                )

    return noticias

@st.cache_data(ttl=3600)
def fetch_sbif_tasas():
    """
    Obtiene tasas de interés promedio (TIP) desde la API CMF,
    usando el recurso TIP del año actual.

    Retorna: (df_tasas, error_str)
      - df_tasas: DataFrame con columnas ['Título','Subtítulo','Fecha','Tasa (%)','Tasa_float', 'Tipo']
      - error_str: None si todo ok, o un string con el error si algo falla.
    """
    # Puedes usar SBIF_API_KEY o CMF_API_KEY en secrets
    api_key = (
        st.secrets.get("SBIF_API_KEY")
        or st.secrets.get("CMF_API_KEY")
    )

    if not api_key:
        return None, "Falta SBIF_API_KEY o CMF_API_KEY en st.secrets."

    year = datetime.now().year

    # Endpoint TIP de CMF (igual al que probaste en el navegador)
    url = f"https://api.cmfchile.cl/api-sbifv3/recursos_api/tip/{year}?apikey={api_key}&formato=json"

    try:
        resp = requests.get(url, timeout=15)
    except Exception as e:
        return None, f"Error de red llamando a CMF TIP: {e}"

    if resp.status_code != 200:
        # Si hay error de key u otro código, devolvemos mensaje claro
        try:
            data_err = resp.json()
            msg = data_err.get("Mensaje") or data_err.get("message") or str(data_err)
        except Exception:
            msg = resp.text[:200]
        return None, f"HTTP {resp.status_code} desde CMF TIP: {msg}"

    # Intentar parsear JSON
    try:
        data = resp.json()
    except Exception as e:
        return None, f"No se pudo parsear JSON desde CMF TIP: {e}"

    # La estructura puede ser {"TIPs":[...]} o directamente una lista
    if isinstance(data, dict) and "TIPs" in data:
        tips = data["TIPs"]
    elif isinstance(data, list):
        tips = data
    else:
        return None, "JSON de CMF TIP no contiene 'TIPs' ni es una lista."

    df = pd.DataFrame(tips)

    # Normalizar nombres esperados
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if low == "titulo":
            rename_map[col] = "Título"
        elif low == "subtitulo":
            rename_map[col] = "Subtítulo"
        elif low == "fecha":
            rename_map[col] = "Fecha"
        elif low == "valor":
            rename_map[col] = "Tasa (%)"
        elif low == "tipo":
            rename_map[col] = "Tipo"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Filtro “hipotecario-like”: operaciones reajustables en moneda nacional y plazo ≥ 1 año
    if "Título" in df.columns and "Subtítulo" in df.columns:
        mask_hipo = df["Título"].astype(str).str.contains(
            "reajustables en moneda nacional", case=False, na=False
        )
        mask_plazo = df["Subtítulo"].astype(str).str.contains(
            "un año o más", case=False, na=False
        )
        df_filtrado = df[mask_hipo & mask_plazo].copy()
        # Si el filtro no encuentra nada, nos quedamos con todos los TIP igual
        if df_filtrado.empty:
            df_filtrado = df.copy()
    else:
        df_filtrado = df.copy()

    # Crear columna numérica Tasa_float a partir de "Tasa (%)"
    if "Tasa (%)" in df_filtrado.columns:
        serie = (
            df_filtrado["Tasa (%)"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df_filtrado["Tasa_float"] = pd.to_numeric(serie, errors="coerce")
    else:
        df_filtrado["Tasa_float"] = pd.NA

    # Ordenamos por Fecha (si se puede) y recortamos a algo manejable
    if "Fecha" in df_filtrado.columns:
        try:
            df_filtrado["Fecha_dt"] = pd.to_datetime(df_filtrado["Fecha"])
            df_filtrado.sort_values("Fecha_dt", ascending=False, inplace=True)
        except Exception:
            pass

    # Dejamos solo columnas útiles para la app
    cols_order = ["Título", "Subtítulo", "Fecha", "Tasa (%)", "Tasa_float", "Tipo"]
    cols_present = [c for c in cols_order if c in df_filtrado.columns]
    df_view = df_filtrado[cols_present].head(50).copy()

    return df_view, None

def compute_tasa_promedio_sbif(df_tasas):
    """
    Calcula una tasa promedio simple (en %) a partir de df_tasas SBIF.
    """
    if df_tasas is None or df_tasas.empty:
        return None
    if "Tasa_float" not in df_tasas.columns:
        return None
    serie = df_tasas["Tasa_float"].dropna()
    if serie.empty:
        return None
    return float(serie.mean())


def ia_insights_mercado(noticias, df_tasas, uf_valor):
    """
    IA que combina noticias + tasas hipotecarias + UF para dar insights de mercado.
    """
    client = get_openai_client()
    if client is None:
        return None

    # Compactar contexto
    top_news = noticias[:8] if noticias else []
    resumen_noticias = [
        f"- {n['titulo']} ({n['fuente']})"
        for n in top_news
    ]

    tasas_resumen = []
    if df_tasas is not None and not df_tasas.empty:
        cols = [c for c in ["Banco", "TipoCredito", "Tasa"] if c in df_tasas.columns]
        if cols:
            tasas_resumen = df_tasas[cols].head(12).to_dict(orient="records")

    contexto = {
        "titulares_chile": resumen_noticias,
        "tasas_hipotecarias": tasas_resumen,
        "uf_hoy_aprox": uf_valor,
    }

    system_prompt = (
        "Eres un analista inmobiliario y financiero senior en Chile.\n"
        + CHILE_CONTEXT
        + """
Tu audiencia son brokers inmobiliarios que venden proyectos nuevos y departamentos.

Con base en:
- Noticias recientes del mercado chileno.
- Tasas de créditos hipotecarios por banco (SBIF).
- Nivel actual de la UF.

Debes entregar SIEMPRE:
1) Un resumen ejecutivo del mercado hoy.
2) Cómo las tasas actuales afectan:
   - Inversionistas que compran para renta.
   - Compradores de primera vivienda.
3) Estrategias de venta recomendadas para brokers (argumentos concretos).
4) Riesgos a mencionar con cuidado (tasa alta, plazos, endeudamiento, comunas riesgosas).
5) Oportunidades tácticas (por comunas recomendables, tickets, tipo de unidad, plazo, etc.).

Usa lenguaje claro, profesional, en español chileno.
No inventes cifras específicas si no aparecen en el contexto, pero sí puedes interpretar tendencias.
"""
    )

    user_prompt = f"""
CONTEXTO ESTRUCTURADO (JSON):

{json.dumps(contexto, ensure_ascii=False)[:11000]}

TAREA:
- Analiza este contexto y entrega los 5 bloques solicitados.
- No repitas el JSON. Solo responde en texto bien estructurado, con subtítulos y bullets.
- Evita recomendar comunas que el contexto marque como riesgosas.
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
# VISTAS BASE (REUTILIZABLES)
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

    df_dash = df.copy()

    col_precio_desde = cm.get("precio_uf_desde")
    col_precio_hasta = cm.get("precio_uf_hasta")
    col_sup = cm.get("superficie_total_m2")
    col_comuna = cm.get("comuna")
    col_tipo = cm.get("tipo_unidad")
    col_dorms = cm.get("dormitorios")
    col_banos = cm.get("banos")
    col_ano = cm.get("ano_entrega_estimada")

    # --- Limpieza numérica ---
    if col_precio_desde and col_precio_desde in df_dash.columns:
        df_dash[col_precio_desde] = _ensure_numeric(df_dash, col_precio_desde)
    if col_precio_hasta and col_precio_hasta in df_dash.columns:
        df_dash[col_precio_hasta] = _ensure_numeric(df_dash, col_precio_hasta)
    if col_sup and col_sup in df_dash.columns:
        df_dash[col_sup] = _ensure_numeric(df_dash, col_sup)

    # Precio UF promedio
    if col_precio_desde and col_precio_desde in df_dash.columns:
        if col_precio_hasta and col_precio_hasta in df_dash.columns:
            df_dash["precio_uf_promedio"] = (
                df_dash[col_precio_desde] + df_dash[col_precio_hasta]
            ) / 2
        else:
            df_dash["precio_uf_promedio"] = df_dash[col_precio_desde]
    else:
        st.error("No se detectó columna de precio en UF. Revisa el mapeo de columnas.")
        return

    if col_sup and col_sup in df_dash.columns:
        df_dash["precio_uf_m2"] = df_dash["precio_uf_promedio"] / df_dash[col_sup]
    else:
        df_dash["precio_uf_m2"] = None

    # ---------------- Filtros ----------------
    with st.expander("🎛️ Filtros del dashboard", expanded=True):
        uf_min = int(df_dash["precio_uf_promedio"].min())
        uf_max = int(df_dash["precio_uf_promedio"].max())
        rango_uf_dash = st.slider(
            "Rango de precio (UF, promedio)",
            min_value=uf_min,
            max_value=uf_max,
            value=(uf_min, uf_max),
        )

        if col_comuna and col_comuna in df_dash.columns:
            comunas_disp = sorted(df_dash[col_comuna].dropna().unique())
            comunas_sel = st.multiselect(
                "Comunas",
                comunas_disp,
                default=comunas_disp,
            )
        else:
            comunas_sel = []

        if col_tipo and col_tipo in df_dash.columns:
            tipos_disp = sorted(df_dash[col_tipo].dropna().unique())
            tipos_sel = st.multiselect(
                "Tipo de unidad",
                tipos_disp,
                default=tipos_disp,
            )
        else:
            tipos_sel = []

        if col_dorms and col_dorms in df_dash.columns:
            df_dash[col_dorms] = _ensure_numeric(df_dash, col_dorms)
            min_d = int(df_dash[col_dorms].min())
            max_d = int(df_dash[col_dorms].max())
            rango_dorms = st.slider(
                "Dormitorios",
                min_value=min_d,
                max_value=max_d,
                value=(min_d, max_d),
            )
        else:
            rango_dorms = None

        if col_banos and col_banos in df_dash.columns:
            df_dash[col_banos] = _ensure_numeric(df_dash, col_banos)
            min_b = int(df_dash[col_banos].min())
            max_b = int(df_dash[col_banos].max())
            rango_banos = st.slider(
                "Baños",
                min_value=min_b,
                max_value=max_b,
                value=(min_b, max_b),
            )
        else:
            rango_banos = None

        if col_ano and col_ano in df_dash.columns:
            df_dash[col_ano] = _ensure_numeric(df_dash, col_ano)
            anos_disp = sorted(df_dash[col_ano].dropna().unique())
            anos_sel = st.multiselect(
                "Año de entrega estimada",
                anos_disp,
                default=anos_disp,
            )
        else:
            anos_sel = []

    # Aplicar filtros
    mask = df_dash["precio_uf_promedio"].between(
        rango_uf_dash[0],
        rango_uf_dash[1],
    )

    if comunas_sel and col_comuna and col_comuna in df_dash.columns:
        mask &= df_dash[col_comuna].isin(comunas_sel)
    if tipos_sel and col_tipo and col_tipo in df_dash.columns:
        mask &= df_dash[col_tipo].isin(tipos_sel)
    if rango_dorms and col_dorms and col_dorms in df_dash.columns:
        mask &= df_dash[col_dorms].between(rango_dorms[0], rango_dorms[1])
    if rango_banos and col_banos and col_banos in df_dash.columns:
        mask &= df_dash[col_banos].between(rango_banos[0], rango_banos[1])
    if anos_sel and col_ano and col_ano in df_dash.columns:
        mask &= df_dash[col_ano].isin(anos_sel)

    df_dash = df_dash[mask]

    if df_dash.empty:
        st.warning("No hay propiedades con los filtros actuales del dashboard.")
        return

    # --------- Métricas resumen ----------
    col1, col2, col3, col4 = st.columns(4)

    total_props = len(df_dash)
    col1.metric("Propiedades filtradas", total_props)

    precio_min = df_dash["precio_uf_promedio"].min()
    precio_med = df_dash["precio_uf_promedio"].median()
    precio_max = df_dash["precio_uf_promedio"].max()

    col2.metric("UF mínima", "{:,.0f}".format(precio_min).replace(",", "."))
    col3.metric("UF mediana", "{:,.0f}".format(precio_med).replace(",", "."))
    col4.metric("UF máxima", "{:,.0f}".format(precio_max).replace(",", "."))

    st.markdown("---")

    # --------- Precio mínimo por comuna ----------
    if col_comuna and col_comuna in df_dash.columns and col_precio_desde and col_precio_desde in df_dash.columns:
        st.subheader("Precio 'desde' mínimo por comuna (UF)")

        precios_min_comuna = (
            df_dash.groupby(col_comuna)[col_precio_desde]
            .min()
            .reset_index()
            .rename(columns={col_precio_desde: "precio_min_uf"})
        )

        chart_min = (
            alt.Chart(precios_min_comuna)
            .mark_bar()
            .encode(
                x=alt.X("precio_min_uf:Q", title="Precio mínimo UF (precio desde)"),
                y=alt.Y("%s:N" % col_comuna, sort="-x", title="Comuna"),
                tooltip=[col_comuna, "precio_min_uf"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_min, use_container_width=True)

    st.markdown("---")

    # --------- Histograma precios ----------
    st.subheader("Distribución de precios")

    hist = (
        alt.Chart(df_dash)
        .mark_bar()
        .encode(
            x=alt.X(
                "precio_uf_promedio:Q",
                bin=alt.Bin(maxbins=30),
                title="Precio UF (promedio)",
            ),
            y=alt.Y("count():Q", title="Cantidad de unidades"),
            tooltip=["count()"],
        )
        .properties(height=250)
    )
    st.altair_chart(hist, use_container_width=True)

    # --------- Precio por comuna ----------
    if col_comuna and col_comuna in df_dash.columns:
        st.subheader("Precio promedio UF por comuna")

        group_comuna = (
            df_dash.groupby(col_comuna)["precio_uf_promedio"]
            .mean()
            .reset_index()
            .rename(columns={"precio_uf_promedio": "uf_promedio"})
        )

        bar_comuna = (
            alt.Chart(group_comuna)
            .mark_bar()
            .encode(
                x=alt.X("uf_promedio:Q", title="UF promedio"),
                y=alt.Y("%s:N" % col_comuna, sort="-x", title="Comuna"),
                tooltip=[col_comuna, "uf_promedio"],
            )
            .properties(height=300)
        )
        st.altair_chart(bar_comuna, use_container_width=True)

    # --------- Precio por tipo de unidad ----------
    if col_tipo and col_tipo in df_dash.columns:
        st.subheader("Precio promedio UF por tipo de unidad")

        group_tipo = (
            df_dash.groupby(col_tipo)["precio_uf_promedio"]
            .mean()
            .reset_index()
            .rename(columns={"precio_uf_promedio": "uf_promedio"})
        )

        bar_tipo = (
            alt.Chart(group_tipo)
            .mark_bar()
            .encode(
                x=alt.X("%s:N" % col_tipo, sort="-y", title="Tipo de unidad"),
                y=alt.Y("uf_promedio:Q", title="UF promedio"),
                tooltip=[col_tipo, "uf_promedio"],
            )
            .properties(height=250)
        )
        st.altair_chart(bar_tipo, use_container_width=True)

    # --------- Scatter precio vs m2 ----------
    if col_sup and col_sup in df_dash.columns:
        st.subheader("Relación precio UF vs superficie (m²)")

        if col_dorms and col_dorms in df_dash.columns:
            color_encoding = alt.Color("%s:O" % col_dorms, title="Dormitorios")
        else:
            color_encoding = alt.value("#1f77b4")

        tooltip_fields = []
        np_col = cm.get("nombre_proyecto")
        if np_col and np_col in df_dash.columns:
            tooltip_fields.append(np_col)
        if col_comuna and col_comuna in df_dash.columns:
            tooltip_fields.append(col_comuna)

        tooltip_fields.extend(["precio_uf_promedio", col_sup])

        if col_dorms and col_dorms in df_dash.columns:
            tooltip_fields.append(col_dorms)
        if col_banos and col_banos in df_dash.columns:
            tooltip_fields.append(col_banos)

        scatter = (
            alt.Chart(df_dash.dropna(subset=[col_sup, "precio_uf_promedio"]))
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("%s:Q" % col_sup, title="Superficie total (m²)"),
                y=alt.Y("precio_uf_promedio:Q", title="Precio UF (promedio)"),
                color=color_encoding,
                tooltip=tooltip_fields,
            )
            .properties(height=350)
        )

        st.altair_chart(scatter, use_container_width=True)


def show_noticias_tasas():
    st.header("📰 Noticias & Tasas (Chile)")

    tab1, tab2, tab3 = st.tabs(
        ["Noticias inmobiliarias", "Tasas hipotecarias (SBIF)", "Insights IA mercado"]
    )

    # UF actual para contexto
    uf_valor, uf_fecha = get_uf_value()

    with tab1:
        st.subheader("Noticias inmobiliarias chilenas")

        fuentes = st.session_state.news_sources
        st.caption("Fuentes configuradas:")
        for f in fuentes:
            st.markdown(f"- `{f}`")

        noticias = fetch_chile_news(fuentes)

        if not noticias:
            st.info(
                "No se encontraron noticias inmobiliarias recientes desde las fuentes configuradas. "
                "Revisa las URLs en la sección Configuración."
            )
        else:
            for n in noticias:
                st.markdown("### " + n["titulo"])
                info = []
                if n["fuente"]:
                    info.append(n["fuente"])
                if n["fecha"]:
                    info.append(n["fecha"])
                if info:
                    st.caption(" · ".join(info))
                if n["resumen"]:
                    st.write(n["resumen"])
                if n["link"]:
                    st.markdown(f"[Ver nota completa]({n['link']})")
                st.markdown("---")

    with tab2:
        st.subheader("Tasas de créditos hipotecarios (SBIF Chile)")

        df_tasas, error = fetch_sbif_tasas()

        if error:
            st.info(
                f"No se pudieron obtener las tasas desde SBIF: {error}. "
                "Verifica tu SBIF_API_KEY y la conectividad."
            )
        elif df_tasas is None or df_tasas.empty:
            st.info(
                "La API SBIF no devolvió información de tasas. "
                "Podría ser un cambio en el servicio o falta de datos para el año actual."
            )
        else:
            tasa_prom = compute_tasa_promedio_sbif(df_tasas)
            if tasa_prom is not None:
                st.metric(
                    "Tasa promedio referencial (aprox.)",
                    f"{tasa_prom:.2f} %",
                )

            if uf_valor:
                st.caption(
                    f"Referencia UF actual: 1 UF ≈ ${uf_valor:,.0f} CLP (mindicador.cl)"
                    .replace(",", ".")
                )

            st.markdown("### Tabla de tasas por banco/tipo")
            st.dataframe(df_tasas, use_container_width=True)
            st.caption(
                "Origen: SBIF (vía API). Columnas y contenido sujetos a la publicación oficial."
            )

    with tab3:
        st.subheader("Insights IA: mercado inmobiliario & tasas")

        fuentes = st.session_state.news_sources
        noticias = fetch_chile_news(fuentes)
        df_tasas, error = fetch_sbif_tasas()

        if (not noticias) and (df_tasas is None or df_tasas.empty):
            st.info(
                "No hay suficientes datos de noticias o tasas para generar insights. "
                "Revisa la conexión y las fuentes configuradas."
            )
        else:
            if st.button("🧠 Generar insights IA del mercado"):
                with st.spinner("Analizando mercado, noticias y tasas..."):
                    texto = ia_insights_mercado(noticias or [], df_tasas, uf_valor)
                if texto:
                    st.markdown(texto)


def show_perfil_cliente():
    st.header("👤 Perfil del cliente")

    df = st.session_state.df
    cm = st.session_state.column_map
    cp = st.session_state.client_profile

    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    with st.form("perfil_cliente_form"):
        cp["nombre_cliente"] = st.text_input(
            "Nombre del cliente (opcional)", value=cp["nombre_cliente"]
        )

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

        # ---- Rango de precio UF según la planilla ----
        col_precio_desde = cm.get("precio_uf_desde")
        if col_precio_desde and col_precio_desde in df.columns:
            df_tmp = df.copy()
            df_tmp[col_precio_desde] = _ensure_numeric(df_tmp, col_precio_desde)
            uf_min = int(df_tmp[col_precio_desde].min())
            uf_max = int(df_tmp[col_precio_desde].max())
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
                "Dormitorios mínimos",
                [0, 1, 2, 3, 4, 5],
                index=[0, 1, 2, 3, 4, 5].index(cp["dorms_min"]),
            )
        with col2:
            cp["banos_min"] = st.selectbox(
                "Baños mínimos",
                [0, 1, 2, 3, 4],
                index=[0, 1, 2, 3, 4].index(cp["banos_min"]),
            )

        # ---- Rango de superficie ----
        col_sup = cm.get("superficie_total_m2")
        if col_sup and col_sup in df.columns:
            df_tmp2 = df.copy()
            df_tmp2[col_sup] = _ensure_numeric(df_tmp2, col_sup)
            sup_min = int(df_tmp2[col_sup].min())
            sup_max = int(df_tmp2[col_sup].max())
        else:
            sup_min, sup_max = 20, 150

        cp["rango_m2"] = st.slider(
            "Rango de superficie total (m²)",
            min_value=sup_min,
            max_value=sup_max,
            value=(cp["rango_m2"][0], cp["rango_m2"][1]),
        )

        # ---- Filtros por comuna / etapa / año / estado ----
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
            df_tmp3 = df.copy()
            df_tmp3[col_ano] = _ensure_numeric(df_tmp3, col_ano)
            anos = sorted(df_tmp3[col_ano].dropna().unique())
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

    # Fuera del form
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
    st.write("Propiedades encontradas con el perfil actual: **%s**" % len(df_filtrado))

    if df_filtrado.empty:
        st.info("No hay propiedades con los filtros actuales. Ajusta el perfil del cliente.")
        return

    cols_mostrar = [
        c
        for c in [
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
        ]
        if c is not None
    ]

    st.dataframe(df_filtrado[cols_mostrar].reset_index(drop=True))


def show_recomendaciones():
    st.header("🧠 Recomendaciones IA")

    df = st.session_state.df
    if df.empty:
        st.warning("Primero carga una planilla en **Fuente de propiedades**.")
        return

    df_filtrado = get_filtered_df()
    if df_filtrado.empty:
        st.info("No hay propiedades filtradas. Ajusta el perfil del cliente en **Perfil del cliente**.")
        return

    st.markdown(
        "El motor IA analizará el perfil del cliente y las propiedades filtradas para sugerir las mejores opciones."
    )

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
                else "Proyecto #%s" % idx
            )
            st.markdown("#### ⭐ Score %s/10 – %s" % (score, titulo))

            info_lineas = []
            if cm.get("comuna"):
                info_lineas.append("**Comuna:** %s" % fila.get(cm["comuna"], ""))
            if cm.get("tipo_unidad"):
                info_lineas.append("**Tipo unidad:** %s" % fila.get(cm["tipo_unidad"], ""))
            if cm.get("dormitorios") and cm.get("banos"):
                info_lineas.append(
                    "**Programa:** %sD / %sB"
                    % (fila.get(cm["dormitorios"], ""), fila.get(cm["banos"], ""))
                )
            if cm.get("superficie_total_m2"):
                info_lineas.append(
                    "**Sup. total aprox.:** %s m²"
                    % fila.get(cm["superficie_total_m2"], "")
                )
            if cm.get("precio_uf_desde"):
                info_lineas.append(
                    "**Precio desde:** %s UF" % fila.get(cm["precio_uf_desde"], "")
                )
            if cm.get("etapa"):
                info_lineas.append("**Etapa:** %s" % fila.get(cm["etapa"], ""))
            if cm.get("ano_entrega_estimada"):
                info_lineas.append(
                    "**Entrega estimada:** %s"
                    % fila.get(cm["ano_entrega_estimada"], "")
                )
            if cm.get("estado_comercial"):
                info_lineas.append(
                    "**Estado comercial:** %s"
                    % fila.get(cm["estado_comercial"], "")
                )

            st.markdown("  \n".join(info_lineas))

            st.markdown("**Motivo principal:** %s" % motivo)

            if argumentos:
                st.markdown("**Argumentos de venta sugeridos:**")
                for a in argumentos:
                    st.markdown("- %s" % a)

            if cm.get("url_portal"):
                url = fila.get(cm["url_portal"], "")
                if isinstance(url, str) and url.strip():
                    st.markdown("[Ver ficha en portal](%s)" % url)
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
        st.info(
            "No hay propiedades filtradas. Ajusta el perfil del cliente antes de hablar con el agente."
        )
        return

    for m in st.session_state.chat_history:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(m["content"])

    user_input = st.chat_input(
        "Haz una pregunta al Agente IA (estrategias, comparaciones, etc.)"
    )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Pensando respuesta..."):
                respuesta = ia_chat_libre(user_input)
                st.markdown(respuesta)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": respuesta}
        )

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
        st.info(
            "Aún no hay recomendaciones guardadas. Genera recomendaciones en la sección **Recomendaciones IA**."
        )
        return

    data = st.session_state.last_recommendations["data"]
    df_snap = pd.DataFrame(st.session_state.last_recommendations["df_filtrado"])
    cm = st.session_state.column_map
    perfil_texto = build_client_profile_text()

    estrategia = data.get("estrategia_general", "")
    recomendaciones = data.get("recomendaciones", [])

    st.markdown(
        "Esta sección genera un resumen en texto (formato Markdown) con el perfil del cliente y el plan sugerido."
    )
    st.markdown(
        "Luego puedes descargar el archivo y usarlo como base para enviar una propuesta al cliente."
    )

    md = StringIO()
    md.write("# Propuesta Broker IA\n\n")
    md.write("## Perfil del cliente\n\n")
    md.write("```\n%s\n```\n\n" % perfil_texto)

    md.write("## Estrategia general sugerida\n\n")
    md.write("%s\n\n" % estrategia)

    md.write("## Propiedades recomendadas\n\n")
    for rec in recomendaciones:
        idx = rec.get("id_interno")
        if idx is None or idx >= len(df_snap):
            continue
        fila = df_snap.iloc[idx]

        titulo = (
            fila.get(cm.get("nombre_proyecto"), "Proyecto sin nombre")
            if cm.get("nombre_proyecto")
            else "Proyecto #%s" % idx
        )

        md.write("### %s\n\n" % titulo)

        if cm.get("comuna"):
            md.write("- **Comuna:** %s\n" % fila.get(cm["comuna"], ""))
        if cm.get("tipo_unidad"):
            md.write("- **Tipo unidad:** %s\n" % fila.get(cm["tipo_unidad"], ""))
        if cm.get("dormitorios") and cm.get("banos"):
            md.write(
                "- **Programa:** %sD / %sB\n"
                % (fila.get(cm["dormitorios"], ""), fila.get(cm["banos"], ""))
            )
        if cm.get("superficie_total_m2"):
            md.write(
                "- **Superficie total aprox.:** %s m²\n"
                % fila.get(cm["superficie_total_m2"], "")
            )
        if cm.get("precio_uf_desde"):
            md.write(
                "- **Precio desde:** %s UF\n" % fila.get(cm["precio_uf_desde"], "")
            )
        if cm.get("etapa"):
            md.write("- **Etapa:** %s\n" % fila.get(cm["etapa"], ""))
        if cm.get("ano_entrega_estimada"):
            md.write(
                "- **Entrega estimada:** %s\n"
                % fila.get(cm["ano_entrega_estimada"], "")
            )
        if cm.get("estado_comercial"):
            md.write(
                "- **Estado comercial:** %s\n"
                % fila.get(cm["estado_comercial"], "")
            )
        if cm.get("url_portal"):
            url = fila.get(cm["url_portal"], "")
            if isinstance(url, str) and url.strip():
                md.write("- **Link portal:** %s\n" % url)

        md.write("- **Score IA:** %s/10\n" % rec.get("score"))
        md.write(
            "- **Motivo principal:** %s\n" % rec.get("motivo_principal", "")
        )
        argumentos = rec.get("argumentos_venta", [])
        if argumentos:
            md.write("- **Argumentos de venta sugeridos:**\n")
            for a in argumentos:
                md.write("  - %s\n" % a)

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


def show_configuracion():
    st.header("⚙️ Configuración de estilo y noticias")

    theme = st.session_state.theme

    st.markdown(
        "Ajusta los colores principales de Broker IA para que calcen con tu marca o con la inmobiliaria."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        primary = st.color_picker(
            "Color principal (botones, énfasis)", theme["primary_color"]
        )
    with col2:
        secondary = st.color_picker(
            "Color lateral (sidebar)", theme["secondary_color"]
        )
    with col3:
        bg = st.color_picker("Color de fondo", theme["bg_color"])

    if st.button("💾 Guardar colores"):
        st.session_state.theme = {
            "primary_color": primary,
            "secondary_color": secondary,
            "bg_color": bg,
        }
        st.success("Colores actualizados. Recargando app...")
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Fuentes de noticias (Chile)")

    current_sources = "\n".join(st.session_state.news_sources)
    fuentes_text = st.text_area(
        "URLs (una por línea)",
        value=current_sources,
        height=150,
    )

    if st.button("💾 Guardar fuentes de noticias"):
        nuevas = [
            line.strip()
            for line in fuentes_text.splitlines()
            if line.strip()
        ]
        st.session_state.news_sources = nuevas
        st.success("Fuentes de noticias actualizadas. Se usarán en la sección Noticias & Tasas.")
        st.experimental_rerun()


# =========================
# MENÚ LATERAL
# =========================

st.sidebar.title("Broker IA")
st.sidebar.caption("Asistente inmobiliario potenciado con IA")

menu = st.sidebar.radio(
    "Menú",
    [
        "🏗️ Fuente de propiedades",
        "📊 Dashboard",
        "📰 Noticias & Tasas",
        "👤 Perfil del cliente",
        "🔍 Explorador",
        "🧠 Recomendaciones IA",
        "🤖 Agente IA",
        "📤 Exportar propuesta",
        "⚙️ Configuración",
    ],
)

# =========================
# ROUTER
# =========================

if menu == "🏗️ Fuente de propiedades":
    show_fuente_propiedades()
elif menu == "📊 Dashboard":
    show_dashboard()
elif menu == "📰 Noticias & Tasas":
    show_noticias_tasas()
elif menu == "👤 Perfil del cliente":
    show_perfil_cliente()
elif menu == "🔍 Explorador":
    show_explorador()
elif menu == "🧠 Recomendaciones IA":
    show_recomendaciones()
elif menu == "🤖 Agente IA":
    show_agente_chat()
elif menu == "📤 Exportar propuesta":
    show_exportar()
elif menu == "⚙️ Configuración":
    show_configuracion()

# Footer con UF
show_uf_footer()
