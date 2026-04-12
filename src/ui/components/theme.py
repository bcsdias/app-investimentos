import streamlit as st

def init_theme():
    """Inicializa o estado do tema na sessão."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

def get_theme_css():
    """Retorna o CSS baseado no tema atual."""
    is_dark = st.session_state.get("theme", "dark") == "dark"
    
    if is_dark:
        bg_color = "#0E1117"
        card_bg = "#161B22"
        text_color = "#E6EDF3"
        secondary_text = "#8B949E"
        accent_color = "#58A6FF"
        border_color = "#30363D"
        metric_bg = "rgba(88, 166, 255, 0.1)"
    else:
        bg_color = "#F0F2F6"
        card_bg = "#FFFFFF"
        text_color = "#1F2937"
        secondary_text = "#4B5563"
        accent_color = "#0068C9"
        border_color = "#E5E7EB"
        metric_bg = "rgba(0, 104, 201, 0.05)"

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600&display=swap');

        /* Global Styles */
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', sans-serif;
            background-color: {bg_color};
            color: {text_color};
        }}

        /* Headers */
        h1, h2, h3, [data-testid="stHeader"] {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }}

        /* Cards and Containers */
        div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid"] {{
            background-color: {card_bg};
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}

        /* Metrics Customization */
        div[data-testid="stMetric"] {{
            background-color: {metric_bg};
            border: 1px solid {accent_color}22;
            border-radius: 10px;
            padding: 15px !important;
            transition: transform 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px {accent_color}11;
        }}
        div[data-testid="stMetricLabel"] > div > p {{
            color: {secondary_text} !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        div[data-testid="stMetricValue"] > div {{
            color: {text_color} !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }}

        /* Tab Customization */
        button[data-baseweb="tab"] {{
            font-weight: 600 !important;
            font-size: 1rem !important;
        }}

        /* Sidebar Cleanup */
        [data-testid="stSidebar"] {{
            background-color: {card_bg};
            border-right: 1px solid {border_color};
        }}

        /* Global Button Style */
        div.stButton > button {{
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }}
        
        /* Hide decoration bar */
        div[data-testid="stDecoration"] {{
            display: none;
        }}
    </style>
    """
    return css

def render_theme_toggle():
    """Renderiza o botão de alternância de tema no sidebar."""
    with st.sidebar:
        st.markdown("---")
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            st.write(f"🌓 Tema: **{st.session_state.theme.upper()}**")
        with cols[1]:
            if st.button("🔄", key="theme_toggle_btn", help="Alternar entre Light e Dark"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()

def apply_theme():
    """Aplica o CSS do tema na página atual."""
    init_theme()
    st.markdown(get_theme_css(), unsafe_allow_html=True)
