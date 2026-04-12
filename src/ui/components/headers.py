import streamlit as st

def render_page_header(title: str, icon: str, description: str = None, metrics: list = []):
    """
    Renderiza um cabeçalho padronizado com título, ícone, descrição e barramento de métricas.
    :param metrics: Lista de dicionários [{'label': 'Rentabilidade', 'value': '12.5%', 'delta': '+0.5%'}, ...]
    """
    st.markdown(f"## {icon} {title}")
    if description:
        st.caption(description)
    
    if metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            cols[i].metric(
                label=m['label'], 
                value=m['value'], 
                delta=m.get('delta'),
                delta_color=m.get('delta_color', 'normal')
            )
    st.markdown("---")
