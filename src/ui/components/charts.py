import streamlit as st
import altair as alt
import pandas as pd

def render_altair_line(df: pd.DataFrame, title: str, y_format: str = ".0%", y_title: str = "Valor"):
    if df is None or df.empty: return
    
    # Ordenação por Rentabilidade Final (Maior para Menor)
    last_vals = df.ffill().iloc[-1]
    sorted_cols = last_vals.sort_values(ascending=False).index
    df = df[sorted_cols]

    df_display = df.copy()
    
    # OTIMIZAÇÃO: Downsampling se > 800
    MAX_POINTS = 800
    if len(df) > MAX_POINTS:
        step = len(df) // MAX_POINTS
        df_resampled = df.iloc[::step]
        if df.index[-1] != df_resampled.index[-1]:
            df_resampled = pd.concat([df_resampled, df.iloc[[-1]]])
        df = df_resampled

    if df.index.name is None: df.index.name = 'Data'
    df = df.reset_index()
    
    safe_cols = [str(c).replace('.', '_') for c in df.columns]
    df.columns = safe_cols
    x_col = safe_cols[0] 
    
    legend_sort = [c for c in safe_cols if c != x_col]
    df_melt = df.melt(id_vars=[x_col], var_name='Ativo', value_name='Valor')
    
    # Tema de cores
    is_dark = st.session_state.get("theme", "dark") == "dark"
    color_range = ['#58A6FF', '#7EE787', '#FFA166', '#CB71D7', '#F1E05A', '#FF7B72', '#D29922'] if is_dark else \
                  ['#0068C9', '#2E8540', '#E36209', '#6F42C1', '#B08800', '#D73A49', '#24292E']

    tooltip_list = [alt.Tooltip(x_col, type='temporal', title='Data', format='%d/%m/%Y')]
    for col in df.columns:
        if col == x_col: continue
        tooltip_list.append(alt.Tooltip(col, type='quantitative', format=y_format))

    nearest = alt.selection_point(nearest=True, on='mouseover', fields=[x_col], empty=False)

    base = alt.Chart(df_melt).encode(
        x=alt.X(f'{x_col}:T', title=None, axis=alt.Axis(grid=False)),
        y=alt.Y('Valor:Q', title=y_title, axis=alt.Axis(format=y_format, grid=True, gridDash=[2,2])),
        color=alt.Color('Ativo:N', scale=alt.Scale(range=color_range), sort=legend_sort, legend=alt.Legend(title=None, orient='top-left', offset=10))
    )

    lines = base.mark_line(strokeWidth=2.5, interpolate='monotone').encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0.9))
    )

    points = base.mark_circle(size=60).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    rule = alt.Chart(df).mark_rule(color='#8B949E' if is_dark else '#D1D5DB').encode(
        x=f'{x_col}:T',
        opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
        tooltip=tooltip_list
    ).add_params(nearest)
    
    chart = alt.layer(lines, points, rule).properties(
        title=alt.TitleParams(text=title, anchor='start', fontSize=18, fontWeight=600, dy=-10),
        height=450
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    with st.expander(f"📊 Dados Brutos: {title}"):
        st.dataframe(df_display, use_container_width=True)
