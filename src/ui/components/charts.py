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
    
    tooltip_list = [alt.Tooltip(x_col, type='temporal', title='Data', format='%d/%m/%Y')]
    for col in df.columns:
        if col == x_col: continue
        tooltip_list.append(alt.Tooltip(col, type='quantitative', format=y_format))

    nearest = alt.selection_point(nearest=True, on='mouseover', fields=[x_col], empty=False)

    lines = alt.Chart(df_melt).mark_line(point=False).encode(
        x=alt.X(f'{x_col}:T', title='Data'),
        y=alt.Y('Valor:Q', title=y_title, axis=alt.Axis(format=y_format)),
        color=alt.Color('Ativo:N', sort=legend_sort, legend=alt.Legend(title="Ativo"))
    )

    points = lines.mark_circle().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    rule = alt.Chart(df).mark_rule(color='gray').encode(
        x=f'{x_col}:T',
        opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
        tooltip=tooltip_list
    ).add_params(nearest)
    
    chart = alt.layer(lines, points, rule).properties(title=title, height=400).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    with st.expander(f"🔍 Ver dados: {title}"):
        st.dataframe(df_display, use_container_width=True)
