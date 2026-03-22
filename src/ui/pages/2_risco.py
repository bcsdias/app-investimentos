import streamlit as st

st.title("⚖️ Risco e Retorno")

st.markdown("Analise o risco inerente à sua carteira e compare com os benchmarks.")

tab1, tab2, tab3 = st.tabs(["Volatilidade Móvel", "Queda Máxima (Drawdown)", "Risco x Retorno (Scatter)"])

with tab1:
    st.markdown("### Volatilidade Móvel")
    st.caption("A volatilidade mede o grau de oscilação dos rendimentos de um ativo (desvio padrão anualizado). Uma volatilidade maior indica maior risco e instabilidade.")
    st.info("Carregue os dados na configuração.")

with tab2:
    st.markdown("### Queda Máxima (Drawdown)")
    st.caption("O Drawdown mede a queda percentual do preço de um ativo desde o seu último pico máximo até o momento atual ou até o próximo fundo. Ele ilustra o pior cenário de perda que um investidor poderia ter sofrido.")
    
with tab3:
    st.markdown("### Matriz Risco x Retorno")
    st.caption("Compara a rentabilidade histórica anualizada com a volatilidade. O ideal é buscar ativos/carteiras no quadrante superior esquerdo (Alto Retorno, Baixo Risco).")
