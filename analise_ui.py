# -*- coding: utf-8 -*-
"""Renderização da aba "Análise Textual" do dashboard SocioInova RAG."""

import streamlit as st

import analise_corpus as ac


def _chave_filtros(filtros) -> tuple:
    f = filtros or {}
    return (
        tuple(f.get("ufs", [])), tuple(f.get("ifs", [])), tuple(f.get("anos", [])),
        tuple(f.get("regioes", [])), tuple(f.get("tipos", [])), tuple(f.get("dimensoes", [])),
    )


def _filtros_de_chave(chave) -> dict:
    return {"ufs": list(chave[0]), "ifs": list(chave[1]), "anos": list(chave[2]),
            "regioes": list(chave[3]), "tipos": list(chave[4]), "dimensoes": list(chave[5])}


def _filtros_gerais(resumo) -> dict:
    ufs_all, ifs_all, anos = resumo
    with st.expander("🎛️ Filtros da análise", expanded=False):
        c1, c2 = st.columns(2)
        sel_ufs = c1.multiselect("UFs", ufs_all)
        sel_ifs = c2.multiselect("Institutos", ifs_all)
        c3, c4, _ = st.columns(3)
        sel_anos = c3.multiselect("Anos", anos)
        sel_regioes = c4.multiselect(
            "Regiões", ["Nacional", "NORTE", "NORDESTE", "CENTRO OESTE", "SUDESTE", "SUL"])
        c5, c6 = st.columns(2)
        sel_tipos = c5.multiselect("Tipos de documento", ac.tipos_disponiveis())
        sel_dims = c6.multiselect(
            "Dimensões (Quadro 1.5)", list(ac.EIXOS_ANALISE.keys()),
            help="Restringe o corpus aos documentos que mencionam as dimensões analíticas "
                 "selecionadas (recorte por episteme/diretriz).")
    filtros = {"ufs": sel_ufs, "ifs": sel_ifs, "anos": sel_anos, "regioes": sel_regioes,
               "tipos": sel_tipos, "dimensoes": sel_dims}
    st.session_state["_filtros_analise_atuais"] = filtros
    return filtros


def render_aba(resumo):
    filtros = _filtros_gerais(resumo)

    tab_nuvem, tab_cooc, tab_lda, tab_freq, tab_dims, tab_leitura = st.tabs(
        ["☁️ Nuvem de Palavras", "🕸️ Co-ocorrência", "📚 Tópicos (LDA)",
         "📊 Frequência de Temas", "🎯 Dimensões da Diretriz", "🔎 Leitura de Documentos"])

    with tab_nuvem:
        _render_nuvem(filtros)
    with tab_cooc:
        _render_coocorrencia(filtros)
    with tab_lda:
        _render_lda(filtros)
    with tab_freq:
        _render_frequencia(filtros)
    with tab_dims:
        _render_dimensoes(filtros)
    with tab_leitura:
        _render_leitura(filtros)


# --- Nuvem de palavras ---

@st.cache_data(show_spinner=False)
def _nuvem_cacheada(chave, max_palavras, altura):
    docs = ac.carregar_documentos_analise(_filtros_de_chave(chave))
    if not docs:
        return None
    from wordcloud import WordCloud
    texto = " ".join(d["texto_limpo"] for d in docs)
    wc = WordCloud(
        width=900, height=altura, max_words=max_palavras,
        background_color="white", collocations=False, colormap="viridis",
        stopwords=ac.STOPWORDS_PT,
    ).generate(texto)
    return wc.to_array()


def _render_nuvem(filtros):
    st.subheader("☁️ Nuvem de Palavras")
    st.caption("Frequência dos principais termos no corpus (texto normalizado em português).")

    c1, c2, c3 = st.columns(3)
    max_palavras = c1.slider("Nº de palavras", 100, 800, 300, step=50)
    altura = c2.slider("Altura (px)", 300, 900, 500, step=50)
    top_n = c3.slider("Ranking top-N", 10, 60, 30)

    docs = ac.carregar_documentos_analise(filtros)
    if not docs:
        st.warning("Nenhum documento corresponde aos filtros.")
        return

    matriz = _nuvem_cacheada(_chave_filtros(filtros), max_palavras, altura)
    if matriz is None:
        st.warning("Nenhum dado para a nuvem.")
        return

    import matplotlib.pyplot as plt
    import pandas as pd

    top = ac.top_termos(docs, top_n)
    st.caption(f"{len(docs)} documentos · ranking top-{top_n}: "
               f"{', '.join(t[0] for t in top[:5])}, …")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        fig, ax = plt.subplots(figsize=(9, altura / 100))
        ax.imshow(matriz, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    with col_right:
        df = pd.DataFrame(top, columns=["Termo", "Ocorrências"])
        df["#"] = range(1, len(df) + 1)
        st.dataframe(df, hide_index=True, width="stretch")


# --- Co-ocorrência ---

def _render_coocorrencia(filtros):
    st.subheader("🕸️ Mapa de Co-ocorrência de Palavras")
    st.caption(
        "Grafo dinâmico em que nós = termos e arestas = frequência com que dois termos "
        "aparecem juntos no mesmo documento. Hover revela as contagens; sliders ajustam o grafo.")

    c1, c2, c3 = st.columns(3)
    n_termos = c1.slider("Nº de termos (nós)", 8, 60, 25)
    limiar_aresta = c2.slider("Limiar de co-ocorrência (mín. documentos em comum)", 1, 10, 2)
    layout_seed = c3.slider("Semente do layout (muda o arranjo)", 0, 99, 42)

    docs = ac.carregar_documentos_analise(filtros)
    if not docs:
        st.warning("Nenhum documento corresponde aos filtros.")
        return

    termos = [t for t, _ in ac.top_termos(docs, n_termos)]
    ocorrencias, arestas = ac.coocorrencia(docs, termos, limiar=limiar_aresta)
    grafo = ac.grafo_coocorrencia(ocorrencias, arestas, limiar=limiar_aresta, seed=layout_seed)

    if not grafo or not grafo["edges"]:
        st.info("Sem arestas acima do limiar. Reduza o limiar ou aumente o nº de termos.")
        return

    import plotly.graph_objects as go

    nodes = grafo["nodes"]
    pos = grafo["pos"]
    xv = [pos[n][0] for n in nodes]
    yv = [pos[n][1] for n in nodes]
    arestas_x, arestas_y = [], []
    for a, b, _ in grafo["edges"]:
        arestas_x += [pos[a][0], pos[b][0], None]
        arestas_y += [pos[a][1], pos[b][1], None]

    textos_nos = [f"<b>{n}</b><br>Frequência: {p}" for n, p in zip(nodes, grafo["pesos"])]
    hover_arestas = [f"{a} ↔ {b}<br>Documentos em comum: {p}"
                     for a, b, p in grafo["edges"]]

    trace_arestas = go.Scatter(
        x=arestas_x, y=arestas_y, mode="lines",
        line=dict(width=0.5, color="rgba(136,136,136,0.55)"),
        hoverinfo="text", text=hover_arestas,
    )
    trace_nos = go.Scatter(
        x=xv, y=yv, mode="markers+text",
        marker=dict(
            size=grafo["tamanhos"], color=grafo["pesos"],
            colorscale="Blues", showscale=True,
            colorbar=dict(title="Frequência"), line=dict(width=1, color="#fff"),
        ),
        text=[n[:20] for n in nodes], textposition="top center",
        textfont=dict(size=10, color="#222"), hoverinfo="text", hovertext=textos_nos,
    )

    fig = go.Figure(data=[trace_arestas, trace_nos])
    fig.update_layout(
        height=650, title="Co-ocorrência de termos",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(f"{len(nodes)} nós · {len(grafo['edges'])} arestas. "
               "Termos em azul escuro aparecem em mais documentos.")
    with st.expander("📋 Tabela de co-ocorrências (pares)"):
        import pandas as pd
        df = pd.DataFrame(grafo["edges"], columns=["Termo A", "Termo B", "Documentos em comum"])
        df = df.sort_values("Documentos em comum", ascending=False)
        st.dataframe(df, hide_index=True, width="stretch")


# --- Tópicos LDA ---

@st.cache_data(show_spinner=False)
def _lda_cacheado(chave, n_topics, max_feat):
    docs = ac.carregar_documentos_analise(_filtros_de_chave(chave))
    if not docs:
        return None
    res = ac.lda_topicos(docs, n_topics=n_topics, max_features=max_feat)
    if res is None:
        return None
    return {"matriz": res["matriz_doc_topic"].round(4).tolist(),
            "topicos": res["topicos"]}


def _render_lda(filtros):
    st.subheader("📚 Tópicos Latentes (LDA)")
    st.caption(
        "Modelagem de tópicos com Latent Dirichlet Allocation (scikit-learn), com o texto "
        "em português normalizado. Ajuste o número de tópicos e veja as palavras dominantes.")

    c1, c2 = st.columns(2)
    n_topics = c1.slider("Nº de tópicos", 3, 10, 5)
    max_feat = c2.slider("Vocabulário máximo", 500, 5000, 2000, step=100)

    docs = ac.carregar_documentos_analise(filtros)
    if not docs:
        st.warning("Nenhum documento corresponde aos filtros.")
        return

    with st.spinner("Ajustando LDA..."):
        result = _lda_cacheado(_chave_filtros(filtros), n_topics, max_feat)

    if result is None:
        st.warning("Corpus pequeno demais para LDA. Remova filtros ou adicione documentos.")
        return

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    matriz = np.array(result["matriz"], dtype=float)
    topicos = result["topicos"]
    nomes_topicos = [f"Tópico {i+1}" for i in range(len(topicos))]

    col_tb = st.columns(2)
    for i, tp in enumerate(topicos):
        with col_tb[i % 2]:
            st.markdown(f"**{nomes_topicos[i]}**")
            st.caption("  ·  ".join(w for w, _ in tp))

    st.divider()
    metricas = st.multiselect(
        "Ver distribuição dos tópicos por:", ["UF", "Instituto", "Região", "Ano"],
        default=["UF"])

    df_docs = pd.DataFrame([
        {"arquivo": d["arquivo"], "uf": d["uf"], "instituto": d["instituto"],
         "regiao": d["regiao"], "ano": str(d["ano"]) if d["ano"] else "S/d"} for d in docs
    ])
    for i in range(matriz.shape[1]):
        df_docs[nomes_topicos[i]] = matriz[:, i]
    df_docs["Tópico Dominante"] = df_docs[nomes_topicos].idxmax(axis=1)

    if metricas:
        graficos = st.columns(len(metricas))
        mapa_coluna = {"UF": "uf", "Instituto": "instituto", "Região": "regiao", "Ano": "ano"}
        for col, met in zip(graficos, metricas):
            with col:
                col_met = mapa_coluna[met]
                contagem = (df_docs.groupby([col_met, "Tópico Dominante"])
                            .size().reset_index(name="Documentos")
                            .rename(columns={col_met: met}))
                fig = px.bar(
                    contagem, x=met, y="Documentos", color="Tópico Dominante",
                    barmode="stack", height=400, title=f"Distribuição por {met}")
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, yref="container"))
                st.plotly_chart(fig, width="stretch")

        txt_correlacao = (
            "<details><summary><b>💡 Como ler</b></summary>"
            "Cada barra mostra quantos documentos da categoria (UF/Instituto/Região/Ano) "
            "têm cada tópico como dominante. Tópicos com distribuição desigual indicam "
            "marcos temáticos específicos daquela categoria.</details>")
        st.markdown(txt_correlacao, unsafe_allow_html=True)

    with st.expander("🧬 Distribuição documento → tópico (área empilhada)"):
        xs = list(range(matriz.shape[0]))
        fig = go.Figure()
        for i, nome in enumerate(nomes_topicos):
            fig.add_trace(go.Scatter(
                x=xs, y=matriz[:, i], mode="lines",
                name=nome, stackgroup="one", groupnorm="fraction"))
        fig.update_layout(
            height=400, title=f"Participação de cada documento nos tópicos ({len(docs)} docs)",
            yaxis_title="proporção", xaxis_title="documento",
            margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
        st.plotly_chart(fig, width="stretch")


# --- Frequência de temas ---

@st.cache_data(show_spinner=False)
def _freq_geral_cacheado(chave):
    docs = ac.carregar_documentos_analise(_filtros_de_chave(chave))
    if not docs:
        return None
    return ac.frequencia_temas(docs)


@st.cache_data(show_spinner=False)
def _freq_por_cacheado(chave, categoria):
    docs = ac.carregar_documentos_analise(_filtros_de_chave(chave))
    if not docs:
        return None
    return ac.frequencia_temas_por(docs, categoria)


def _render_frequencia(filtros):
    st.subheader("📊 Frequência de Temas")
    st.caption(
        "Cobertura e profundidade das dimensões analíticas (Quadro 1.5). 'Sinais' = palavras-chave "
        "distintas detectadas; 'Profundidade média' = fração média de sinais presentes por documento "
        "— pondera por sinal da diretriz, não apenas pela frequência bruta de menções.")

    docs = ac.carregar_documentos_analise(filtros)
    if not docs:
        st.warning("Nenhum documento corresponde aos filtros.")
        return

    import pandas as pd
    import plotly.graph_objects as go

    chave = _chave_filtros(filtros)
    df = _freq_geral_cacheado(chave).copy()
    if df is None or df.empty:
        st.warning("Nenhum dado para os eixos.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Documentos analisados", len(docs))
        if not df.empty:
            top = df.loc[df["Documentos"].idxmax()]
            st.metric("Maior cobertura", top["Eixo"],
                      f"{int(top['Documentos'])} docs · {top['Profundidade média (%)']:.0f}% prof.")
    with c2:
        fig1 = go.Figure(go.Bar(
            x=df["Documentos"], y=df["Eixo"], orientation="h",
            marker_color="#2c7fb8", text=df["Documentos"], textposition="outside",
            hovertemplate="%{y}: %{x} documentos (%{customdata}%)<extra></extra>",
            customdata=df["% documentos"]))
        fig1.update_layout(height=340, title="Documentos por dimensão analítica",
                           margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
        st.plotly_chart(fig1, width="stretch")

    quebra = st.radio("Quebrar por:", ["UF", "Região", "Ano"], horizontal=True)
    categoria = {"UF": "uf", "Região": "regiao", "Ano": "ano"}[quebra]
    df_group = _freq_por_cacheado(chave, categoria)
    if df_group is None or df_group.empty:
        st.info("Nenhum dado para essa quebra.")
        return

    fig2 = go.Figure()
    for eixo in ac.EIXOS_ANALISE.keys():
        sub = df_group[df_group["Eixo"] == eixo]
        if sub.empty:
            continue
        fig2.add_trace(go.Bar(x=sub[categoria].astype(str), y=sub["Documentos"], name=eixo))
    if quebra == "Ano":
        anos = sorted({str(d["ano"]) for d in docs if d["ano"]})
        fig2.update_xaxes(categoryorder="array", categoryarray=anos)
    fig2.update_layout(
        barmode="group", height=420, title=f"Frequência de dimensões por {quebra}",
        margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, yref="container"))
    st.plotly_chart(fig2, width="stretch")

    with st.expander("📋 Tabela de frequência"):
        st.dataframe(df, hide_index=True, width="stretch")


# --- Dimensões da diretriz ---

@st.cache_data(show_spinner=False)
def _sinais_todas_cacheado(chave):
    docs = ac.carregar_documentos_analise(_filtros_de_chave(chave))
    if not docs:
        return {}
    return ac.sinais_todas_dimensoes(docs)


def _render_dimensoes(filtros):
    st.subheader("🎯 Dimensões da Diretriz (Quadro 1.5)")
    st.caption(
        "Checagem dos sinais analíticos de cada dimensão da Sociologia da Inovação no corpus filtrado. "
        "Sinais 'ausentes' indicam lacunas analíticas (possível pensamento imitativo ou racionalidade "
        "predominantemente instrumental).")

    docs = ac.carregar_documentos_analise(filtros)
    if not docs:
        st.warning("Nenhum documento corresponde aos filtros.")
        return

    chave = _chave_filtros(filtros)
    df = _freq_geral_cacheado(chave)
    if df is None or df.empty:
        st.info("Sem dados para as dimensões.")
        return

    for eixo, meta in ac.EIXOS_ANALISE.items():
        with st.expander(f"**{eixo}** · {meta['pilar']}", expanded=False):
            st.caption(f"Princípios: {meta['referencia']}")
            st.caption(f"Pergunta: {meta['pergunta']}")
            linha = df[df["Eixo"] == eixo]
            if linha.empty:
                st.write("Nenhum documento do corpus filtrado menciona esta dimensão.")
                continue
            r = linha.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Documentos", int(r["Documentos"]))
            c2.metric("Sinais distintos", int(r["Sinais"]))
            c3.metric("Ocorrências", int(r["Ocorrências"]))
            c4.metric("Profundidade média", f"{r['Profundidade média (%)']:.1f}%")

            todas = _sinais_todas_cacheado(chave)
            sinais_bruto = todas.get(eixo)
            if sinais_bruto is None:
                sinais_bruto = ac.sinais_por_dimensao(docs, eixo)
            sinais = sinais_bruto.sort_values("Documentos", ascending=False)
            st.dataframe(sinais, hide_index=True, width="stretch")

            ausentes = sinais[sinais["Situação"] == "Ausente"]
            if not ausentes.empty:
                nomes = ", ".join(ausentes["Sinal"].tolist()[:12])
                suf = "…" if len(ausentes) > 12 else ""
                st.warning(f"{len(ausentes)} sinais ausentes no corpus filtrado (lacuna analítica): "
                           f"{nomes}{suf}")


# --- Leitura de documentos ---

def _render_leitura(filtros):
    st.subheader("🔎 Leitura de Documentos")
    modo = st.radio("Modo de leitura:", ["Concordância (KWIC)", "Navegador de trechos"],
                    horizontal=True)

    docs = ac.carregar_documentos_analise(filtros)
    if not docs:
        st.warning("Nenhum documento corresponde aos filtros.")
        return

    if modo == "Concordância (KWIC)":
        c1, c2 = st.columns([2, 1])
        termo = c1.text_input("Termo para buscar (aceita acentos):", placeholder="ex: royalties")
        janela = c2.slider("Contexto (chars por lado)", 20, 120, 60, step=10)
        maximo = st.slider("Máx. ocorrências", 20, 200, 80, step=10)

        if termo and termo.strip():
            ocorrencias = ac.kwic(docs, termo, janela=janela, maximo=maximo)
            st.caption(f"{len(ocorrencias)} ocorrências de “{termo.strip()}”")
            for o in ocorrencias:
                st.markdown(f"**[{o['ano']}]** {o['instituto']} ({o['uf']}) — *{o['arquivo']}*")
                st.markdown(
                    f"> …{o['antes'][-janela:]} **{o['termo']}** {o['depois'][:janela]}…")
                st.divider()
        else:
            st.info("Digite um termo para ver suas ocorrências com contexto.")
    else:
        opcoes = {f"[{d['ano']}] {d['instituto']} ({d['uf']})" for d in docs}
        sel = st.selectbox("Escolher documento:", sorted(opcoes))
        doc = next(d for d in docs if f"[{d['ano']}] {d['instituto']} ({d['uf']})" == sel)
        st.caption(doc["arquivo"])
        trechos = [t.strip() for t in doc["texto"].split("\n") if len(t.strip()) > 80]
        q = st.number_input("Trechos por página", 5, 50, 10, step=5)
        if st.button("↕️ Mostrar trechos", type="primary"):
            for i, trecho in enumerate(trechos[:q]):
                st.markdown(f"**Trecho {i+1}** ({doc['instituto']}, {doc['uf']})")
                st.write(trecho)
                st.divider()