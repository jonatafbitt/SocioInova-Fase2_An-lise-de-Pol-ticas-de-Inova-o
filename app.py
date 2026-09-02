import streamlit as st
import os
import json
import shutil
import subprocess  # Adicionado para listar modelos Ollama
import pandas as pd
from datetime import datetime
from fpdf import FPDF
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import analise_corpus as ac
import analise_ui

# Caminho base para dados
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "documentos_inovação"
PERSIST_DIR = BASE_DIR / "memoria_longo_prazo"

# --- 1. CONFIGURAÇÕES, MEMÓRIA E BACKUP ---
st.set_page_config(page_title="SocioInova RAG - UFBA", layout="wide")
DB_DIR = "./memoria_longo_prazo"
PASTA_BASE = "documentos_inovação"
ARQUIVO_MATRIZ = "matriz_extracao_tese.csv"
LLM_PADRAO = "phi4-mini:latest"
MAX_TOKENS_RESPOSTA = 1200
MODELOS_EMBEDDING = {"nomic-embed-text", "all-minilm", "bge-m3", "mxbai-embed-large", "snowflake-arctic-embed"}
MODELOS_PRIORIDADE = [LLM_PADRAO, "gemma3:4b", "qwen2:1.5b", "phi3:latest"]
MODELOS_FALLBACK = MODELOS_PRIORIDADE + ["llama3", "mistral", "gemma2", "qwen3:1.7b", "qwen3:4b"]
MODELOS_DESCRICAO = {
    "phi4-mini": "Boa síntese técnica e normativas",
    "gemma3": "Multilíngue, boa com textos jurídicos",
    "qwen2": "Leve e rápido no CPU",
    "phi3": "Mais leve, respostas rápidas",
    "llama3": "Referência clássica, uso geral",
    "mistral": "Boa para análise textual",
    "gemma2": "Multilíngue, fluidez em português",
    "qwen3": "Raciocínio robusto; requer Ollama atualizado",
}

DIRETRIZES_ANALITICAS = """Você é um assistente de pesquisa sociológica, especializado na governança da inovação em Instituições de Ciência, Tecnologia e Inovação (ICTIs) de direito público da Rede Federal de Educação Profissional, Científica e Tecnológica (Institutos Federais).

OBJETIVO DA SESSÃO DE PESQUISA:
{objetivo}

QUADRO ANALÍTICO - responda articulando, quando aplicáveis, as cinco dimensões da Sociologia da Inovação:

1. FUNDAMENTOS EPISTEMOLÓGICOS DA POLÍTICA: quais formas de conhecimento são reconhecidas, valorizadas e priorizadas? A política reforça o pensamento imitativo, o eurocentrismo e o tecnocentrismo, ou promove a simetria de saberes (justiça epistêmica, pluriversalidade, colonialidade do saber - Guerreiro Ramos)?
2. ATORES, REDES E RELAÇÕES DE PODER: quem participa da formulação e da implementação da política? O instrumento normativo atua como Ponto de Passagem Obrigatório (Teoria Ator-Rede - Callon), centralizando poder, ou distribui agência? Quem é incluído ou excluído (sociedade civil, comunidades, arranjos produtivos locais, economia solidária)?
3. RELEVÂNCIA E IMPACTO SOCIAL: a política transcende o imperativo de mercado? Prioriza a solução de problemas locais e a distribuição equitativa de benefícios (razão substantiva - Guerreiro Ramos; Bem Viver - Acosta; ponto de vista da sociedade civil - Burawoy)?
4. DESENHO INSTITUCIONAL E GOVERNANÇA: as normas operam como Código Técnico rígido (Feenberg) ou abrem margem de manobra para a Racionalização Subversiva nos campi? Há governança participativa, transparência e autonomia institucional (gestores de NIT, pesquisadores)?
5. CONTEXTUALIZAÇÃO E DESENVOLVIMENTO REGIONAL: a política responde ao contexto sócio-histórico, cultural e ambiental do território? Promove desenvolvimento endógeno e redução sociológica (Guerreiro Ramos) ou reproduz dependências históricas e externas?

PRINCÍPIOS ORIENTADORES: especificidade do contexto e pluriversalidade; justiça social e equidade; justiça epistêmica e cocriação de conhecimento; impacto social transformador; propósito público e governança democrática.

REGRAS DE CONDUTA ANALÍTICA:
- Trate a inovação como fenômeno socialmente construído e campo de disputa; evite o solucionismo tecnológico e a suposta neutralidade da técnica.
- Distinga racionalidade instrumental (métricas de patentes, produtividade, eficiência de mercado) de racionalidade substantiva (bem-estar coletivo, valores, impacto social).
- Contextualize territorialmente: identifique a instituição, a UF, o ano e o tipo de documento (política de inovação, edital, portaria, regimento de NIT, legislação federal).
- Fundamente cada afirmação em trechos dos documentos recuperados; quando a evidência for insuficiente, registre a lacuna como hipótese analítica, sem inventar fontes.
- Aponte tensões entre o arcabouço normativo nacional e as dinâmicas locais dos campi e territórios.
- Estruture a resposta em seções curtas: síntese; leitura por dimensões; tensões e lacunas; implicações para a pesquisa."""

def realizar_backup():
    data = datetime.now().strftime("%Y-%m-%d")
    nome_zip = f"backup_tese_{data}"
    arquivos_para_backup = ["memoria_pesquisa.json", "registro_analise_tese.txt", ARQUIVO_MATRIZ]
    if not os.path.exists(nome_zip): os.makedirs(nome_zip)
    for p in [PASTA_BASE, DB_DIR]:
        if os.path.exists(p):
            shutil.copytree(p, os.path.join(nome_zip, os.path.basename(p)), dirs_exist_ok=True)
    for a in arquivos_para_backup:
        if os.path.exists(a): shutil.copy(a, nome_zip)
    shutil.make_archive(nome_zip, 'zip', nome_zip)
    shutil.rmtree(nome_zip)
    return f"{nome_zip}.zip"

def registrar_na_matriz(eixo, variavel, instituto, resumo_ia):
    nova_linha = {
        "Data": datetime.now().strftime("%d/%m/%Y"),
        "Eixo": eixo,
        "Variável": variavel,
        "Instituto": instituto,
        "Achado Sociológico": resumo_ia
    }
    if os.path.exists(ARQUIVO_MATRIZ):
        df = pd.read_csv(ARQUIVO_MATRIZ)
        df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    else:
        df = pd.DataFrame([nova_linha])
    df.to_csv(ARQUIVO_MATRIZ, index=False, encoding="utf-8-sig")

def salvar_no_log(pergunta, resposta, modelo, objetivo="Análise Geral"):
    data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open("registro_analise_tese.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"📝 ENTRADA DE DIÁRIO DE BORDO - {data_hora}\n")
        f.write(f"MODELO UTILIZADO: {modelo}\n")
        f.write(f"OBJETIVO DA SESSÃO: {objetivo}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"❓ PERGUNTA DO PESQUISADOR:\n{pergunta}\n\n")
        f.write(f"🤖 RESPOSTA DA IA:\n{resposta}\n\n")
        f.write(f"✍️ NOTAS SOCIOLÓGICAS (Preencher Manualmente):\n")
        f.write(f"- Insight:\n")
        f.write(f"- Validação de Fontes:\n")
        f.write(f"{'_'*60}\n")

def carregar_memoria_conversa():
    """Carrega histórico com versionamento - arquiva versões antigas."""
    arquivo = Path("memoria_pesquisa.json")
    if not arquivo.exists(): return []
    
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            historico = json.load(f)
        
        # Versionamento: se houver mais de 50 entradas, archive a primeira terceira parte
        if len(historico) > 50:
            terca_parte = len(historico) // 3
            entradas_antigas = historico[:terca_parte]
            entradas_atuais = historico[terca_parte:]
            
            # Salvar arquivado
            arquivo_archive = Path("memoria_archive.json")
            if not arquivo_archive.exists():
                with open(arquivo_archive, "w", encoding="utf-8") as f:
                    json.dump(entradas_antigas, f, ensure_ascii=False, indent=4)
            
            # Manter apenas as atuais + 10 de margem
            return entradas_atuais[-50:]
        return historico
    except (json.JSONDecodeError, KeyError):
        # Arquivo corrompido - reiniciar histórico vazio
        return []

def salvar_memoria_conversa(historico):
    """Salva histórico com controle de versão - mantém última terceira parte."""
    arquivo = Path("memoria_pesquisa.json")
    
    # Se houver mais de 50 entradas, aplicar versionamento suave
    if len(historico) > 50:
        terca_parte = len(historico) // 3
        entradas_antigas = historico[:terca_parte]
        entradas_atuais = historico[terca_parte:]
        
        # Arquivar entradas antigas
        arquivo_archive = Path("memoria_archive.json")
        if not arquivo_archive.exists():
            with open(arquivo_archive, "w", encoding="utf-8") as f:
                json.dump(entradas_antigas, f, ensure_ascii=False, indent=4)
        
        # Manter apenas as atuais (limitado a 50)
        historico = entradas_atuais[-50:]
    
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=4)

def _filtro_pdf(texto) -> str:
    """Mantém ASCII e acentos latin-1 (faixa suportada pela fonte core); remove o resto."""
    saida = []
    for ch in str(texto):
        o = ord(ch)
        if ch in "\t\n\r" or 0x20 <= o <= 0x7E or 0xA0 <= o <= 0xFF:
            saida.append(ch)
        else:
            saida.append(" ")
    return "".join(saida)


def _secao_pdf(pdf, texto):
    pdf.set_font("Arial", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 8, text=_filtro_pdf(texto))
    pdf.ln(1)


def _sub_pdf(pdf, texto):
    pdf.set_font("Arial", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6.5, text=_filtro_pdf(texto))
    pdf.ln(0.5)


def _linha_pdf(pdf, texto, tamanho=10):
    pdf.set_font("Arial", "", tamanho)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, text=_filtro_pdf(texto))


def _tabela_pdf(pdf, cabecalhos, linhas, largs, tamanho=8):
    """Tabela simples de uma linha por registro (sem quebra interna)."""
    from fpdf.enums import XPos, YPos
    w_total = pdf.w - pdf.l_margin - pdf.r_margin
    largura = [w_total * p / sum(largs) for p in largs]

    def celula(texto, w_final=False):
        pdf.set_font("Arial", "B", tamanho)
        texto_c = _filtro_pdf(str(texto))
        while pdf.get_string_width(texto_c) > w_total * 0.95 and len(texto_c) > 4:
            texto_c = texto_c[:-1]
        if w_final:
            pdf.cell(w_total, 5, text=texto_c, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.cell(w_total, 5, text=texto_c, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)

    for i, h in enumerate(cabecalhos):
        celula(h, w_final=(i == len(cabecalhos) - 1))
    for linha in linhas:
        for j, val in enumerate(linha):
            if j < len(largura):
                pdf.set_font("Arial", "", tamanho)
                texto_c = _filtro_pdf(str(val))
                while pdf.get_string_width(texto_c) > largura[j] - 3 and len(texto_c) > 3:
                    texto_c = texto_c[:-1]
                if j == len(linha) - 1:
                    pdf.cell(largura[j], 5, text=texto_c, border=1,
                             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                else:
                    pdf.cell(largura[j], 5, text=texto_c, border=1,
                             new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(3)


def gerar_pdf_consolidado(historico, filtros=None):
    """Relatório PDF consolidado: escopo/análise textual + conversa do Chat RAG."""
    filtros = filtros or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(14, 14, 14)

    pdf.set_font("Arial", "B", 16)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 8, text="Relatorio de Analise: Politicas de Inovacao IFs", align="C")
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, text=_filtro_pdf(
        "Relatório consolidado do dashboard SocioInova RAG (Chat RAG e Análise Textual)."),
        align="C")
    pdf.ln(4)

    # 1. Escopo da análise textual (filtros ativos)
    _secao_pdf(pdf, "1. Escopo da análise textual (filtros ativos)")
    for label, valores in (
        ("UFs", filtros.get("ufs")), ("Institutos", filtros.get("ifs")),
        ("Anos", filtros.get("anos")), ("Regiões", filtros.get("regioes")),
        ("Tipos", filtros.get("tipos")), ("Dimensões (Quadro 1.5)", filtros.get("dimensoes")),
    ):
        texto = f"{label}: {', '.join(str(v) for v in valores) if valores else 'todos'}"
        _linha_pdf(pdf, "• " + texto)

    # 2. Resultados da Análise Textual
    docs = ac.carregar_documentos_analise(filtros)
    if docs:
        _secao_pdf(pdf, "2. Análise textual do corpus filtrado")
        _linha_pdf(pdf, f"Documentos analisados: {len(docs)}.")
        pdf.ln(1)

        _sub_pdf(pdf, "2.1 Termos mais frequentes (nuvem de palavras)")
        for termo, n in ac.top_termos(docs, 12):
            _linha_pdf(pdf, f"  • {termo}: {n}", tamanho=9)
        pdf.ln(1)

        _sub_pdf(pdf, "2.2 Frequência de temas por dimensão analítica (Quadro 1.5)")
        df = ac.frequencia_temas(docs)
        linhas_tabela = [
            [r.Eixo, int(r.Documentos), int(r.Sinais), int(r.Ocorrências),
             f"{r._5:.1f}%", f"{r._6:.1f}%"]
            for r in df.itertuples()
        ]
        _tabela_pdf(pdf, ["Eixo / dimensão", "Docs", "Sinais", "Ocorr.", "Prof. %", "% docs"],
                    linhas_tabela, [86, 18, 18, 20, 18, 18])
        pdf.ln(1)

        _sub_pdf(pdf, "2.3 Quebra por Região e por Ano")
        for categoria, nome in (("regiao", "Região"), ("ano", "Ano")):
            dfp = ac.frequencia_temas_por(docs, categoria)
            if not dfp.empty:
                _linha_pdf(pdf, f"{nome}:", tamanho=9)
                _tabela_pdf(pdf, [nome, "Eixo / dimensão", "Docs"],
                            [[getattr(r, categoria), r.Eixo, int(r.Documentos)] for r in dfp.itertuples()],
                            [26, 120, 22])
                pdf.ln(1)

        _sub_pdf(pdf, "2.4 Sinais analíticos por dimensão (checagem de lacunas)")
        total_ausentes = 0
        total_sinais = 0
        todas_dim = ac.sinais_todas_dimensoes(docs)
        for eixo in ac.EIXOS_ANALISE:
            s = todas_dim.get(eixo)
            if s is None or s.empty:
                continue
            presentes = int((s["Situação"] == "Presente").sum())
            ausentes_lista = s.loc[s["Situação"] == "Ausente", "Sinal"].tolist()
            total_ausentes += len(ausentes_lista)
            total_sinais += len(s)
            texto = f"{eixo}: {presentes} sinais presentes de {len(s)}."
            if ausentes_lista:
                texto += " Ausentes: " + "; ".join(ausentes_lista[:10])
                if len(ausentes_lista) > 10:
                    texto += " …"
            _linha_pdf(pdf, "  • " + texto, tamanho=9)
        _linha_pdf(pdf, f"Total: {total_sinais} sinais analíticos, {total_ausentes} ausentes no recorte (lacunas).", tamanho=9)
        pdf.ln(1)

        _sub_pdf(pdf, "2.5 Tópicos latentes (LDA)")
        lda = ac.lda_topicos(docs)
        if lda:
            for i, topo in enumerate(lda["topicos"], 1):
                _linha_pdf(pdf, "  • Tópico " + str(i) + ": " + ", ".join(p for p, _ in topo[:8]), tamanho=9)
        else:
            _linha_pdf(pdf, "  — Corpus pequeno demais para LDA.", tamanho=9)
    else:
        _secao_pdf(pdf, "2. Análise textual do corpus filtrado")
        _linha_pdf(pdf, "Nenhum documento corresponde aos filtros ativos.")

    # 3. Conversa com o Chat RAG
    _secao_pdf(pdf, "3. Conversa com o Chat RAG")
    if not historico:
        _linha_pdf(pdf, "Sem conversas registradas nesta sessão.")
    for msg in historico:
        role = "Pesquisador" if msg["role"] == "user" else f"IA ({msg.get('model', 'Assistente')})"
        pdf.set_font("Arial", "B", 11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, text=_filtro_pdf(role) + ":")
        pdf.set_font("Arial", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, text=_filtro_pdf(msg["content"]))
        pdf.ln(2)

    return bytes(pdf.output(dest="S"))

def inicializar_planilha_tese():
    if not os.path.exists(ARQUIVO_MATRIZ):
        colunas = ["Data", "Eixo", "Variável", "Instituto", "Achado Sociológico", "Evidência/Fonte"]
        df = pd.DataFrame(columns=colunas)
        df.to_csv(ARQUIVO_MATRIZ, index=False, encoding="utf-8-sig")

# Cache hierárquico com validação por data de modificação
@st.cache_data(ttl=3600, show_spinner="Carregando base de documentos...")
def carregar_dados_hierarquicos_cache(pasta_raiz_str):
    """Versão cacheada apoiada no corpus persistente (re-extração incremental por mtime)."""
    if not os.path.exists(pasta_raiz_str):
        return []
    return ac.carregar_docs_paginas()


@st.cache_resource(show_spinner="Indexando fragmentos...")
def get_cached_docs(pasta_raiz):
    """Alias para compatibilidade com código existente."""
    return carregar_dados_hierarquicos_cache(pasta_raiz)

# --- 3. MODELOS E RETRIEVER ---
@st.cache_resource
def carregar_llm(nome_modelo):
    return ChatOllama(model=nome_modelo, temperature=0, num_predict=MAX_TOKENS_RESPOSTA)

@st.cache_data(ttl=300)
def obter_modelos_disponiveis():
    """Lista modelos de chat disponíveis no Ollama, com fallback para defaults."""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            linhas = result.stdout.strip().split("\n")[1:]  # Pular header
            modelos = []
            for linha in linhas:
                # Formato: "llama3:latest    8B    q8_0    4.7 GB"
                partes = linha.split()
                if not partes:
                    continue
                nome_completo = partes[0]
                base = nome_completo.split(":")[0] if ":" in nome_completo else nome_completo
                # Ignorar apenas modelos de embedding; manter todos os tags (ex.: qwen3:1.7b e qwen3:4b)
                if base in MODELOS_EMBEDDING or nome_completo in modelos:
                    continue
                modelos.append(nome_completo)
            # Ordenar: modelos recomendados primeiro, depois o restante alfabeticamente
            prioridade = {m: i for i, m in enumerate(MODELOS_PRIORIDADE)}
            def chave_ordem(nome):
                return (prioridade.get(nome, 999), nome.lower())
            modelos_ordenados = sorted(modelos, key=chave_ordem)
            return modelos_ordenados if modelos_ordenados else MODELOS_FALLBACK
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    # Fallback para defaults conhecidos
    return MODELOS_FALLBACK

@st.cache_resource
def carregar_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def configurar_retriever():
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=carregar_embeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return retriever
    return None

def rotulo_modelo(nome_modelo):
    """Retorna o nome do modelo com breve justificativa de uso (ex.: gemma3:4b (Multilíngue...))."""
    base = nome_modelo.split(":")[0]
    descricao = MODELOS_DESCRICAO.get(base)
    return f"{nome_modelo} ({descricao})" if descricao else nome_modelo

# --- 4. BARRA LATERAL ---
inicializar_planilha_tese()
if "messages" not in st.session_state:
    st.session_state.messages = carregar_memoria_conversa()

with st.sidebar:
    st.header("🤖 Configurações de IA")
    modelos_disponiveis = obter_modelos_disponiveis()
    indice_padrao = modelos_disponiveis.index(LLM_PADRAO) if LLM_PADRAO in modelos_disponiveis else 0
    selecao_label = st.selectbox("Modelo Ativo:", modelos_disponiveis, index=indice_padrao, format_func=rotulo_modelo)
    llm = carregar_llm(selecao_label)
    
    st.divider()
    st.subheader("📝 Diário de Bordo")
    objetivo_sessao = st.text_input("Objetivo da Pesquisa hoje:", value="Análise Comparativa de Políticas")
    
    st.divider()
    st.header("🔍 Filtros de Analise")
    if os.path.exists(PASTA_BASE):
        ufs_disponiveis, ifs_disponiveis, _ = ac.resumo_estrutura()

        all_ufs = st.checkbox("Selecionar Todas UFs", value=False)
        filtro_uf = st.multiselect("Filtrar por UF:", ufs_disponiveis) if not all_ufs else ufs_disponiveis
        all_ifs = st.checkbox("Selecionar Todos IFs", value=False)
        filtro_if = st.multiselect("Filtrar por Instituto:", ifs_disponiveis) if not all_ifs else ifs_disponiveis
        incluir_federal = st.checkbox("Incluir Legislação Federal (leis e decretos base)", value=True)

        if st.button("🔄 Indexar/Atualizar Base"):
            with st.spinner("Extraindo texto e indexando fragmentos..."):
                def selecionar(d):
                    return (all_ufs or not filtro_uf or d.metadata["uf"] in filtro_uf) and (
                        (all_ifs or not filtro_if or d.metadata["instituto"] in filtro_if)
                        or (incluir_federal and d.metadata.get("tipo") == "Legislação Federal"))
                docs_filtrados = [d for d in carregar_dados_hierarquicos_cache(PASTA_BASE) if selecionar(d)]
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\nArt. ", "\n§ ", "\nI - ", "\n\n", " "])
                chunks = splitter.split_documents(docs_filtrados)
                if not chunks:
                    st.warning("Nenhum documento selecionado para indexar. Marque UFs/IFs ou deixe os filtros vazios para indexar toda a base.")
                else:
                    Chroma.from_documents(documents=chunks, embedding=carregar_embeddings(), persist_directory=DB_DIR)
                    st.cache_resource.clear()
                    st.success(f"Indexado com sucesso: {len(chunks)} trechos!")

    st.divider()
    st.subheader("💾 Gestão")
    if st.button("📦 Gerar Backup (.zip)"):
        nome_zip = realizar_backup()
        with open(nome_zip, "rb") as f:
            st.download_button("Baixar Backup", data=f, file_name=nome_zip)
    st.download_button(
        "📄 Exportar Relatório consolidado (PDF)",
        data=gerar_pdf_consolidado(
            st.session_state.messages,
            st.session_state.get("_filtros_analise_atuais")),
        file_name="relatorio_consolidado.pdf")
    if st.session_state.messages:
        if st.button("🗑️ Limpar Chat"):
            st.session_state.messages = []; salvar_memoria_conversa([]); st.rerun()

# --- 5. AREA DE CHAT ---
st.title("🏛️ SocioInova RAG")
st.caption(f"IA: {selecao_label} | Pesquisador: Jonatã França Bittencourt | Orientador: Prof. Leonardo Fernandes Nascimento")

aba_chat, aba_analise = st.tabs(["💬 Chat RAG", "📊 Análise Textual"])

with aba_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Inicie sua análise sociológica..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever = configurar_retriever()
            if retriever:
                prompt_doc = ChatPromptTemplate.from_messages([
                    ("system", DIRETRIZES_ANALITICAS),
                    ("human", "Contexto recuperado dos documentos:\n{context}\n\nPergunta do pesquisador:\n{question}"),
                ])
                campos = {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "objetivo": lambda _: objetivo_sessao,
                }
                chain = (campos | prompt_doc | llm | StrOutputParser())

                with st.status(f"Processando com {selecao_label}...", expanded=False) as status:
                    try:
                        docs_rec = retriever.invoke(prompt)
                    except Exception as e:
                        st.error(f"Erro na recuperação de contexto: {str(e)[:200]}")
                        docs_rec = []
                    status.update(label="Evidências localizadas. Redigindo...", state="running")
                    try:
                        full_res = st.write_stream(chain.stream(prompt))
                    except Exception as e:
                        st.warning(f"Erro na geração da resposta da IA: {str(e)[:300]}")
                        full_res = ""
                    status.update(label=f"Concluído por {selecao_label}", state="complete")

                if full_res:
                    salvar_no_log(prompt, full_res, selecao_label, objetivo=objetivo_sessao)
                    st.session_state.messages.append({"role": "assistant", "content": full_res, "model": selecao_label})
                    salvar_memoria_conversa(st.session_state.messages)

                with st.expander("🔍 Auditoria de Fontes"):
                    for d in docs_rec:
                        titulo = d.metadata.get("tipo") if d.metadata.get("tipo") == "Legislação Federal" else d.metadata['instituto']
                        tipo_tag = f" | {d.metadata['tipo']}" if d.metadata.get("tipo") else ""
                        st.write(f"**{titulo}** ({d.metadata['uf']}, {d.metadata['ano']}){tipo_tag}")
                        st.caption(d.page_content)
                        st.divider()

                st.subheader("📥 Registrar na Matriz de Extração")
                with st.form("extracao_dados", clear_on_submit=True):
                    c1, c2 = st.columns(2)
                    eixo = c1.selectbox("Eixo:", list(ac.EIXOS_ANALISE.keys()))
                    variavel = c2.text_input("Variável:", placeholder="Ex: Royalties")
                    resumo = st.text_area("Achado Sociológico:", value=full_res[:500] + "...")
                    if st.form_submit_button("Confirmar Registro"):
                        inst = docs_rec[0].metadata['instituto'] if docs_rec else "N/A"
                        registrar_na_matriz(eixo, variavel, inst, resumo)
                        st.success("Registrado na planilha CSV!")
            else:
                st.warning("Indexe a base primeiro.")

with aba_analise:
    analise_ui.render_aba(ac.resumo_estrutura())