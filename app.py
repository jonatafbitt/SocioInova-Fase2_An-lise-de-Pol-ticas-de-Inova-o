import streamlit as st
import os
import re
import json
import shutil
import pandas as pd
from datetime import datetime
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor # Para acelerar o carregamento
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURAÇÕES, MEMÓRIA E BACKUP ---
st.set_page_config(page_title="SocioInova RAG - UFBA", layout="wide")
DB_DIR = "./memoria_longo_prazo"
PASTA_BASE = "C:/Users/jonat/analise_ifs/documentos_inovação"
ARQUIVO_MATRIZ = "matriz_extracao_tese.csv"

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

def carregar_memoria_conversa():
    if os.path.exists("memoria_pesquisa.json"):
        with open("memoria_pesquisa.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_memoria_conversa(historico):
    with open("memoria_pesquisa.json", "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=4)

def gerar_pdf(historico):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Relatorio de Analise: Politicas de Inovacao IFs", ln=True, align='C')
    pdf.ln(10)
    for msg in historico:
        role = "Pesquisador" if msg["role"] == "user" else f"IA ({msg.get('model', 'Assistente')})"
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, txt=f"{role}:")
        pdf.set_font("Arial", '', 11)
        texto_limpo = msg["content"].encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 8, txt=texto_limpo)
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

def inicializar_planilha_tese():
    if not os.path.exists(ARQUIVO_MATRIZ):
        colunas = ["Data", "Eixo", "Variável", "Instituto", "Achado Sociológico", "Evidência/Fonte"]
        df = pd.DataFrame(columns=colunas)
        df.to_csv(ARQUIVO_MATRIZ, index=False, encoding="utf-8-sig")

# --- 2. PROCESSAMENTO HIERÁRQUICO ---
def extrair_ano(texto):
    match = re.search(r'20[0-2][0-9]', texto)
    return int(match.group(0)) if match else 0

@st.cache_resource
def carregar_dados_hierarquicos(pasta_raiz):
    docs_com_metadados = []
    if not os.path.exists(pasta_raiz): return []
    
    regioes_validas = ["NORTE", "NORDESTE", "CENTRO-OESTE", "SUDESTE", "SUL", "BRASIL"]
    arquivos_pdf = []

    # Passo 1: Listagem rápida dos arquivos
    for root, _, files in os.walk(pasta_raiz):
        for file in files:
            if file.endswith(".pdf"):
                arquivos_pdf.append(os.path.join(root, file))

    # Passo 2: Função de processamento por arquivo para Multithreading
    def processar_arquivo(caminho_completo):
        try:
            root = os.path.dirname(caminho_completo)
            file = os.path.basename(caminho_completo)
            partes = os.path.normpath(root).split(os.sep)
            
            regiao, uf, if_nome = "Outros", "S/D", "Não Identificado"
            if "BRASIL" in [p.upper() for p in partes]:
                regiao, uf, if_nome = "Nacional", "BR", "Legislação Federal"
            else:
                for r in regioes_validas:
                    if r in [p.upper() for p in partes]:
                        idx = [p.upper() for p in partes].index(r)
                        regiao = partes[idx]
                        if len(partes) > idx + 1: uf = partes[idx+1]
                        if len(partes) > idx + 2: if_nome = partes[idx+2]
                        break
            
            loader = PyPDFLoader(caminho_completo)
            paginas = loader.load()
            ano = extrair_ano(file)
            for p in paginas:
                p.metadata.update({"regiao": regiao, "uf": uf, "instituto": if_nome, "ano": ano})
            return paginas
        except:
            return []

    # Passo 3: Execução paralela (Acelera drasticamente a base nacional)
    with ThreadPoolExecutor() as executor:
        resultados = list(executor.map(processar_arquivo, arquivos_pdf))
    
    for r in resultados:
        docs_com_metadados.extend(r)
        
    return docs_com_metadados

# --- 3. MODELOS E RETRIEVER ---
@st.cache_resource
def carregar_llm(nome_modelo):
    return ChatOllama(model=nome_modelo, temperature=0)

@st.cache_resource
def carregar_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def configurar_retriever():
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=carregar_embeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    return None

# --- 4. BARRA LATERAL ---
inicializar_planilha_tese()
if "messages" not in st.session_state:
    st.session_state.messages = carregar_memoria_conversa()

with st.sidebar:
    st.header("🤖 Configurações de IA")
    opcoes_modelos = {"Llama 3 (8B)": "llama3", "Phi-3 Mini (3.8B)": "phi3", "Mistral (7B)": "mistral", "Gemma 2 (9B)": "gemma2"}
    selecao_label = st.selectbox("Modelo Ativo:", list(opcoes_modelos.keys()))
    llm = carregar_llm(opcoes_modelos[selecao_label])
    
    st.divider()
    st.subheader("📝 Diário de Bordo")
    objetivo_sessao = st.text_input("Objetivo da Pesquisa hoje:", value="Análise Comparativa de Políticas")
    
    st.divider()
    st.header("🔍 Filtros de Analise")
    if os.path.exists(PASTA_BASE):
        with st.spinner("Mapeando estrutura nacional..."):
            todos_docs = carregar_dados_hierarquicos(PASTA_BASE)
        
        ufs_disponiveis = sorted(list(set(d.metadata["uf"] for d in todos_docs)))
        ifs_disponiveis = sorted(list(set(d.metadata["instituto"] for d in todos_docs)))
        
        all_ufs = st.checkbox("Selecionar Todas UFs", value=False)
        filtro_uf = st.multiselect("Filtrar por UF:", ufs_disponiveis) if not all_ufs else ufs_disponiveis
        all_ifs = st.checkbox("Selecionar Todos IFs", value=False)
        filtro_if = st.multiselect("Filtrar por Instituto:", ifs_disponiveis) if not all_ifs else ifs_disponiveis
        
        if st.button("🔄 Indexar/Atualizar Base"):
            with st.spinner("Indexando fragmentos..."):
                docs_filtrados = [d for d in todos_docs if (all_ufs or d.metadata["uf"] in filtro_uf) and (all_ifs or d.metadata["instituto"] in filtro_if)]
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\nArt. ", "\n§ ", "\nI - ", "\n\n", " "])
                chunks = splitter.split_documents(docs_filtrados)
                Chroma.from_documents(documents=chunks, embedding=carregar_embeddings(), persist_directory=DB_DIR)
                st.cache_resource.clear()
                st.success(f"Indexado com sucesso: {len(chunks)} trechos!")

    st.divider()
    st.subheader("💾 Gestão")
    if st.button("📦 Gerar Backup (.zip)"):
        nome_zip = realizar_backup()
        with open(nome_zip, "rb") as f:
            st.download_button("Baixar Backup", data=f, file_name=nome_zip)
    if st.session_state.messages:
        st.download_button("📄 Exportar PDF", data=gerar_pdf(st.session_state.messages), file_name="tese_analise.pdf")
        if st.button("🗑️ Limpar Chat"):
            st.session_state.messages = []; salvar_memoria_conversa([]); st.rerun()

# --- 5. AREA DE CHAT ---
st.title("🏛️ SocioInova RAG")
st.caption(f"IA: {selecao_label} | Pesquisador: Jonatã França Bittencourt | Orientador: Prof. Leonardo Fernandes Nascimento")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Inicie sua análise sociológica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        retriever = configurar_retriever()
        if retriever:
            prompt_doc = ChatPromptTemplate.from_template("Analise sociologicamente:\nContexto: {context}\nPergunta: {question}")
            chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt_doc | llm | StrOutputParser())
            
            with st.status(f"Processando com {selecao_label}...", expanded=False) as status:
                docs_rec = retriever.invoke(prompt)
                status.update(label="Evidências localizadas. Redigindo...", state="running")
                full_res = st.write_stream(chain.stream(prompt))
                status.update(label=f"Concluído por {selecao_label}", state="complete")

            salvar_no_log(prompt, full_res, selecao_label, objetivo=objetivo_sessao)
            st.session_state.messages.append({"role": "assistant", "content": full_res, "model": selecao_label})
            salvar_memoria_conversa(st.session_state.messages)
            
            with st.expander("🔍 Auditoria de Fontes"):
                for d in docs_rec:
                    st.write(f"**{d.metadata['instituto']}** ({d.metadata['uf']}, {d.metadata['ano']})")
                    st.caption(d.page_content)
                    st.divider()

            st.subheader("📥 Registrar na Matriz de Extração")
            with st.form("extracao_dados", clear_on_submit=True):
                c1, c2 = st.columns(2)
                eixo = c1.selectbox("Eixo:", ["I. Governança", "II. Propriedade Intelectual", "III. Inovação Social", "IV. Capital Humano"])
                variavel = c2.text_input("Variável:", placeholder="Ex: Royalties")
                resumo = st.text_area("Achado Sociológico:", value=full_res[:500] + "...")
                if st.form_submit_button("Confirmar Registro"):
                    inst = docs_rec[0].metadata['instituto'] if docs_rec else "N/A"
                    registrar_na_matriz(eixo, variavel, inst, resumo)
                    st.success("Registrado na planilha CSV!")
        else: st.warning("Indexe a base primeiro.")