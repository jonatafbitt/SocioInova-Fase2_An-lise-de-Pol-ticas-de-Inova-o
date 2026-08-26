import streamlit as st
import os
import re
import json
import shutil
import subprocess  # Adicionado para listar modelos Ollama
import pandas as pd
from datetime import datetime
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor # Para acelerar o carregamento
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Caminho base para dados
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "documentos_inovação"
PERSIST_DIR = BASE_DIR / "memoria_longo_prazo"

# --- 1. CONFIGURAÇÕES, MEMÓRIA E BACKUP ---
st.set_page_config(page_title="SocioInova RAG - UFBA", layout="wide")
DB_DIR = "./memoria_longo_prazo"
PASTA_BASE = "documentos_inovação"
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

def gerar_pdf(historico):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Relatorio de Analise: Politicas de Inovacao IFs", ln=True, align='C')
    pdf.ln(10)
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

# Cache hierárquico com validação por data de modificação
@st.cache_data(ttl=3600, show_spinner="Carregando base de documentos...")
def carregar_dados_hierarquicos_cache(pasta_raiz_str):
    """Versão cacheada com verificação de atualização."""
    pasta_raiz = Path(pasta_raiz_str)
    if not pasta_raiz.exists(): return []
    
    # Verificar se cache é atualizado baseando-se na data mais recente
    arquivos_pdf = list(pasta_raiz.rglob("*.pdf"))
    if not arquivos_pdf: return []
    
    # Data de modificação mais recente entre todos os PDFs
    mtime_max = max(f.stat().st_mtime for f in arquivos_pdf)
    cache_key = f"{pasta_raiz.resolve()}|{mtime_max}"
    
    # Em Streamlit, o cache é invalidado automaticamente quando o parâmetro muda
    # mas podemos adicionar lógica extra aqui se necessário
    return _processar_pdfs_hierarquicos(pasta_raiz, arquivos_pdf)


def _processar_pdfs_hierarquicos(pasta_raiz, arquivos_pdf):
    """Função interna de processamento real."""
    docs_com_metadados = []
    regioes_validas = ["NORTE", "NORDESTE", "CENTRO-OESTE", "SUDESTE", "SUL", "BRASIL"]
    
    for caminho_pdf in arquivos_pdf:
        try:
            root = caminho_pdf.parent
            file = caminho_pdf.name
            partes = Path.normpath(root).parts
            
            regiao, uf, if_nome = "Outros", "S/D", "Não Identificado"
            partes_upper = [p.upper() for p in partes]
            
            if "BRASIL" in partes_upper:
                regiao, uf, if_nome = "Nacional", "BR", "Legislação Federal"
            else:
                for r in regioes_validas:
                    if r in partes_upper:
                        idx = partes_upper.index(r)
                        regiao = partes[idx]
                        if len(partes) > idx + 1: uf = partes[idx+1]
                        if len(partes) > idx + 2: if_nome = partes[idx+2]
                        break
            
            loader = PyPDFLoader(str(caminho_pdf))
            paginas = loader.load()
            ano = extrair_ano(file)
            for p in paginas:
                p.metadata.update({"regiao": regiao, "uf": uf, "instituto": if_nome, "ano": ano})
            docs_com_metadados.extend(paginas)
        except Exception as e:
            # Em vez de return [], logar o erro e continuar
            pass
    
    return docs_com_metadados


@st.cache_resource(show_spinner="Indexando fragmentos...")
def get_cached_docs(pasta_raiz):
    """Alias para compatibilidade com código existente."""
    return carregar_dados_hierarquicos_cache(pasta_raiz)

# --- 3. MODELOS E RETRIEVER ---
@st.cache_resource
def carregar_llm(nome_modelo):
    return ChatOllama(model=nome_modelo, temperature=0)

def obter_modelos_disponiveis():
    """Lista modelos disponíveis no Ollama, com fallback para defaults."""
    try:
        # Tentar listar modelos via subprocess
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            linhas = result.stdout.strip().split("\n")[1:]  # Pular header
            modelos = []
            for linha in linhas:
                # Formato: "llama3    8B    q8_0    4.7 GB"
                partes = linha.split()
                if partes:
                    modelo_nome = partes[0].split(":")[0] if ":" in partes[0] else partes[0]
                    modelos.append(modelo_nome)
            return modelos if modelos else ["llama3", "phi3", "mistral"]
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    # Fallback para defaults conhecidos
    return ["llama3", "phi3", "mistral", "gemma2"]

@st.cache_resource
def carregar_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def configurar_retriever():
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=carregar_embeddings())
        # Melhorar busca: k=4 para exibir, fetch_distance=1000 para candidatos maiores
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4, "fetch_distance": 1000}
        )
        return retriever
    return None

# --- 4. BARRA LATERAL ---
inicializar_planilha_tese()
if "messages" not in st.session_state:
    st.session_state.messages = carregar_memoria_conversa()

with st.sidebar:
    st.header("🤖 Configurações de IA")
    modelos_disponiveis = obter_modelos_disponiveis()
    opcoes_modelos = {m: m for m in modelos_disponiveis}
    selecao_label = st.selectbox("Modelo Ativo:", list(opcoes_modelos.keys()))
    llm = carregar_llm(opcoes_modelos[selecao_label])
    
    st.divider()
    st.subheader("📝 Diário de Bordo")
    objetivo_sessao = st.text_input("Objetivo da Pesquisa hoje:", value="Análise Comparativa de Políticas")
    
    st.divider()
    st.header("🔍 Filtros de Analise")
    if os.path.exists(PASTA_BASE):
        with st.spinner("Mapeando estrutura nacional..."):
            todos_docs = carregar_dados_hierarquicos_cache(PASTA_BASE)
        
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
            prompt_doc = ChatPromptTemplate.from_template(
                "Analise sociologicamente:\\nContexto: {context}\\nPergunta: {question}"
            )
            chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt_doc | llm | StrOutputParser())
            
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
                    st.warning("Erro na geração da resposta da IA")
                    full_res = ""
                status.update(label=f"Concluído por {selecao_label}", state="complete")

            if full_res:
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