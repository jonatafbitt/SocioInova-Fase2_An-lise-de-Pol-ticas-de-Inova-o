# Socioinova RAG

Interface sociológica para análise de políticas de inovação em Institutos Federais usando arquitetura RAG (Retrieval-Augmented Generation) com Ollama e LangChain.

## 📋 Visão Geral

O **Socioinova RAG** é uma ferramenta de pesquisa desenvolvida para análise sociológica de políticas de inovação nas Instituições Federais de Educação Superior (IFs). O sistema processa documentos PDF hierarquicamente (região → UF → Instituto), extrai evidências via LLM local (Ollama) e mantena uma matriz de codificação sociológica registrada em CSV.

## ✨ Funcionalidades Principais

- **Processamento hierárquico de PDFs**: Estrutura região/UF/IF automática
- **Matriz de extração sociológica**: Codificação de eixos (Governança, Propriedade Intelectual, Inovação Social, Capital Humano) e variáveis
- **Diário de bordo**: Registro de perguntas, respostas e anotações manuais em `registro_analise_tese.txt`
- **Memória persistente**: Histórico de conversas em `memoria_pesquisa.json`
- **Backup automático**: Geração de `.zip` com base de dados, memórias e relatórios
- **Chat RAG**: Interface conversacional com recuperação de contexto de documentos locais
- **Análise decolonial**: Lente para identificar mimetismo vs. inovação situada nas políticas
- **Aba "Análise Textual"**: Nuvem de palavras, mapa dinâmico de co-ocorrência de termos, tópicos LDA e frequência de temas (eixos da pesquisa), com filtros por UF/Instituto/Ano/Região
- **Leitura de documentos**: Concordância KWIC (termo com contexto) e navegador de trechos por documento
- **Startup acelerado**: Cache persistente de extração de PDFs com re-extração incremental (por data de modificação) — o corpus é extraído uma única vez e reutilizado

## 🛠️ Tecnologias Utilizadas

| Categoria | Bibliotecas |
|---|---|
| **Interface** | streamlit, plotly |
| **Dados** | pandas, pypdf |
| **IA Local** | ollama, langchain, langchain-ollama, chromadb |
| **Relatórios** | fpdf2 |
| **Google Drive** | google-api-python-client, google-auth-httplib2, google-auth-oauthlib |

## 📦 Requisitos

- Python 3.10+
- Ollama instalado e modelos configurados (`ollama pull llama3`, `ollama pull nomic-embed-text`)
- Ambiente virtual recomendado

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

## 🚀 Como Executar

```bash
# 1. Ativar ambiente virtual (já configurado no iniciar_socioinova.bat)
.\venv\Scripts\activate

# 2. Iniciar o Streamlit
streamlit run app.py
```

Acesse: `http://localhost:8502`

## 📁 Estrutura de Pastas

```
socioinova-rag/
├── app.py                    # Interface Streamlit principal
├── iniciar_socioinova.bat    # Ativador de ambiente e interface
├── requirements.txt          # Dependências Python
├── .gitignore               # Padrões de exclusão Git
├── memoria_pesquisa.json    # Histórico de conversas
├── registro_analise_tese.txt # Diário de bordo sociológico
├── matriz_extracao_tese.csv # Matriz de codificação eixo/variável
├── config_rag.py            # Configuração LLM/Ollama embeddings
├── documentos_inovação/     # 76 PDFs organizados por região/UF/IF
│   └── ... (BRASIL, Centro Oeste, Nordeste, Norte, Sudeste, Sul)
├── memoria_longo_prazo/    # ChromaDB persistente + pickles
│   ├── chroma.sqlite3
│   └── índices/*.bin / *.pickle
└── README.md                # Esta documentação
```

## 📊 Fluxo de Trabalho Diário

1. **Iniciar**: Executar `iniciar_socioinova.bat` (ativa venv e abre o Streamlit)
2. **Indexar**: Clicar em "🔄 Indexar/Atualizar Base" para processar PDFs novos/atualizados
3. **Analisar**: Digitar perguntas no chat para análise sociológica com recuperação de contexto
4. **Registrar**: Usar o formulário "📥 Registrar na Matriz de Extração" para salvar achados
5. **Fazer Backup**: Clicar em "📦 Gerar Backup (.zip)" para salvar estado atual
6. **Exportar**: Baixar PDF ou JSON do histórico de análise

## 🏗️ Arquitetura RAG

```mermaid
graph LR
    A([PDFs documentos_inovação]) --> B[PyPDFLoader]
    B --> C[RecursiveCharacterTextSplitter]
    C --> D[OllamaEmbeddings(model="nomic-embed-text")]
    D --> E[ChromaDB persist_directory]
    E --> F[Retriever (k=4)]
    F --> G[ChatOllama(model=selecionado)]
    G --> H[Resposta sociológica]
```

A extração de texto dos PDFs é feita pelo módulo `analise_corpus.py`, que mantém um
cache persistente em `corpus_cache.pkl`: na primeira execução o texto é extraído uma vez;
nas seguintes, apenas PDFs novos ou alterados (detectados por `mtime`) são reprocessados.
As análises textuais (nuvem, co-ocorrência, LDA, KWIC) consomem esse corpus pré-processado.

## 📁 Estrutura de Pastas

```
socioinova-rag/
├── app.py                    # Interface Streamlit principal
├── analise_corpus.py         # Cache de extração + NLP leve (stopwords, LDA, KWIC)
├── analise_ui.py             # Renderização da aba "Análise Textual"
├── iniciar_socioinova.bat    # Ativador de ambiente e interface
├── requirements.txt          # Dependências Python
├── .gitignore
├── memoria_pesquisa.json
├── registro_analise_tese.txt
├── matriz_extracao_tese.csv
├── config_rag.py
├── documentos_inovação/      # 76 PDFs (BRASIL + 5 regiões, 42 IFs)
└── memoria_longo_prazo/      # ChromaDB persistente
```

## 🤝 Contribuição

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Add: nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional (CC BY-NC-SA 4.0).

---

**Desenvolvido por:** Jonatã França Bittencourt  
**Instituição:** IFBA - Campus Jacobina  
**Programa:** Doutorado em Ciências Sociais (PPGCS - UFBA)  
**Orientador:** Prof. Leonardo Fernandes Nascimento