from langchain_ollama import ChatOllama, OllamaEmbeddings


llm = ChatOllama(model="llama3")
try:
    response = llm.invoke("O que é uma política de inovação?")
    print("Conexão com LLM: OK!")
except Exception as e:
    print(f"Erro no LLM: {e}")


embeddings = OllamaEmbeddings(model="nomic-embed-text")
try:
    vector = embeddings.embed_query("Inovação tecnológica")
    print(f"Conexão com Embeddings: OK! (Vetor gerado com {len(vector)} dimensões)")
except Exception as e:
    print(f"Erro nos Embeddings: {e}")





