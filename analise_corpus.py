# -*- coding: utf-8 -*-
"""
Módulo de análise de corpus.

Responsabilidades:
- Cache persistente de extração de PDFs (re-extração incremental por mtime),
  eliminando a re-parsing completa dos PDFs a cada inicialização do dashboard.
- NLP leve: normalização de texto, stopwords em português embutidas, tokenização.
- Cálculos para análise textual: contagem de termos, co-ocorrência, LDA (sklearn),
  frequência de temas (eixos da pesquisa) e concordância KWIC.
"""

from __future__ import annotations

import os
import re
import unicodedata
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "documentos_inovação"
CORPUS_CACHE = BASE_DIR / "corpus_cache.pkl"

REGIOES_VALIDAS = ["NORTE", "NORDESTE", "CENTRO OESTE", "SUDESTE", "SUL", "BRASIL"]

EIXOS_ANALISE = {
    "I. Fundamentos Epistemológicos e Saberes": {
        "keywords": [
            "tecnologia social", "tecnologias sociais", "saberes tradicionais",
            "conhecimento local", "conhecimento popular", "conhecimento indígena",
            "interculturalidade", "intercultural", "justiça epistêmica",
            "pluriversal", "pluriversalidade", "decolonial", "decolonialidade",
            "descolonial", "colonialidade", "epistemologia", "epistemológico",
            "epistemológica", "cosmovisão", "sociologia da inovação",
            "razão substantiva", "pensamento imitativo",
        ],
        "pilar": "Dimensão 1 · Fundamentos Epistemológicos (Quadro 1.5)",
        "referencia": "Justiça epistêmica, pluriversalidade e simetria de saberes; crítica ao "
                      "pensamento imitativo e à colonialidade do saber (Guerreiro Ramos).",
        "pergunta": "Quais formas de conhecimento são reconhecidas e priorizadas? A política "
                    "reforça o pensamento imitativo/eurocêntrico ou promove a simetria de saberes?",
    },
    "II. Atores, Redes e Capital Humano": {
        "keywords": [
            "sociedade civil", "movimento social", "movimentos sociais",
            "comunidade", "comunidades", "arranjo produtivo local",
            "arranjos produtivos locais", "economia solidária", "cooperativismo",
            "cooperativa", "participação", "parceria", "parcerias",
            "colaboração", "cocriação", "capital humano", "formação",
            "capacitação", "bolsa", "bolsas", "docentes", "pesquisadores",
            "pesquisador", "técnicos", "discentes", "recursos humanos",
        ],
        "pilar": "Dimensão 2 · Atores, Redes e Relações de Poder (Quadro 1.5) + eixo Capital Humano",
        "referencia": "Redes de tradução e Ponto de Passagem Obrigatório (Callon/TAR); inclusão de "
                      "sociedade civil, APLs, economia solidária e valorização dos atores e do capital humano.",
        "pergunta": "Quem participa da formulação e implementação? O instrumento centraliza poder "
                    "(PPO) ou distribui agência? Quem é incluído ou excluído das redes?",
    },
    "III. Relevância, Impacto e Inovação Social": {
        "keywords": [
            "inovação social", "impacto social", "bem-estar", "bem viver",
            "buen vivir", "sustentabilidade", "sustentável", "emancipação",
            "inclusão", "acessibilidade", "território", "territórios",
            "desenvolvimento social", "função social", "extensão",
            "solidariedade", "equidade",
        ],
        "pilar": "Dimensão 3 · Relevância e Impacto Social (Quadro 1.5) + eixo Inovação Social",
        "referencia": "Razão substantiva (Ramos), Bem Viver (Acosta), ponto de vista da sociedade "
                      "civil (Burawoy); impacto social, desenvolvimento social e distribuição equitativa.",
        "pergunta": "A política transcende o imperativo de mercado? Prioriza soluções para problemas "
                    "locais e a equidade na distribuição dos benefícios?",
    },
    "IV. Desenho Institucional e Governança": {
        "keywords": [
            "governança", "gestão", "conselho", "conselhos", "comitê", "comitês",
            "colegiado", "colegiados", "administração", "procuradoria",
            "decisão", "decisões", "estrutura organizacional", "planejamento",
            "orçamento", "transparência", "autonomia", "regimento",
            "burocracia", "racionalização", "deliberação", "tomada de decisão",
            "marco regulatório",
        ],
        "pilar": "Dimensão 4 · Desenho Institucional e Governança (Quadro 1.5) + eixo Governança",
        "referencia": "Racionalização burocrática (Weber), Código Técnico e Racionalização Subversiva "
                      "(Feenberg); governança participativa, transparência e autonomia.",
        "pergunta": "As normas operam como código técnico rígido ou abrem margem de manobra aos campi? "
                    "A tomada de decisão é transparente e participativa?",
    },
    "V. Contextualização e Desenvolvimento Regional": {
        "keywords": [
            "desenvolvimento regional", "vocação regional", "desenvolvimento endógeno",
            "endógeno", "inserção regional", "arranjo local", "arranjos locais",
            "arranjos produtivos locais", "especificidade", "contextualização",
            "vulnerabilidade", "vulnerabilidades", "economia local",
            "ecossistema", "ecossistemas", "pdi", "desenvolvimento local",
        ],
        "pilar": "Dimensão 5 · Contextualização e Desenvolvimento Regional (Quadro 1.5)",
        "referencia": "Desenvolvimento endógeno, especificidade regional e redução sociológica (Ramos); "
                      "fortalecimento de ecossistemas locais e redução de dependências históricas.",
        "pergunta": "A política responde ao contexto socioeconômico, cultural e histórico da região? "
                    "Fortalece ecossistemas locais ou reproduz dependências?",
    },
    "VI. Propriedade Intelectual e Racionalidade Instrumental": {
        "keywords": [
            "propriedade intelectual", "nit", "patente", "patentes", "marca", "marcas",
            "licenciamento", "royalties", "royalty", "software", "invenção",
            "invenções", "inovação tecnológica", "transferência de tecnologia",
            "escritório de inovação", "spin-off", "spin off",
            "empreendedorismo", "startup", "startups", "competitividade", "produtividade",
        ],
        "pilar": "Eixo tradicional — Propriedade Intelectual (contraponto da racionalidade instrumental)",
        "referencia": "Métricas de patentes, royalties, transferência de tecnologia e competitividade; "
                      "caracteriza a racionalidade instrumental (Weber) e o viés hegemônico (Feenberg).",
        "pergunta": "A política prioriza métricas econômico-tecnológicas? Como a inovação tecnológica é "
                    "posicionada frente aos demais domínios?",
    },
}

# --- Processamento textual leve ---

_STOPWORDS_CRUAS = set("""
a ao aos as ate até com da das de do dos e em na nas no nos o os ou para pela pelas
pelo pelos por que se sem sob um uma umas uns
abaixo acerca ainda alem alto ambos ano anos ante antes apos apesar aquela aquelas
aquele aqueles aquilo assim ate atraves cada caso causa certa certas certo certos
cima com conforme contra contudo cuja cujas cujo cujos daquela daquelas daquele
daqueles depois desde desprovido desta destas deste destes dessa dessas desse desses
detras deus diante directamente dispõe dispoe dispõem dispoem diz dizer dois durante
e ela elas ela ele eles em entao entre era eram essa essas esse esses esta este estes
esteja estejam estamos estando estara estarao estavam este estado estes estou etc eu
exceto existem existiu fazer feita feito feitos fez ficar fim final forma foram fora
ha haja hajam haver haviam hei hoje homem hora horas ja jamais juntamente la lhe lhes
lo logo mais mas me mesma mesmas mesmo mesmos meu meus minha minhas muito muitos na
naquele naquela naqueles naquilo nas nem nenhum nenhuma ninguém ninguem no nos nossa
nossas nosso nossos num numa não nunca onde outrem outros outra outras outro para
parece parecem parte partir por porque porém pra pelas pensao perante pode podem poder
poderia poderiam pois pela ponto pontos por pouco porem povo primeiro primeira
primeiros propria proprias proprio proprios propro prioia quai qual quando quanto
quantos quarta quatr que quem quer quere razão razao mesma se seja sejam sem sempre
sendo ser sera serem serei seremos sera serão serto seu seus simples sobre sob
sobrenome soja somos sou souceis su cada sua suas tao também tambem ter tem teoria
teve tinha tive todos toda todas todo tras tres tua tuas tudo tás u ou a um um esta
ultima ultimo umas uns um uma uos vais vai varias varios vem vezes vez ser terei
teremos teria terse vai verso via vos vosso vossa voces voce voce vi tera podera
poderao pode podiam possa possam podemos pomos proceder pro posse sao serao seja
serem serei seremos seria serei serao seremos ser tenha tenhamo ter test test
artigo artigos art art arts caput inciso incisos paragrafo paragrafos lei leis
decreto decretos portaria portarias resolucao resolucaoes resolucões regulamento
regulamentos norma normas cf conforme disposto considera considerando estabelece
estabelecimento fins efeitos forma conforme adotar adotado aplicacao aplicação rede
licença licenca consoante observado no qual nos quais na qual nas quais
""".split())

def remover_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    sem_acento = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return sem_acento


# Palavras-chave normalizadas (minúsculas e sem acento) por eixo, para casamento no texto.
_KW_NORMALIZADAS = {
    eixo: sorted({remover_acentos(palavra) for palavra in dados["keywords"]})
    for eixo, dados in EIXOS_ANALISE.items()
}


def _kw_match(palavra: str, texto: str) -> bool:
    """Casamento por subconjunto no texto normalizado; tokens curtos exigem fronteira de palavra."""
    if len(palavra) <= 4:
        return re.search(r"(?<![a-z0-9])" + re.escape(palavra) + r"(?![a-z0-9])", texto) is not None
    return palavra in texto


STOPWORDS_PT = {remover_acentos(w) for w in _STOPWORDS_CRUAS}
_STOPWORDS_EXTRA = {
    "nao", "sim", "aho", "so", "sera", "serao", "estara", "estarao", "tambem",
    "sao", "nesta", "neste", "nestas", "nestes", "nessa", "nesse", "1",
    # Conjunções, preposições e conectivos em falta
    "como", "enquanto", "entre", "durante", "mediante", "sobre", "sobretudo",
    "salvo", "todavia", "portanto", "entretanto", "tampouco", "embora",
    "porquanto", "senao", "ora", "afim", "ante", "diante", "avante", "aquando",
    "isso", "isto", "tal", "tais", "qualquer", "quaisquer", "demais",
    "apenas", "somente", "sequer", "pouco", "poucos", "pouca", "poucas",
    # Contrações e locuções prepositivas
    "dum", "duma", "duns", "dumas", "numa", "nuns", "numas",
    "pra", "pro", "pros", "pras", "noutro", "noutra", "noutros", "noutras",
    # Formas verbais funcionais recorrentes
    "tendo", "houve", "houver", "havido", "sido",
    # Siglas de instituições (IFs), de órgãos e ruído de documentos — não são conteúdo
    # analítico e não devem aglutinar tópicos LDA, nuvem, co-ocorrência ou ranking.
    "ifb", "ifpb", "ifro", "ifto", "ifmt", "ifap", "ifnmg", "ifpe", "ifrj",
    "ifba", "ifrs", "ifce", "ifrr", "ifes", "ifpr", "ifsul", "ifpi", "ifal",
    "ifmg", "ifms", "ifpa", "ifsc", "ifsp", "iffar", "ifam", "ifflu", "ifs",
    "iffs", "ifg", "ifc", "ifr", "ifn", "ifm", "ife", "ifp",
    "ifsuldeminas", "ifsudestemg", "ifsertao", "ifgoiano", "ifmtech",
    "cefet", "cefetes", "ict", "icts", "nit", "pdi", "ead", "conif", "forplad",
    "pigete", "capq", "cnpq", "mec", "fndct", "fap", "ufba", "ufb", "ieps",
    "www", "http", "https", "edu", "gov", "ccivil", "planalto", "htm", "html",
    "govbr", "orgbr",
    # Siglas de sistema e fragmentos de extração PDF (ex.: "ins\ntuto" → "ins", "tuto")
    "sei", "ins", "tuto",
    # Jargão legislativo (fórmulas de texto de lei sem valor analítico) e meses de
    # referência legal — saturam a tokenização e criam tópicos artificiais no LDA.
    # ("marco" é intencionalmente preservado: aparece em "marco regulatório").
    "incluido", "redacao", "provisoria", "vigencia", "revogado", "dada", "vide",
    "dezembro", "janeiro", "fevereiro", "abril", "julho", "agosto", "setembro",
    # Fragmentos de palavras partidas por extração de PDF (pedaços sem significado).
    "vidades", "tucional", "instit", "atividad", "sertaoa", "educa8vo",
    "educac", "educao", "insa", "pireitor",
}
STOPWORDS_PT |= _STOPWORDS_EXTRA


_RUIDO_TOKEN = re.compile(r"^(?:[gx]\d{1,4}|[ivxlcdm]{2,5}|z\d{1,4}|[a-z]\d{3,})$")


def normalizar_texto(texto: str) -> list[str]:
    """Tokeniza um texto: minúsculas, sem acentos, só alfanuméricos, sem stopwords."""
    sem_acento = remover_acentos(texto)
    tokens = re.sub(r"[^a-z0-9 ]", " ", sem_acento).split()
    return [
        t for t in tokens
        if len(t) >= 3 and not t.isdigit()
        and t not in STOPWORDS_PT
        and not _RUIDO_TOKEN.fullmatch(t)
    ]


def texto_filtrado(texto: str) -> str:
    """Texto normalizado com um espaço entre tokens (para LDA/wordcloud)."""
    return " ".join(normalizar_texto(texto))


# --- Metadados de caminho (barato, sem abrir PDFs) ---

def _extrair_ano(nome_arquivo):
    busca = re.search(r"(19|20)\d{2}", nome_arquivo)
    return int(busca.group()) if busca else None


def _metadados_caminho(caminho: Path) -> dict:
    partes = Path(os.path.normpath(str(caminho.parent))).parts
    regiao, uf, if_nome, tipo = "Outros", "S/D", "Não Identificado", "Política Institucional"
    partes_upper = [p.upper().replace("-", " ") for p in partes]

    if "BRASIL" in partes_upper:
        regiao, uf, if_nome, tipo = "Nacional", "BR", "N/A", "Legislação Federal"
    else:
        for r in REGIOES_VALIDAS:
            if r in partes_upper:
                idx = partes_upper.index(r)
                regiao = partes[idx]
                if len(partes) > idx + 1:
                    uf = partes[idx + 1]
                if len(partes) > idx + 2:
                    if_nome = partes[idx + 2]
                break

    return {"regiao": regiao, "uf": uf, "instituto": if_nome, "tipo": tipo}


def escanear_estrutura() -> list[dict]:
    """Lista os PDFs com metadados derivados apenas do caminho/nome (sem ler conteúdo)."""
    if not DATA_DIR.exists():
        return []
    estrutura = []
    for caminho in sorted(DATA_DIR.rglob("*.pdf")):
        meta = _metadados_caminho(caminho)
        meta.update({
            "rel": caminho.relative_to(DATA_DIR).as_posix(),
            "caminho": caminho,
            "mtime": caminho.stat().st_mtime,
            "tamanho_mb": round(caminho.stat().st_size / 1024 / 1024, 2),
            "arquivo": caminho.name,
            "ano": _extrair_ano(caminho.name),
        })
        estrutura.append(meta)
    return estrutura


def resumo_estrutura() -> tuple[list, list, list]:
    """UFs, IFs e anos disponíveis — para filtros sem processar PDFs."""
    estrutura = escanear_estrutura()
    ufs = sorted({e["uf"] for e in estrutura})
    ifs = sorted({e["instituto"] for e in estrutura if e["tipo"] != "Legislação Federal"})
    anos = sorted({a for a in (e["ano"] for e in estrutura) if a is not None})
    return ufs, ifs, anos


def tipos_disponiveis() -> list[str]:
    """Tipos de documento presentes no corpus (para filtro de análise)."""
    return sorted({e["tipo"] for e in escanear_estrutura() if e["tipo"]})


# --- Extração de texto com cache persistente ---

def _extrair_um_item(item):
    rel, caminho, meta, ano = item
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(caminho))
        paginas = loader.load()
        return rel, {
            "arquivo": caminho.name,
            "regiao": meta["regiao"],
            "uf": meta["uf"],
            "instituto": meta["instituto"],
            "tipo": meta["tipo"],
            "ano": ano,
            "paginas": [p.page_content or "" for p in paginas],
        }
    except Exception as e:
        print(f"[ERRO] Falha ao processar {caminho.name}: {e}")
        return rel, None


def _salvar_cache(cache):
    with open(CORPUS_CACHE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def _carregar_cache_disco() -> dict:
    if not CORPUS_CACHE.exists():
        return {"index": {}, "docs": {}}
    try:
        with open(CORPUS_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {"index": {}, "docs": {}}


@lru_cache(maxsize=1)
def carregar_corpus() -> dict:
    """Carrega todos os PDFs com re-extração incremental (só PDFs novos/alterados).

    Os tokens e o texto normalizado são computados uma única vez por processo e
    anexados ao dict em memória (não são persistidos no disco para não inflar o
    cache), de modo que todas as análises reutilizam a mesma tokenização.
    """
    estrutura = escanear_estrutura()
    cache = _carregar_cache_disco()
    index_antigo = cache.get("index", {})
    docs = cache.get("docs", {})

    atuais = {e["rel"]: e for e in estrutura}

    removidos = [rel for rel in index_antigo if rel not in atuais]
    for rel in removidos:
        index_antigo.pop(rel, None)
        docs.pop(rel, None)

    a_processar = []
    for rel, e in atuais.items():
        if index_antigo.get(rel) != e["mtime"]:
            a_processar.append((rel, e["caminho"], _metadados_caminho(e["caminho"]), e["ano"]))

    if a_processar:
        novos = {}
        max_workers = min(8, max(2, len(a_processar)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for rel, rec in pool.map(_extrair_um_item, a_processar):
                if rec is not None:
                    novos[rel] = rec
        if novos:
            docs.update(novos)
        for rel, e in atuais.items():
            index_antigo[rel] = e["mtime"]
        _salvar_cache({"index": index_antigo, "docs": docs})

    # Pré-computar tokens e texto normalizado uma única vez (em memória).
    for rec in docs.values():
        if "tokens_ok" not in rec:
            texto = "\n".join(rec["paginas"])
            rec["_texto"] = texto
            rec["_tokens"] = normalizar_texto(texto)
            rec["_texto_limpo"] = " ".join(rec["_tokens"])
            rec["tokens_ok"] = True

    return docs


def carregar_docs_paginas() -> list[Document]:
    """Flatten do corpus em documentos LangChain (uma por página) para indexação."""
    corpus = carregar_corpus()
    saida = []
    for rec in corpus.values():
        base = {"regiao": rec["regiao"], "uf": rec["uf"], "instituto": rec["instituto"],
                "ano": rec["ano"], "tipo": rec["tipo"], "arquivo": rec["arquivo"]}
        for i, texto in enumerate(rec["paginas"], 1):
            meta = dict(base)
            meta["pagina"] = i
            saida.append(Document(page_content=texto, metadata=meta))
    return saida


def _filtros_como_chave(filtros: dict | None) -> tuple:
    f = filtros or {}
    return (
        tuple(sorted(f.get("ufs", []))),
        tuple(sorted(f.get("ifs", []))),
        tuple(sorted(f.get("anos", []))),
        tuple(sorted(f.get("regioes", []))),
        tuple(sorted(f.get("tipos", []))),
        tuple(sorted(f.get("dimensoes", []))),
    )


def _dims_do_documento(rec: dict) -> set[str]:
    """Conjunto de eixos/dimensões analíticas mencionadas pelo documento (Quadro 1.5)."""
    texto = remover_acentos(rec.get("_texto") or "")
    if not texto:
        return set()
    atingidas = set()
    for eixo, palavras in _KW_NORMALIZADAS.items():
        if any(_kw_match(p, texto) for p in palavras):
            atingidas.add(eixo)
    return atingidas


@lru_cache(maxsize=16)
def _carregar_documentos_analise_cache(chave: tuple) -> list[dict]:
    """Corpus no nível de documento com tokens pré-computados, cacheado por filtros."""
    corpus = carregar_corpus()
    filtros = {
        "ufs": list(chave[0]), "ifs": list(chave[1]), "anos": list(chave[2]),
        "regioes": list(chave[3]), "tipos": list(chave[4]), "dimensoes": list(chave[5]),
    }
    linhas = []
    for rec in corpus.values():
        dims = _dims_do_documento(rec) if filtros["dimensoes"] else None
        if not _passa_filtro(rec, filtros, dims):
            continue
        linhas.append({
            "texto": rec["_texto"],
            "tokens": rec["_tokens"],
            "texto_limpo": rec["_texto_limpo"],
            "regiao": rec["regiao"],
            "uf": rec["uf"],
            "instituto": rec["instituto"],
            "tipo": rec["tipo"],
            "ano": rec["ano"],
            "arquivo": rec["arquivo"],
            "dims": dims,
        })
    return linhas


def carregar_documentos_analise(filtros: dict | None = None) -> list[dict]:
    """Corpus no nível de documento (texto completo + tokens + metadados), com filtros opcionais."""
    return _carregar_documentos_analise_cache(_filtros_como_chave(filtros))


def _passa_filtro(rec: dict, filtros: dict, dims: set[str] | None = None) -> bool:
    def ok(valor, lista):
        return not lista or valor in lista
    if not (ok(rec["uf"], filtros.get("ufs", []))
            and ok(rec["instituto"], filtros.get("ifs", []))
            and ok(rec["regiao"], filtros.get("regioes", []))
            and ok(rec["tipo"], filtros.get("tipos", []))
            and ok(rec["ano"], filtros.get("anos", []))):
        return False
    dims_sel = filtros.get("dimensoes", [])
    if dims_sel:
        dims = dims if dims is not None else _dims_do_documento(rec)
        return bool(dims & set(dims_sel))
    return True


# --- Análises ---

def _tokens_de(d):
    return d.get("tokens") or normalizar_texto(d["texto"])


def contagem_termos(documentos: list[dict]) -> Counter:
    cont = Counter()
    for d in documentos:
        cont.update(_tokens_de(d))
    return cont


def top_termos(documentos: list[dict], n: int = 30) -> list[tuple[str, int]]:
    return contagem_termos(documentos).most_common(n)


def coocorrencia(documentos: list[dict], termos: list[str], limiar: int = 1):
    """Co-ocorrência por documento entre os termos. Retorna ocorrências e arestas."""
    ocorrencias = Counter()
    pares = defaultdict(int)
    conjunto = set(termos)
    for d in documentos:
        toks = set(_tokens_de(d)) & conjunto
        ocorrencias.update(toks)
        lista = sorted(toks)
        for i in range(len(lista)):
            for j in range(i + 1, len(lista)):
                pares[(lista[i], lista[j])] += 1
    arestas = [(a, b, n) for (a, b), n in pares.items() if n >= limiar]
    return dict(ocorrencias), arestas


def grafo_coocorrencia(ocorrencias: dict, arestas: list, limiar: int = 1, seed: int = 42):
    """Constrói os dados do grafo (nós, arestas, posições) para plotagem em Plotly."""
    import networkx as nx
    G = nx.Graph()
    for termo, n in ocorrencias.items():
        if n >= 1:
            G.add_node(termo, weight=int(n))
    for a, b, n in arestas:
        if a in G and b in G and n >= limiar:
            G.add_edge(a, b, weight=int(n))
    if G.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(G, seed=seed, k=0.55, iterations=60)
    nodes_termos = list(G.nodes())
    nodes_maior = max(G.nodes[n].get("weight", 1) for n in nodes_termos)
    nodes_maior = nodes_maior or 1
    nodes_tamanho = [8 + 22 * (G.nodes[n].get("weight", 1) / nodes_maior) for n in nodes_termos]
    edges = [(a, b, G.edges[a, b]["weight"]) for a, b in G.edges()]
    edges_maior = max((w for _, _, w in edges), default=0) or 1
    return {
        "nodes": nodes_termos,
        "pos": pos,
        "tamanhos": nodes_tamanho,
        "pesos": [G.nodes[n].get("weight", 1) for n in nodes_termos],
        "edges": edges,
        "edges_larguras": [1 + 6 * (w / edges_maior) for _, _, w in edges],
        "edges_pesos": [w for _, _, w in edges],
    }


def _top_palavras_topicos(modelo, nomes_features, n):
    resultado = []
    for idx, topic in enumerate(modelo.components_):
        top = topic.argsort()[:-n - 1:-1]
        resultado.append([(nomes_features[i], round(float(topic[i]), 2)) for i in top])
    return resultado


def lda_topicos(documentos: list[dict], n_topics: int = 5, max_features: int = 2000):
    """Ajusta LDA (scikit-learn). Retorna None se o corpus for pequeno demais."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    textos = [d.get("texto_limpo") or texto_filtrado(d["texto"]) for d in documentos]
    if len(textos) < 5:
        return None
    vetorizador = CountVectorizer(
        max_features=max_features, min_df=2, max_df=0.95,
        token_pattern=r"\b[a-z0-9]{3,}\b",
        stop_words=sorted(STOPWORDS_PT),
    )
    X = vetorizador.fit_transform(textos)
    if X.shape[1] < 5:
        return None
    n_topics = max(2, n_topics)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=20,
        learning_method="online", n_jobs=1,
    )
    matriz = lda.fit_transform(X)
    nomes = vetorizador.get_feature_names_out().tolist()
    return {
        "vetorizador": vetorizador,
        "lda": lda,
        "matriz_doc_topic": matriz,
        "nomes_features": nomes,
        "topicos": _top_palavras_topicos(lda, nomes, 12),
    }


def _textos_normalizados(documentos: list[dict]) -> list[str]:
    return [remover_acentos(d["texto"]) for d in documentos]


def frequencia_temas(documentos: list[dict]) -> pd.DataFrame:
    """Cobertura e profundidade de cada dimensão analítica no corpus filtrado.

    Profundidade média = fração média de sinais distintos da dimensão presentes por
    documento (peso por sinal, não apenas frequência bruta de menções).
    """
    import pandas as pd
    total = max(len(documentos), 1)
    textos = _textos_normalizados(documentos)
    linhas = []
    for eixo, dados in EIXOS_ANALISE.items():
        palavras = _KW_NORMALIZADAS[eixo]
        n_sinais = len(palavras)
        n_docs = 0
        ocorrencias = 0
        sinais = set()
        prof_soma = 0.0
        for texto in textos:
            hits = [p for p in palavras if _kw_match(p, texto)]
            if hits:
                n_docs += 1
                sinais.update(hits)
                ocorrencias += sum(texto.count(p) for p in hits)
                prof_soma += len(hits) / n_sinais
        linhas.append({
            "Eixo": eixo,
            "Documentos": n_docs,
            "Sinais": len(sinais),
            "Ocorrências": ocorrencias,
            "Profundidade média (%)": round(prof_soma / n_docs * 100, 1) if n_docs else 0.0,
        })
    df = pd.DataFrame(linhas)
    df["% documentos"] = (df["Documentos"] / total * 100).round(1)
    return df


def frequencia_temas_por(documentos: list[dict], categoria: str) -> pd.DataFrame:
    """Frequência por dimensão analítica agrupada por UF, região ou ano."""
    import pandas as pd
    linhas = []
    for d in documentos:
        texto = remover_acentos(d["texto"])
        for eixo, dados in EIXOS_ANALISE.items():
            if any(_kw_match(p, texto) for p in _KW_NORMALIZADAS[eixo]):
                linhas.append({"Eixo": eixo, categoria: d.get(categoria, "S/d")})
    if not linhas:
        return pd.DataFrame(columns=["Eixo", categoria])
    df = pd.DataFrame(linhas)
    return df.groupby([categoria, "Eixo"]).size().reset_index(name="Documentos")


def sinais_todas_dimensoes(documentos: list[dict]) -> dict[str, pd.DataFrame]:
    """Contagem por sinal analítico de todas as dimensões — passagem única pelo corpus.

    Recusa a contagem por contagem; usa a mesma semântica de correspondência tolerante
    a acentos (_kw_match) das demais medidas. Retorna {eixo: DataFrame}.
    """
    import pandas as pd
    textos = _textos_normalizados(documentos)
    res: dict[str, pd.DataFrame] = {}
    for eixo, palavras in _KW_NORMALIZADAS.items():
        linhas = []
        for palavra in palavras:
            n_docs = 0
            ocorrencias = 0
            for texto in textos:
                if _kw_match(palavra, texto):
                    n_docs += 1
                    ocorrencias += texto.count(palavra)
            linhas.append({"Sinal": palavra, "Documentos": n_docs, "Ocorrências": ocorrencias})
        df = pd.DataFrame(linhas)
        df["Situação"] = df["Documentos"].apply(lambda n: "Presente" if n > 0 else "Ausente")
        res[eixo] = df
    return res


def sinais_por_dimensao(documentos: list[dict], eixo: str) -> pd.DataFrame:
    """Contagem por sinal analítico (palavra-chave) de uma dimensão — apoio à checagem de lacunas."""
    todas = sinais_todas_dimensoes(documentos)
    df = todas.get(eixo)
    if df is None:
        import pandas as pd
        return pd.DataFrame(columns=["Sinal", "Documentos", "Ocorrências", "Situação"])
    return df


def kwic(documentos: list[dict], termo: str, janela: int = 60, maximo: int = 60) -> list[dict]:
    """Concordância KWIC: ocorrências do termo com contexto (tolerante a acentos)."""
    if not termo or not termo.strip():
        return []
    termo_limpo = termo.strip().lower()
    padrao = "".join(re.escape(c) + r"[\u0300-\u036f]*" for c in termo_limpo)
    regex = re.compile(padrao, re.IGNORECASE)
    linhas = []
    contador = 0
    for d in documentos:
        texto = d["texto"]
        for m in regex.finditer(texto):
            ini = max(0, m.start() - janela)
            fim = min(len(texto), m.end() + janela)
            antes = texto[ini:m.start()].strip()
            depois = texto[m.end():fim].strip()
            linhas.append({
                "arquivo": d["arquivo"], "uf": d["uf"], "instituto": d["instituto"],
                "ano": d["ano"], "antes": antes, "termo": m.group(0), "depois": depois,
            })
            contador += 1
            if contador >= maximo:
                return linhas
    return linhas


# --- Série histórica (volume documental por ano de aprovação) ---

def serie_historica(documentos: list[dict]) -> dict:
    """Volume documental por ano de aprovação, clusterizado por tipo e por região.

    Retorna dict com DataFrames prontos para plotagem:
      - "por_tipo":  anos × tipo (Legislação Federal / Política Institucional)
      - "por_regiao": anos × região (apenas políticas institucionais, que têm UF/região)
      - "anos":       todos os anos presentes (ordenados)
    """
    import pandas as pd
    tabela = []
    for d in documentos:
        if d.get("ano") is None:
            continue
        tabela.append({"ano": int(d["ano"]), "tipo": d.get("tipo", "S/D"),
                       "regiao": d.get("regiao", "S/D")})
    if not tabela:
        return {"por_tipo": pd.DataFrame(columns=["ano", "tipo", "Documentos"]),
                "por_regiao": pd.DataFrame(columns=["ano", "regiao", "Documentos"]),
                "anos": []}
    df = pd.DataFrame(tabela)
    anos = sorted(df["ano"].unique().tolist())

    por_tipo = (df.groupby(["ano", "tipo"]).size().reset_index(name="Documentos")
                .rename(columns={"tipo": "Cluster"}))
    por_tipo["ano"] = por_tipo["ano"].astype(int)

    inst = df[df["tipo"] != "Legislação Federal"]
    if inst.empty:
        por_regiao = pd.DataFrame(columns=["ano", "regiao", "Documentos"])
    else:
        por_regiao = (inst.groupby(["ano", "regiao"]).size().reset_index(name="Documentos")
                      .rename(columns={"regiao": "Região"}))
        por_regiao["ano"] = por_regiao["ano"].astype(int)
    return {"por_tipo": por_tipo, "por_regiao": por_regiao, "anos": anos}


# --- Bases legais nacionais citadas nos documentos ---

_BASE_LEGAL_NOME = {
    "lei no 8.112": "Lei nº 8.112/1990 (Regime Jurídico)",
    "lei no 8.212": "Lei nº 8.212/1991 (Custeio da Seguridade)",
    "lei no 8.248": "Lei nº 8.248/1991 (Lei de Informática)",
    "lei no 8.666": "Lei nº 8.666/1993 (Licitações e Contratos)",
    "lei no 8.745": "Lei nº 8.745/1993 (Contratação Temporária)",
    "lei no 8.958": "Lei nº 8.958/1994 (Fundações de Apoio)",
    "lei no 9.250": "Lei nº 9.250/1995 (IRPF)",
    "lei no 9.279": "Lei nº 9.279/1996 (Propriedade Industrial)",
    "lei no 9.456": "Lei nº 9.456/1997 (Cultivares)",
    "lei no 9.609": "Lei nº 9.609/1998 (Lei do Software)",
    "lei no 9.610": "Lei nº 9.610/1998 (Direitos Autorais)",
    "lei no 10.168": "Lei nº 10.168/2000 (CIDE Tecnologia)",
    "lei no 10.973": "Lei nº 10.973/2004 (Incentivo à Inovação)",
    "lei no 11.196": "Lei nº 11.196/2005 (Lei do Bem)",
    "lei no 11.314": "Lei nº 11.314/2006",
    "lei no 11.484": "Lei nº 11.484/2007",
    "lei no 11.184": "Lei nº 11.184/2005",
    "lei no 11.892": "Lei nº 11.892/2008 (Lei dos IFs)",
    "lei no 12.772": "Lei nº 12.772/2012 (Carreira EBTT)",
    "lei no 13.123": "Lei nº 13.123/2015 (Biodiversidade)",
    "lei no 13.243": "Lei nº 13.243/2016 (Marco Legal da Inovação)",
    "lei no 13.246": "Lei nº 13.246/2016 (Convertida da MP 690)",
    "lei no 14.133": "Lei nº 14.133/2021 (Nova Lei de Licitações)",
    "lei no 11.091": "Lei nº 11.091/2005 (Plano de Carreira)",
    "lei no 11.784": "Lei nº 11.784/2008 (Reestruturação de Carreiras)",
    "lei no 13.146": "Lei nº 13.146/2015 (Estatuto da Pessoa com Deficiência)",
    "lei complementar no 95": "LC nº 95/1998 (Elaboração de Leis)",
    "lei complementar no 123": "LC nº 123/2006 (Simples Nacional)",
    "lei complementar no 182": "LC nº 182/2021 (Marco Legal das Startups)",
    "lei do bem": "Lei do Bem (nº 11.196/2005)",
    "marco legal das startups": "LC nº 182/2021 (Marco Legal das Startups)",
    "marco legal da inovacao": "Lei nº 13.243/2016 (Marco Legal da Inovação)",
    "lei de informatica": "Lei de Informática (nº 8.248/1991)",
    "lei do software": "Lei do Software (nº 9.609/1998)",
    "lei de incentivo a inovacao": "Lei nº 10.973/2004 (Incentivo à Inovação)",
    "decreto no 2.553": "Decreto nº 2.553/1998 (Software)",
    "decreto no 5.205": "Decreto nº 5.205/2004 (Lei 10.973)",
    "decreto no 5.563": "Decreto nº 5.563/2005 (Lei 10.973)",
    "decreto no 7.423": "Decreto nº 7.423/2010 (Regulamenta Lei 11.892)",
    "decreto no 8.539": "Decreto nº 8.539/2015 (Processo Eletrônico)",
    "decreto no 9.283": "Decreto nº 9.283/2018 (Regulamentação do Marco Legal)",
    "decreto no 10.534": "Decreto nº 10.534/2020 (PNI)",
    "decreto no 12.002": "Decreto nº 12.002/2024",
    "decreto no 6.759": "Decreto nº 6.759/2009 (Impostos/Tributação)",
}

_BASE_LEGAL_REGEX = re.compile(
    r"(lei\s+complementar\s+no\s+\d{1,3}(?:\.\d{3})*(?:/\d{2,4})?|"
    r"lei\s+no\s+\d{1,3}(?:\.\d{3})*(?:/\d{2,4})?|"
    r"decreto\s+no\s+\d{1,3}(?:\.\d{3})*(?:/\d{2,4})?|"
    r"mp\s+no\s+\d{1,3}(?:\.\d{3})*(?:/\d{2,4})?|"
    r"lei\s+do\s+bem|"
    r"marco\s+legal\s+da\s+startups|"
    r"marco\s+legal\s+da\s+inovacao|"
    r"lei\s+de\s+informatica|"
    r"lei\s+do\s+software|"
    r"lei\s+de\s+incentivo\s+a\s+inovacao)",
    re.IGNORECASE,
)


def _limpar_base_legal(texto: str) -> str:
    """Normaliza o texto para casamento robusto (sem acentos, quebras colapsadas).

    - Colapsa quebras de linha/espaços.
    - Normaliza "n. / nº / n°" → "no".
    - Corrige micro-quebras de extração de PDF dentro do número (ex.: "9.2 79" → "9.279").
    - Remove dígito de numeração solto que o PDF insere antes do número da base
      (ex.: "lei no 2 10.973" → "lei no 10.973").
    """
    t = re.sub(r"\s+", " ", remover_acentos(texto))
    t = re.sub(r"\bn\s*[oº°]?\s*(?=\d)", "no ", t)
    # remove dígito solto de numeração entre "no" e o número da base legal
    t = re.sub(r"\b((?:lei|decreto|mp)\s+no\s+)\d\s+(?=\d)", r"\1", t)
    # só remove espaço dentro do grupo decimal (número com ponto: "9.2 79")
    t = re.sub(r"(\d\.\d{1,3})\s+(?=\d)", r"\1", t)
    return t


def _nome_base_legal(chave: str) -> str:
    """Mapeia uma chave detectada para o nome canônico da base legal.

    A chave detectada pode variar na formatação (nº/Nº, /ano, espaços). A canonização
    normaliza "lei no X" e "lei no X/YY" para a mesma entrada do dicionário.
    """
    chave = re.sub(r"/\d{2,4}$", "", chave).lower().strip()
    if chave in _BASE_LEGAL_NOME:
        return _BASE_LEGAL_NOME[chave]
    num = re.sub(r"[^\d]", "", chave)
    return None if len(num) < 4 else f"{chave.split()[0].title()} nº {'.'.join(num[i:i+3] for i in range(0,len(num),3))}"


def bases_legais_citadas(documentos: list[dict]) -> dict:
    """Detecta as bases legais nacionais citadas em cada documento.

    Retorna: {"por_doc": {arquivo: [nomes]}, "por_base": {nome: n_docs},
              "todas": [nomes] (ordenadas por recência de código)}
    """
    from collections import Counter
    por_doc = {}
    base_por_doc = Counter()
    for d in documentos:
        if d.get("tipo") == "Legislação Federal":
            continue
        t = _limpar_base_legal(d.get("texto") or "")
        chaves = set(m.group(1).lower().strip() for m in _BASE_LEGAL_REGEX.finditer(t))
        nomes = sorted({n for c in chaves if (n := _nome_base_legal(c)) is not None})
        if nomes:
            por_doc[d.get("arquivo", "?")] = nomes
            base_por_doc.update(nomes)
    todas = sorted(base_por_doc.keys())
    return {"por_doc": por_doc, "por_base": dict(base_por_doc), "todas": todas}


def tabela_base_legal_por_instituicao(documentos: list[dict]) -> "pd.DataFrame":
    """Tabela UF × Instituição × Documentos vinculados × Bases legais citadas.

    Uma linha por documento institucional; inclui a UF/instituição e as bases citadas.
    """
    import pandas as pd
    dados = bases_legais_citadas(documentos)
    linhas = []
    for d in documentos:
        if d.get("tipo") == "Legislação Federal":
            continue
        arquivo = d.get("arquivo", "?")
        bases = dados["por_doc"].get(arquivo, [])
        linhas.append({
            "UF": d.get("uf", "S/D"),
            "Região": d.get("regiao", "S/D"),
            "Instituição": d.get("instituto", "S/D"),
            "Documento": arquivo,
            "Ano": d.get("ano"),
            "Nº bases citadas": len(bases),
            "Bases legais citadas": "; ".join(bases),
        })
    if not linhas:
        return pd.DataFrame(columns=["UF", "Região", "Instituição", "Documento",
                                     "Ano", "Nº bases citadas", "Bases legais citadas"])
    df = pd.DataFrame(linhas)
    df = df.sort_values(["Região", "UF", "Instituição", "Ano"]).reset_index(drop=True)
    return df