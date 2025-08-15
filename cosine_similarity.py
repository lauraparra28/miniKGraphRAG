from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Obtener la API Key
api_key = os.getenv("OPENAI_API_KEY")

# ==========================
# 1) Conexión a Neo4j
# ==========================
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "diripar8$"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
print("✅ Successfully connection to Neo4j Graph Database")

# -----------------------------
# Cargar embeddings Node2Vec
# -----------------------------
def extract_node2vec_embeddings():
    query = """
    MATCH (n)
    WHERE n.embedding IS NOT NULL
    RETURN elementId(n) AS nodeElementId, n.embedding AS embedding
    ORDER BY nodeElementId
    """
    with driver.session() as session:
        result = session.run(query)
        node_ids = []
        embeddings = []
        for record in result:
            node_ids.append(record["nodeElementId"])
            embeddings.append(record["embedding"])
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return node_ids, embeddings_array

node_ids, node2vec_embeddings = extract_node2vec_embeddings()
print("✅ Node2Vec loaded:", len(node_ids), getattr(node2vec_embeddings, "shape", None))
print("✅ Embeddings Node2Vec cargados:", node2vec_embeddings.shape)

# -----------------------------
# Modelo de embeddings de texto
# -----------------------------
text_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_dim_text = text_model.get_sentence_embedding_dimension()
# -----------------------------
# 3) Utilidades
# -----------------------------
def ensure_list_of_str(x):
    """Normaliza la propiedad Neo4j para obtener siempre lista[str]."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    # si es lista, filtrar elementos no string
    return [str(t) for t in x if t]

def encode_node_texts(texts_list):
    """
    texts_list: list[list[str]]
    Devuelve un array (N, D) promediando el embedding de cada nodo sobre sus textos.
    """
    embeddings = np.zeros((len(texts_list), embeddings_dim_text), dtype=np.float32)
    for i, texts in enumerate(texts_list):
        if not texts:
            continue
        # encode devuelve (k, D) para k textos
        e = text_model.encode(texts)
        if e.ndim == 1:
            embeddings[i] = e
        else:
            embeddings[i] = e.mean(axis=0)
    return embeddings

# -----------------------------
# 4) Función de búsqueda híbrida
# -----------------------------
def hybrid_rag_search (user_query: str, top_k: int = 5, alpha: float = 0.5, beta: float = 0.5, SEED_M: int = 15):
    
    """
    Realiza una búsqueda híbrida combinando similitud semántica de texto
    con Node2Vec en el grafo.
    
    Estrategia para el componente de grafo:
    - Usa los top-M nodos por similitud textual como "semillas".
    - Promedia sus vectores Node2Vec para obtener un vector de consulta en el espacio del grafo.
    - Calcula similitud coseno de cada nodo con ese vector.

    Args:
        user_question (str): Pregunta del usuario.
        top_k (int): Número de nodos top a retornar.
        alpha (float): Peso para la similitud de texto.
        beta (float): Peso para la similitud del grafo.

    Returns:
        context: string con texto concatenado de los top_k.
        top_items: lista de dicts con {node_id, label, text, score_text, score_graph, score_final}.
    """
    print("🔍 Iniciando búsqueda híbrida...")

    # 1️⃣ Embedding de la query en texto
    query_emb_text = text_model.encode([user_query])[0] # (D,)
    print("✅ Embedding de la query generado:", query_emb_text.shape)

    # 4.1) Traer nodos candidatos y sus textos
    with driver.session() as session:
        records = list(session.run("""
            MATCH (n)
            WHERE n.embedding IS NOT NULL
            RETURN elementId(n) AS nodeElementId,
                   n.rdfs_label      AS label,
                   coalesce(n.definition, n.rdfs_label) AS text
        ORDER BY nodeElementId
        """))
    
    node_ids_local = [r["nodeElementId"] for r in records]
    node_labels    = [ensure_list_of_str(r["label"]) for r in records]
    node_texts     = [ensure_list_of_str(r["text"])  for r in records]  # ← siempre lista[str]

    # 4.2) Embeddings de texto (promedio por nodo)
    node_text_embs = encode_node_texts(node_texts)  # (N, D)
    
    # 4.3) Similitud textual
    sim_text = cosine_similarity([query_emb_text], node_text_embs)[0]  # (N,)
    
    # 4.4) Vector de grafo para la query: promedio de Node2Vec de top-M por texto
    #      Alinear con índices globales de NODE2VEC
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    # Filtramos nodos que están en NODE_IDS para tener Node2Vec
    valid_idx_local = [i for i, nid in enumerate(node_ids_local) if nid in idx_map]
    if not valid_idx_local:
        # fallback: sin componente de grafo
        sim_graph = np.zeros_like(sim_text)
    else:
        # ordenar por similitud textual y tomar semillas válidas
        order = np.argsort(-sim_text)
        seeds = [i for i in order if i in valid_idx_local][:max(1, SEED_M)]
        seed_vecs = np.stack([node2vec_embeddings[idx_map[node_ids_local[i]]] for i in seeds], axis=0)  # (M, Dg)
        query_graph_vec = seed_vecs.mean(axis=0, keepdims=True)  # (1, Dg)

        node2vec_sub = np.stack([node2vec_embeddings[idx_map[node_ids_local[i]]] for i in valid_idx_local], axis=0)  # (Nv, Dg)
        # similitud solo para válidos
        sim_graph_valid = cosine_similarity(node2vec_sub, query_graph_vec).flatten()  # (Nv,)
        # expandir a todos con 0 donde no hay node2vec
        sim_graph = np.zeros_like(sim_text)
        for pos, i_local in enumerate(valid_idx_local):
            sim_graph[i_local] = sim_graph_valid[pos]

    # 5️⃣ Score híbrido
    # 4.5) Score final (normaliza cada componente para estabilidad)
    def _zscore(x):
        mu, sd = x.mean(), x.std()
        return (x - mu) / (sd + 1e-8)

    sim_text_n  = _zscore(sim_text)
    sim_graph_n = _zscore(sim_graph)
    final_score = alpha * sim_text_n + beta * sim_graph_n

    # 6️⃣ Seleccionar top-k
    top_idx  = np.argsort(-final_score)[:top_k]
    top_items = []
    for i in top_idx:
        label = " ; ".join(node_labels[i]) if node_labels[i] else ""
        text  = " ".join(node_texts[i])   if node_texts[i] else ""
        top_items.append({
            "node_id": node_ids_local[i],
            "label": label,
            "text": text,
            "score_text": float(sim_text[i]),
            "score_graph": float(sim_graph[i]),
            "score_final": float(final_score[i]),
        })
    
    # 6. Construir el contexto como un único string
    context = "\n".join([f"{it['label']}: {it['text']}".strip(": ") for it in top_items if it["text"]])

    return context, top_items # Cargar embeddings Node2Vec globales

# -----------------------------
# 5) Función para llamar al LLM
# -----------------------------

client = OpenAI(api_key=api_key)

def ask_llm(context, user_query):
    prompt = f"""
Contexto relevante extraído del grafo:
{context}

Pregunta del usuario:
{user_query}

Respuesta:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# -----------------------------
# Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    user_question = "Descreva a unidade cronoestratigráfica Paibiano."
    context, top_nodes = hybrid_rag_search(user_question, top_k=5, alpha=0.5, beta=0.5)
    print("✅ Contexto obtenido:\n", context)
    answer = ask_llm(context, user_question)
    print("\n📝 Respuesta del LLM:\n", answer)
