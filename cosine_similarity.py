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
print("✅ Embeddings Node2Vec cargados:", node2vec_embeddings.shape)

# -----------------------------
# Modelo de embeddings de texto
# -----------------------------
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Función de búsqueda híbrida
# -----------------------------
def hybrid_rag_search(user_query, top_k=5, alpha=0.5, beta=0.5):
    
    """
    Realiza una búsqueda híbrida combinando similitud semántica de texto
    con Node2Vec en el grafo.

    Args:
        user_question (str): Pregunta del usuario.
        top_k (int): Número de nodos top a retornar.
        alpha (float): Peso para la similitud de texto.
        beta (float): Peso para la similitud del grafo.

    Returns:
        context (str): Texto concatenado de los nodos top.
        top_nodes (list): Lista de nodos top seleccionados.
    """
    print("🔍 Iniciando búsqueda híbrida...")

    # 1️⃣ Embedding de la query en texto
    query_emb_text = text_model.encode([user_query])[0]
    print("✅ Embedding de la query generado:", query_emb_text.shape)

    # 2️⃣ Embedding de nodos (texto)
    with driver.session() as session:
        result = session.run("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN elementId(n) AS nodeElementId, n.rdfs_label AS label, coalesce(n.definition, n.rdfs_label) AS text
        ORDER BY nodeElementId
        """)
        node_texts = []
        node_ids_local = []
        for record in result:
            node_ids_local.append(record["nodeElementId"])
            node_texts.append(record["text"] if record["text"] else "")
    
    node_text_embeddings = []
    # Promediar embeddings de texto por nodo
    for sublist in node_texts:
        valid_texts = [t for t in sublist if t]  # eliminar strings vacíos
        if valid_texts:
            embs = text_model.encode(valid_texts)
            node_text_embeddings.append(np.mean(embs, axis=0))  # promedio por nodo
        else:
            # si no hay texto, vector cero
            node_text_embeddings.append(np.zeros(text_model.get_sentence_embedding_dimension()))

    node_text_embeddings = np.array(node_text_embeddings)  # (num_nodos, dim_embedding)


    # 3️⃣ Similitud textual
    # Cosine similarity
    #sim_text = np.dot(node_text_embeddings, query_embedding) / (np.linalg.norm(node_text_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8)
    sim_text = cosine_similarity([query_emb_text], node_text_embeddings)[0]

    # 4️⃣ Similitud Node2Vec
    # Ajusta para que las listas de node_ids coincidan
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    node2vec_sub = np.array([node2vec_embeddings[idx_map[nid]] for nid in node_ids_local])
    sim_graph = cosine_similarity([query_emb_text], node2vec_sub)[0]

    # 5️⃣ Score híbrido
    final_score = alpha * sim_text + beta * sim_graph

    # 6️⃣ Seleccionar top-k
    top_indices = np.argsort(-final_score)[:top_k]
    top_node_ids = [node_ids_local[i] for i in top_indices]

    # 5. Preparar resultados
    top_node_ids = []
    for idx in top_indices:
        node_text = nodes_texts[idx]  # nodes_texts[i] puede ser str o list
        # Convertir cualquier lista en string, sin cambiar la correspondencia
        if isinstance(node_text, list):
            node_text = " ".join(node_text)
        top_node_ids.append({"node_idx": idx, "text": node_text})
    
    # 6. Construir el contexto como un único string
    context = "\n".join([r["text"] for r in result if r["text"]])



    return context, top_node_ids

# -----------------------------
# Función para llamar al LLM
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
    user_question = "Quais são os poços localizados no campo CAMP_CD_CAMPO_0888?"
    context, top_nodes = hybrid_rag_search(user_question, top_k=5, alpha=0.5, beta=0.5)
    print("✅ Contexto obtenido:\n", context)
    answer = ask_llm(context, user_question)
    print("\n📝 Respuesta del LLM:\n", answer)
