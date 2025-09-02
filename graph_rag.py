from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1️⃣ Conexión a Neo4j ---
URI = "bolt://localhost:7687" 
USER = "neo4j"
PASSWORD = "diripar8$"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
print("✅ Successfully connection to Neo4j Graph")

# --- 2️⃣ Configurar modelo LLM y embeddings ---
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 3️⃣ Función para consultar Neo4j con Cypher ---
def cypher_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return [record for record in result]
    
# --- 4️⃣ Recuperación híbrida ---
def hybrid_rag(query, cypher_filter, top_k=3):
    # 4a️) Filtrado estructurado con Cypher
    cypher = f"""
    MATCH (n)
    WHERE {cypher_filter}
    RETURN n.rdfs_label AS name, n.definition AS definition
    """
    results = cypher_query(cypher)
    print(f"✅ Cypher returned results")
    #print("Results sample:", results[:2])
    
    # Si no hay resultados, salir
    if not results:
        return "Sem dados suficientes, nós não relevantes no grafo."
    
    # 4b) Preparar embeddings
    texts = [r["definition"] for r in results if r["definition"]]
    node_names = [r["name"] for r in results if r["definition"]]
    
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    print(f"✅ Generated embeddings of dimension {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    print(f"✅ FAISS index has {index.ntotal} vectors")
    
    # 4c️) Vector search
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)
    retrieved_texts = [texts[i] for i in I[0]]
    retrieved_nodes = [node_names[i] for i in I[0]]

    print("✅Retrieved nodes:", retrieved_nodes)
    print("✅Retrieved texts sample:", retrieved_texts[:2])
    
    # 4d) Construir prompt
    prompt = f"""
    Você é um assistente que responde de forma detalhada na forma culta da lingua portuguesa. 

    REGRAS:
    1) NÃO use conhecimento externo.
    2) Copie termos técnicos e números exatamente como estão.
    3) Se a informação NÃO estiver no contexto, responda apenas: "Sem dados suficientes no contexto."
    4) Utilize a informação dos seguintes nodos e suas relações do grafo:
    Nodos: {retrieved_nodes}
    Información detallada: {retrieved_texts}
    
    Responde la siguiente pregunta con precisión: {query}
    """
    
    print("✅ Built prompt for LLM")
    print("Prompt sample:", prompt[:1500])
    
    # 4e️) LLM genera respuesta
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content    

# --- 5️⃣ Ejemplo de uso ---
query2 = "Que unidades litoestratigráficas o poço 9-FZ-0024-AM atravessa?"
cypher_filter2 = 'n:well AND "9-FZ-0024-AM" IN n.rdfs_label'  # Solo nodos del pozo 9-FZ-0024-AM

query = "Descreva a unidade cronoestratigráfica Idade Bartoniana."
cypher_filter = 'n:geological_time_interval AND "Idade Bartoniana" IN n.rdfs_label'  # Solo nodos del pozo 9-FZ-0024-AM
respuesta = hybrid_rag(query, cypher_filter)

print("✅ Final response from hybrid RAG:")
print(respuesta)