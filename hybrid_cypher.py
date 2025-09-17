import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Importa tus componentes de LangChain ---
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rag_chain_neo4j import build_rag_chain as build_text_to_cypher_chain
from utils import base_utils as bu

# --- Carga de configuración ---
load_dotenv()

# --- 1️⃣ Conexiones y Modelos Centralizados ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "diripar8$"
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
llm_langchain = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

print("✅ Clientes y modelos cargados correctamente.")

# --- 2️⃣ Configuración de la cadena LangChain (Text-to-Cypher) ---
# (Usa los mismos prompts que ya tenías en tu main_neo4j.py)

CYPHER_GENERATION_TEMPLATE = bu.load_prompts()["cypher_nodes_prompt.txt"] 
CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
QA_PROMPT = bu.load_prompts()["qa_prompt.txt"] 
qa_prompt = PromptTemplate(template=QA_PROMPT, input_variables=["context", "question"], ) #from_template(QA_PROMPT)

graph_langchain = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# Construimos la cadena para generar Cypher
text_to_cypher_chain = build_text_to_cypher_chain(
    llm=llm_langchain,
    cypher_llm=llm_langchain,
    graph=graph_langchain,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=qa_prompt  
)

# --- 3️⃣ La Función de RAG Híbrido Integrado ---
def run_intelligent_hybrid_rag(query: str, top_k: int = 3):
    """
    Orquesta el flujo completo: Text-to-Cypher -> Búsqueda Vectorial -> Respuesta.
    """
    print(f"🔄 Iniciando RAG inteligente para la pregunta: '{query}'")

    # --- PASO 1: Generar la consulta Cypher dinámicamente ---
    print("🧠 Paso 1: Generando consulta Cypher con LLM...")
    langchain_result = text_to_cypher_chain.invoke({"query": query})
    generated_cypher = langchain_result["intermediate_steps"][0]["query"]
    print(f"✅ Cypher generado:\n{generated_cypher}")

    # --- PASO 2: Ejecutar el Cypher y obtener los nodos candidatos ---
    print("\n➡️ Paso 2: Ejecutando Cypher en Neo4j para filtrar nodos...")
    with neo4j_driver.session() as session:
        results = session.run(generated_cypher)
        data = [record for record in results]
    
    print(f"✅ Cypher encontró {len(data)} nodos candidatos.")
    if not data or len(data) == 0:
        return "El LLM no pudo generar una consulta Cypher válida o no se encontraron resultados estructurales en el grafo para su pregunta.", []

    # --- PASO 3: Búsqueda vectorial sobre los candidatos (lógica de FAISS) ---
    print("\n🚀 Paso 3: Realizando búsqueda vectorial con FAISS sobre los candidatos...")
    texts = [r["definition"] for r in data if r.get("definition")]
    node_names = [r["name"] for r in data if r.get("definition")]
    
    if not texts:
        return "Los nodos encontrados no tienen una propiedad 'definition' para la búsqueda semántica.", []

    embeddings = sentence_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    query_emb = sentence_model.encode([query])
    _, I = index.search(np.array(query_emb), top_k)
    
    retrieved_texts = [texts[i] for i in I[0]]
    retrieved_nodes = [node_names[i] for i in I[0]]
    print(f"✅ Nodos más relevantes según FAISS: {retrieved_nodes}")

    # --- PASO 4: Generar la respuesta final ---
    print("\n✍️ Paso 4: Generando respuesta final con el contexto refinado...")
    prompt = f"""
    Você é um assistente que responde de forma detalhada na forma culta da lingua portuguesa. 

    REGRAS:
    1) Responda exclusivamente com base na informação detalhada fornecida.
    2) Não forneça informações adicionais que não estejam no contexto.
    3) Se a informação NÃO estiver no contexto, responda apenas: "Sem dados suficientes no contexto."

    Informação detalhada dos nós mais relevantes do grafo:
    {retrieved_texts}
    
    Responda a seguinte pergunta com precisão: {query}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    final_answer = response.choices[0].message.content

    return final_answer, retrieved_texts

# --- 4️⃣ Ejecución ---
if __name__ == "__main__":
    question = input("❓ Pergunta: ")

    respuesta_final, retrieved_texts = run_intelligent_hybrid_rag(question)

    print("\n\n✅✅✅ Respuesta Final del Sistema Integrado ✅✅✅")
    print(respuesta_final)