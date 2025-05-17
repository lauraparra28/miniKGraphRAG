from langchain_neo4j import Neo4jGraph
from llm_ollama import load_llm
from rag_chain_neo4j import build_rag_chain

def main():
    
    graph = Neo4jGraph( url="bolt://localhost:7687", username="neo4j",password="diripar8$")
    print("✅ Conectado a Neo4j")
    
    llm = load_llm()
    chain = build_rag_chain(llm, graph)
    
    question = "Quantos poços estão localizados na bacia chamada AMAZONAS?" # Which geological formations are composed of sandstone?
    print("\n✅ Pregunta " + question)
    
    result = chain.invoke({"query": question})

    print("\n✅ Pregunta ")
    print(question)
    print("\n✅ Respuesta generada ")
    print(result['result'])

    if "intermediate_steps" in result:
        cypher_q, cypher_ctx = result["intermediate_steps"]
        print("\n✅ Cypher Generado:")
        print(cypher_q)
        print("\n✅ Contexto recuperado:")
        print(cypher_ctx)

if __name__ == "__main__":
    main()
