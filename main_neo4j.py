from langchain_neo4j import Neo4jGraph
from llm_ollama import load_llm, load_cypher_llm
from rag_chain_neo4j import build_rag_chain

def main():
    
    graph = Neo4jGraph( url="bolt://localhost:7687", username="neo4j",password="diripar8$")
    print("✅ Successfully connection to Neo4j Graph")
    
    llm = load_llm()
    cypher_llm = load_cypher_llm()
    print("✅ Successfully load LLM")
    chain = build_rag_chain(llm, cypher_llm, graph)
    
    question = "Quantos poços estão localizados na bacia CAMAMU-ALMADA?"

    print("\n❓ " + question)
    
    result = chain.invoke({"query": question}) 
 
    print("\n✅ Pregunta: " + question) 
    print("\n✅ Respuesta gerada: ")
    print(result["result"])

    if "intermediate_steps" in result:
        cypher_q, cypher_ctx = result["intermediate_steps"]
        print("\n✅ Cypher Generado:")
        print(cypher_q)
        print("\n✅ Contexto recuperado:" + str(cypher_ctx))
        
if __name__ == "__main__":
    main()
