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
    
    question = "Quais unidades litoestratigráficas atravessam o poco POCO_CD_POCO_023016?"
    # Which is the rdfs_label of the basin where is located the field CAMP_CD_CAMPO_0633?"
    # Qual é a unidade litoestretigrafica que faz parte de Formação Manacapuru?"
    # How many wells are located in the AMAZONAS basin?" 
    # Quantos poços estão localizados na bacia chamada AMAZONAS
    # Which geological formations are composed of sandstone?
    print("\n❓ " + question)
    
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
