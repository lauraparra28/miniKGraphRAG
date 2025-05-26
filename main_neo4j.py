from langchain_neo4j import Neo4jGraph
from llm_ollama import load_llm, load_cypher_llm
from rag_chain_neo4j import build_rag_chain
from utils import base_utils as bu
from langchain.prompts import PromptTemplate

# Prompt para a geração da query Cypher
CYPHER_GENERATION_TEMPLATE = bu.load_prompts()["cypher_prompt.txt"] 
CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
# Prompt para a resposta da query
QA_PROMPT = bu.load_prompts()["qa_prompt.txt"] 
qa_prompt = PromptTemplate(template=QA_PROMPT, input_variables=["context", "question"], ) #from_template(QA_PROMPT)

graph = Neo4jGraph( url="bolt://localhost:7687", username="neo4j",password="diripar8$")
print("✅ Successfully connection to Neo4j Graph")
    
llm = load_llm()
cypher_llm = load_cypher_llm()
print("✅ Successfully load LLM")

chain = build_rag_chain(
    llm=llm,
    cypher_llm=cypher_llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=qa_prompt
)

def main():
    question = input("❓ Pergunta: ")
         
    out = chain.invoke({ "query": question})
 
    print("\n✅ Resposta:", out["result"])

    if "intermediate_steps" in out:
        cypher_q, cypher_ctx = out["intermediate_steps"]
        print("\n✅ Cypher Generado:")
        print(cypher_q)
        print("\n✅ Contexto recuperado:" + str(cypher_ctx))
        
if __name__ == "__main__":
    main()
