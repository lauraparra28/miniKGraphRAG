from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
from utils import base_utils as bu

# Prompt para a geração da query Cypher
CYPHER_GENERATION_TEMPLATE = bu.load_prompts()["cypher_prompt.txt"] 
CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
# Prompt para a resposta da query
QA_PROMPT = bu.load_prompts()["qa_prompt.txt"] 
qa_prompt = PromptTemplate.from_template(QA_PROMPT)

def build_rag_chain(llm, cypher_llm, graph: Neo4jGraph):
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        qa_prompt=qa_prompt,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        cypher_llm=cypher_llm,
        return_intermediate_steps=True, 
        allow_dangerous_requests=True, 
        verbose=True,
        top_k=10,
    )
    return chain