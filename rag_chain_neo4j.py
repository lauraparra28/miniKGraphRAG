from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
#from langchain_community.graphs.neo4j_graph import Neo4jGraph
#from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
from utils import base_utils as bu

# Prompt para a geração da query Cypher
CYPHER_GENERATION_TEMPLATE = bu.load_prompts()["cypher_prompt.txt"] 
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE )

# Prompt para a resposta da query
QA_PROMPT = bu.load_prompts()["qa_prompt.txt"] # Prompt para a resposta da query
qa_prompt = PromptTemplate.from_template(QA_PROMPT)

def build_rag_chain(llm, cypher_llm, graph: Neo4jGraph):
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        qa_prompt=qa_prompt,
        CYPHER_GENERATION_TEMPLATE=CYPHER_GENERATION_TEMPLATE,
        #cypher_prompt=CYPHER_GENERATION_PROMPT,
        cypher_llm=cypher_llm,
        verbose=True,
        return_intermediate_steps=True,  # para ver la query y el contexto devuelto
        allow_dangerous_requests=True
    )
    return chain