from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
#from langchain_community.graphs.neo4j_graph import Neo4jGraph
#from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}
Examples: Here are examples of generated Cypher statements for particular questions:

# Quais formações estão constituídas pelo material chamado siltito?
MATCH (f:lithostratigraphic_unit)-[:constituted_by]->(m:siltstone)
RETURN f.rdfs_label

# Qual é a entidade que faz parte de Formação Manacapuru?
MATCH (f:lithostratigraphic_unit{rdfs_label: "Formação Manacapuru"})-[r:part_of]->(p:lithostratigraphic_unit) 
RETURN p.rdfs_label

# Que unidades litoestratigráficas o poco POCO_CD_POCO_023016 atravessa?
MATCH (p:well {_id:"POCO_CD_POCO_023016"})-[:crosses]->(u:lithostratigraphic_unit)
RETURN u.rdfs_label

# Quantos poços estão localizados na bacia chamada AMAZONAS?
MATCH (w:well)-[:located_in]->(b:basin {rdfs_label:"AMAZONAS"})
RETURN COUNT (w) AS numero_de_pocos

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

QA_PROMPT = """A sua função é a través das informações submetidas que estejam no grafo, em formato de resposta de uma query cypher é responder de forma detalhada na forma culta da lingua portuguesa. 
Utilize apenas os tipos de relacionamentos e propriedades fornecidos no grafo. Não utilize nenhum outro tipo de relacionamento ou propriedade que não seja fornecido no grafo. 

Estruture sua resposta de forma clara e organizada, fornecendo informações relevantes sobre os relacionamentos e entidades mencionadas na pergunta.
Não inclua explicações ou desculpas em suas respostas.

Informação:
{context}

Questão: {question}

Resposta ideal:
"""
qa_prompt = PromptTemplate.from_template(QA_PROMPT)

def build_rag_chain(llm, graph: Neo4jGraph):
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        qa_prompt=qa_prompt,
        cypher_generation_template=CYPHER_GENERATION_TEMPLATE,
        verbose=True,
        return_intermediate_steps=True,  # para ver la query y el contexto devuelto
        allow_dangerous_requests=True,
        top_k=10,
    )
    return chain