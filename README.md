# 🧠 Mini KG-RAG Application using OWL Knowledge Graph

## 🔍 Overview
This project demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline using only structured data from a Knowledge Graph in OWL format. It uses:

- Neo4j with RDF plugins to store and query the ontology.
- Cypher queries to extract triples.
- Large Language Models (LLMs) to generate answers based on the retrieved context.
- LangChain to orchestrate the entire pipeline.
- Gradio for the user interface (in future).
- The project is designed to be modular and extensible, allowing for easy integration of new components or modifications to existing ones.

## Components
- **KG**: The Knowledge Graph in OWL format, which contains the structured data.
- **Neo4j**: The graph database used to store and query the KG.
- **Cypher Queries**: Used to extract triples from the KG.
- **Natural Language Templates**: Used to convert triples into natural language sentences.
- **LLMs**: Large Language Models used to generate answers based on the retrieved context.
- **LangChain**: A framework for building applications with LLMs, used to orchestrate the entire pipeline.
- **Gradio**: A library for building user interfaces, used to create a simple web interface for the application.
- **Docker**: Used to containerize the application for easy deployment and scalability.

## 🛠️ Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run
1. Make sure you have the following services running:
    - **Neo4j**: Make sure you have the Neo4j database running with the RDF plugin enabled. You can use Docker to run Neo4j with RDF support.
    - **Ollama**: Make sure you have the Ollama server running. You can install it from [Ollama's website](https://ollama.com/).

In a WSL terminal, run the following command to start the Ollama server:

```bash
ollama serve
```

2. When the server is running, open a new terminal and run the following command to run the model:
You should see a message indicating that the model is ready to accept requests.

```bash
ollama run deepseek-r1:1.5b
```

3. Verify that the model is running by checking the output in the terminal. 

```bash
curl http://localhost:11434

```

4. Make sure you have the OWL file in the `data` directory. The OWL file should be named `your_owl_file.owl`. You can replace it with your own OWL file.
5. Update the `main.py` file with your Neo4j connection details and the path to your OWL file.

6. Run the script:

```bash
python main.py
```
You can obtain something like this:

```bash
✅ Successfully connection to Neo4j Graph
✅ Successfully load LLM

❓ Em que bacia está localizado o campo MORRO DO BARRO?

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (c:field{rdfs_label: "MORRO DO BARRO"})-[:located_in]->(b:basin) 
RETURN b.rdfs_label
Full Context:
[]

> Finished chain.

✅ Pregunta: Em que bacia está localizado o campo MORRO DO BARRO?

✅ Respuesta gerada: 
O campo MORRO DO BARRO está localizado na Bacia CAMAMU-ALMADA.

✅ Cypher Generado:
{'query': 'MATCH (c:field{rdfs_label: "MORRO DO BARRO"})-[:located_in]->(b:basin) \nRETURN b.rdfs_label'}

✅ Contexto recuperado:{'context': []}
```

## 🧩 Configuration

Configure the main_neo4j.py file with the parameters for the Neo4j connection.

```bash
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="tu_contraseña"
)
```

## Troubleshooting

- ModuleNotFoundError: openai: instala el paquete openai aunque no uses la API.
- ImportError GraphCypherQAChain: asegúrate de importar todo desde langchain_neo4j.
- ValidationError for GraphCypherQAChain: usa la misma fuente (langchain_neo4j) para el Neo4jGraph y el Chain.