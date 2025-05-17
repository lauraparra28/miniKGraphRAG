# 🔍RAG Application using OWL Knowledge Graph

## Overview
This project demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline using only structured data from a Knowledge Graph in OWL format. It uses:

- Neo4j with RDF plugins to store and query the ontology.
- Cypher queries to extract triples.
- Natural Language Templates to convert triples to sentences.
- FAISS for semantic search over knowledge.
- Large Language Models (LLMs) to generate answers based on the retrieved context.
- LangChain to orchestrate the entire pipeline.
- Gradio for the user interface (in future).
- The project is designed to be modular and extensible, allowing for easy integration of new components or modifications to existing ones.

## Components
- **KG**: The Knowledge Graph in OWL format, which contains the structured data.
- **Neo4j**: The graph database used to store and query the KG.
- **Cypher Queries**: Used to extract triples from the KG.
- **Natural Language Templates**: Used to convert triples into natural language sentences.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors, used for semantic search over the knowledge.
- **LLMs**: Large Language Models used to generate answers based on the retrieved context.
- **LangChain**: A framework for building applications with LLMs, used to orchestrate the entire pipeline.
- **Gradio**: A library for building user interfaces, used to create a simple web interface for the application.
- **Docker**: Used to containerize the application for easy deployment and scalability.

