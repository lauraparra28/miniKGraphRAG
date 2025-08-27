# Orchestrates Hybrid RAG (semantic + Node2Vec) with Cypher QA over Neo4j.

import os
from typing import Dict, Any, List
from llm_ollama import load_llm_with_api_key
from cosine_similarity import hybrid_rag_search, ask_llm, driver 
from rag_chain_neo4j import build_rag_chain
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.prompts import PromptTemplate
from utils import base_utils as bu

# ----------------------------
# 1) LLMs and Neo4j graph
# ----------------------------
llm = load_llm_with_api_key() #load_llm()
print("✅ Successfully load LLM")

graph = Neo4jGraph( url="bolt://localhost:7687", username="neo4j",password="diripar8$")
print("✅ Successfully connection to Neo4j Graph")

# ----------------------------
# 2) Prompts for the Cypher QA
# ----------------------------
# Prompt para a geração da query Cypher
CYPHER_GENERATION_TEMPLATE = bu.load_prompts()["cypher_prompt.txt"] 
CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
# Prompt para a resposta da query
QA_PROMPT = bu.load_prompts()["qa_prompt.txt"] 
qa_prompt = PromptTemplate(template=QA_PROMPT, input_variables=["context", "question"], ) #from_template(QA_PROMPT)

cypher_chain = build_rag_chain(
    llm=llm,
    cypher_llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=qa_prompt
)
print("✅ Successfully build Cypher QA chain")
# ----------------------------
# 3) Utilities
# ----------------------------
def _format_intermediate_steps(steps: List[Dict[str, Any]], max_rows: int = 12) -> str:
    """
    Extracts the generated Cypher and tabular results from GraphCypherQAChain's
    intermediate steps into a compact, LLM-friendly text block.
    """
    blocks = []
    for s in steps:
        if "query" in s:
            blocks.append(f"[Query in Cypher]\n{s['query']}")
        if "context" in s:
            # s["context"] is typically a list of dict rows
            rows = s["context"] if isinstance(s["context"], list) else [s["context"]]
            header = []
            body = []
            # Build a simple table-ish text
            for i, row in enumerate(rows[:max_rows]):
                if isinstance(row, dict):
                    if not header:
                        header = list(row.keys())
                    body.append(" | ".join(str(row.get(k, "")) for k in header))
                else:
                    body.append(str(row))
            if header:
                blocks.append("[Context]\n" + " | ".join(header) + "\n" + "\n".join(body))
    return "\n\n".join(blocks).strip()

# ----------------------------
# 4) The orchestrator
# ----------------------------
def run_hybrid_rag(question: str,
                   top_k: int = 8,
                   alpha: float = 0.5,
                   beta: float = 0.5,
                   seed_m: int = 15) -> Dict[str, Any]:
    """
    1) Hybrid retrieve best nodes (semantic + Node2Vec)
    2) Run Cypher QA to fetch structured facts
    3) Merge both into a single extractive context
    4) Call your extractive ask_llm() to produce the final answer
    """
    print("🔍 Running Hybrid RAG...")
    print("📝 Semantic + Node2Vec fusion over node texts")

    # --- (1) Semantic + Node2Vec fusion over node texts ---
    context_text, top_nodes = hybrid_rag_search(
        question, top_k=top_k, alpha=alpha, beta=beta, SEED_M=seed_m
    )  # from cosine_similarity.py  :contentReference[oaicite:4]{index=4}
    top_ids = [t["node_id"] for t in top_nodes]
    
    # --- (2) Cypher QA chain over the KG ---
    cypher_out = cypher_chain.invoke({"query": question})
    cypher_answer   = cypher_out.get("result", "")
    intermediate    = cypher_out.get("intermediate_steps", [])
    cypher_context  = _format_intermediate_steps(intermediate)

    # --- (3) Merge contexts and ask the LLM (strictly extractive) ---
    merged_context = "\n\n---\n".join(
        block for block in [cypher_context, context_text] if block
    ).strip()

    print("🔍 Merged context:", merged_context)
    
    final_answer = ask_llm(merged_context, question)  # from cosine_similarity.py  :contentReference[oaicite:5]{index=5}

    return {
        "answer": final_answer,
        "hybrid_top_nodes": top_nodes,
        "cypher_answer": cypher_answer,
        "cypher_steps": intermediate,
        "merged_context_preview": merged_context[:1500]  # for inspection/logging
    }

# ----------------------------
# 5) Example
# ----------------------------
if __name__ == "__main__":
    q = "Em que bacia está localizado o campo MORRO DO BARRO?"
    q2 = "Que unidades litoestratigráficas o poço 2-CAST-0002-AM atravessa que são constituídas por rochas do tipo conglomerado?"
    # Em que bacia está localizado o poço 1MD-0001-AM?
    # O que é um sandstone?
    # O que é um(a) quartzoarenito?
    # Que unidades litoestratigráficas o poço 2-CAST-0002-AM atravessa que são constituídas por rochas do tipo conglomerado?" 
    # Descreva a unidade cronoestratigráfica Paibiano.
    # Que unidades litoestratigráficas o poco POCO_CD_POCO_022749 atravessa?

    out = run_hybrid_rag(q, top_k=5, alpha=0.5, beta=0.5, seed_m=15)
    
    print("\n=== Top Nodes (hybrid) ===")
    for i, n in enumerate(out["hybrid_top_nodes"], 1):
        print(f"{i:02d}. {n['label']}  (text={n['score_text']:.3f}, graph={n['score_graph']:.3f}, final={n['score_final']:.3f})")
    print("\n=== Cypher QA (short) ===\n", out["cypher_answer"])

    print("\n=== Final Answer ===\n", out["answer"])