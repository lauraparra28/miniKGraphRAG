import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Importa tus componentes de LangChain ---
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rag_chain_neo4j import build_rag_chain as build_text_to_cypher_chain
from utils import base_utils as bu

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# --- Carga de configuración ---
load_dotenv()

# --- 1️⃣ Conexiones y Modelos Centralizados ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "diripar8$"
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
llm_langchain = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

print("✅ Clientes y modelos cargados correctamente.")

# --- 2️⃣ Configuración de la cadena LangChain (Text-to-Cypher) ---
# (Usa los mismos prompts que ya tenías en tu main_neo4j.py)

CYPHER_GENERATION_TEMPLATE = bu.load_prompts()["cypher_nodes_prompt.txt"] 
CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
QA_PROMPT = bu.load_prompts()["qa_prompt.txt"] 
qa_prompt = PromptTemplate(template=QA_PROMPT, input_variables=["context", "question"], ) #from_template(QA_PROMPT)

graph_langchain = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# Construimos la cadena para generar Cypher
text_to_cypher_chain = build_text_to_cypher_chain(
    llm=llm_langchain,
    cypher_llm=llm_langchain,
    graph=graph_langchain,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=qa_prompt  
)

# --- 3️⃣ La Función de RAG Híbrido Integrado ---
def run_intelligent_hybrid_rag(query: str, top_k: int):
    """
    Orquesta el flujo completo: Text-to-Cypher -> Búsqueda Vectorial -> Respuesta.
    """
    print(f"🔄 Iniciando RAG inteligente para la pregunta: '{query}'")

    # --- PASO 1: Generar la consulta Cypher dinámicamente ---
    print("🧠 Paso 1: Generando consulta Cypher con LLM...")
    langchain_result = text_to_cypher_chain.invoke({"query": query})
    generated_cypher = langchain_result["intermediate_steps"][0]["query"]
    print(f"✅ Cypher generado:\n{generated_cypher}")

    # --- PASO 2: Ejecutar el Cypher y obtener los nodos candidatos ---
    print("\n➡️ Paso 2: Ejecutando Cypher en Neo4j para filtrar nodos...")
    with neo4j_driver.session() as session:
        results = session.run(generated_cypher)
        data = [record for record in results]
    
    print(f"✅ Cypher encontró {len(data)} nodos candidatos.")
    if not data:
        return "El LLM no pudo generar una consulta Cypher válida o no se encontraron resultados estructurales en el grafo para su pregunta."

    # --- PASO 3: Búsqueda vectorial sobre los candidatos (lógica de FAISS) ---
    print("\n🚀 Paso 3: Realizando búsqueda vectorial con FAISS sobre los candidatos...")
    
    texts_for_faiss = []
    node_names = []

    for record in data:
        # Lógica de Fallback:
        # 1. Intenta obtener la 'definition'.
        # 2. Si es nula o no existe, el 'or' pasará a evaluar y obtener el 'name'.
        text_to_embed = record.get("definition") or record.get("name")
        
        # Asegúrate de que el nodo tenga un nombre y algún texto para analizar
        if text_to_embed and record.get("name"):
            texts_for_faiss.append(text_to_embed)
            node_names.append(record.get("name"))
    
    # Si después del fallback no hay nada que procesar, termina.
    if not texts_for_faiss:
        return "Los nodos encontrados no tienen propiedades 'definition' o 'name' para analizar."

    print(f"✅ Se procesarán {len(texts_for_faiss)} nodos para la búsqueda vectorial (usando 'name' como fallback).")

    # Evita repetir múltiples veces el mismo nodo cuando hay pocos candidatos
    top_k_eff = min(top_k, len(texts_for_faiss))

    # Genera los embeddings usando la lista que ahora contiene definitions o names     
    embeddings = sentence_model.encode(texts_for_faiss)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    query_emb = sentence_model.encode([query])
    _, I = index.search(np.array(query_emb), top_k_eff)
    
    retrieved_texts = [texts_for_faiss[i] for i in I[0]]
    print(f"✅ Textos más relevantes según FAISS: {retrieved_texts}")
    retrieved_nodes = [node_names[i] for i in I[0]]
    print(f"✅ Nodos más relevantes según FAISS: {retrieved_nodes}")

    # --- PASO 4: Generar la respuesta final ---
    print("\n✍️ Paso 4: Generando respuesta final con el contexto refinado...")
    prompt = f"""
    Você é um assistente que responde de forma detalhada na forma culta da lingua portuguesa. 
    Responda exclusivamente com base na informação detalhada fornecida.
    Não utilize nenhum outro tipo de informação além da fornecida no contexto do grafo.
    
    Contexto ou informação detalhada dos nós mais relevantes do grafo:
    {retrieved_texts}
    
    Responda a seguinte pergunta: {query}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, embeddings, node_names, texts_for_faiss, top_k_eff, query_emb

def debug_plot_embeddings(embeddings, node_names=None, texts=None, out_dir="debug_emb", prefix="", *,
                          neighbors_k=2, do_tsne=True, dpi=160, random_state=0, verbose=True,
                          annotate=True, show=False, label_max_chars=40, return_dict=True,
                          query_embedding=None, query_label="[QUERY]", plot_query=True):

    # --- Helpers --------------------------------------------------------------
    def _to_np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _clean_label(v):
        # convierte listas/tuplas a string "a / b / c"
        if isinstance(v, (list, tuple, set)):
            return " / ".join(map(str, v))
        return str(v)

    def _truncate(s, n):
        return s if len(s) <= n else (s[: n - 1] + "…")
    
    def _has_variation(arr, atol=1e-12):
        return np.any(np.std(arr, axis=0) > atol)

    # --- Sanitización de entradas --------------------------------------------
    X = _to_np(embeddings)
    if X.ndim != 2:
        raise ValueError(f"embeddings debe ser 2D (N,D). Shape recibido: {X.shape}")
    
    if not np.all(np.isfinite(X)):
        bad = np.argwhere(~np.isfinite(X))
        raise ValueError(f"Embeddings contienen NaN/Inf: {bad.tolist()}")

    N, D = X.shape
    if N == 0:
        raise ValueError("No hay embeddings (N=0).")

    if node_names is None:
        node_names = [f"item_{i}" for i in range(N)]
    if len(node_names) != N:
        raise ValueError(f"node_names debe tener N elementos (= {N}); recibido {len(node_names)}.")

    clean_names = [_clean_label(n) for n in node_names]

    if texts is None:
        texts = clean_names
    if len(texts) != N:
        raise ValueError(f"texts debe tener N elementos (= {N}); recibido {len(texts)}.")
    clean_texts = [_clean_label(t) for t in texts]

    os.makedirs(out_dir, exist_ok=True)
    prefix = (prefix + "_") if (prefix and not prefix.endswith("_")) else prefix

    # --------------- Sanitización query ---------------
    qX = None
    query_plotted = False
    if plot_query and query_embedding is not None:
        qX = _to_np(query_embedding)
        if qX.ndim == 1:
            qX = qX[None, :]  # (1,D)
        if qX.shape[1] != D:
            raise ValueError(f"query_embedding dimensión {qX.shape[1]} != D de candidatos {D}")
        if not np.all(np.isfinite(qX)):
            raise ValueError("query_embedding contiene NaN/Inf.")
        
    os.makedirs(out_dir, exist_ok=True)
    prefix = (prefix + "_") if (prefix and not prefix.endswith("_")) else prefix

    # Aviso duplicados
    unique_rows = np.unique(X, axis=0).shape[0]
    if verbose and unique_rows < N:
        print(f"[WARN] {N - unique_rows} embedding(s) duplicado(s) detectado(s).")        
    
    # --- CSV de embeddings ----------------------------------------------------
    emb_df = pd.DataFrame(X)
    emb_df.insert(0, "node_name", clean_names)
    emb_df.insert(1, "text", clean_texts)
    if qX is not None:
        # añade la query como última fila con etiquetas especiales
        qrow = [query_label, "[QUERY_TEXT]"] + qX.flatten().tolist()
        # alinear columnas
        while len(qrow) < emb_df.shape[1]:
            qrow.append(None)
        if len(qrow) > emb_df.shape[1]:
            # expandir columnas si hiciera falta (raro)
            for _ in range(len(qrow) - emb_df.shape[1]):
                emb_df[f"extra_{_}"] = None
        emb_df.loc[len(emb_df)] = qrow[:emb_df.shape[1]] 
    
    csv_path = os.path.join(out_dir, f"{prefix}embeddings.csv")
    emb_df.to_csv(csv_path, index=False, encoding="utf-8")

    X = np.asarray(embeddings)
    if X.ndim == 1:
        X = X[None, :]  # asegura shape (N,D) aunque venga (D,)

    N, D = X.shape
    # --------------- PCA (fit en candidatos; transform de query) -------------
    pca_img = None
    pca_var = 0.0
    
    if D >= 1 and N >= 1 and _has_variation(X):
        ncomp = min(2, N, D)
        pca = PCA(n_components=ncomp, random_state=random_state)
        X_pca = pca.fit_transform(X)
        pca_var = float(pca.explained_variance_ratio_.sum())
        
        # si ncomp==1, dibuja con y=0 para evitar crash (fallback)
        if ncomp == 1: 
            X_plot = np.column_stack([X_pca[:, 0], np.zeros(N)])
            title = f"PCA 1D de embeddings (y=0) (var. exp.: {pca_var:.1%})"
            q_plot = None
            if qX is not None:
                q_pca = pca.transform(qX)
                q_plot = np.column_stack([q_pca[:, 0], [0.0]])
        else:
            X_plot = X_pca
            title = f"PCA 2D (var. exp.: {pca_var:.1%})"
            q_plot = None
            if qX is not None:
                q_plot = pca.transform(qX)  # (1,2)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=70)
        if annotate:
            for i, lbl in enumerate(clean_names):
                plt.annotate(_truncate(lbl, label_max_chars), (X_plot[i, 0], X_plot[i, 1]),
                         xytext=(5, 3), textcoords="offset points", fontsize=9)
        # Dibuja la query como un marcador distinto
        if q_plot is not None:
            plt.scatter(q_plot[:, 0], q_plot[:, 1], s=140, marker="*", edgecolor="k", linewidths=1.0)
            if annotate:
                plt.annotate(_truncate(query_label, label_max_chars),
                             (q_plot[0, 0], q_plot[0, 1]),
                             xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold")
            query_plotted = True
            
        plt.title(title)
        plt.xlabel("PC1"); plt.ylabel("PC2" if ncomp == 2 else "0")
        plt.tight_layout()
        pca_img = os.path.join(out_dir, f"{prefix}embeddings_pca{'2d' if ncomp==2 else '1d'}_plot.png")
        plt.savefig(pca_img, dpi=dpi)
        if show: plt.show()
        plt.close()
    
    else:
        if verbose:
            print("[EMB] PCA omitido (sin variación suficiente o N/D muy pequeños).")

    # --- t-SNE 2D (opcional) -------------------------------------------------
    tsne_img = None
    # t-SNE solo si hay ≥3 muestras y variación suficiente
    if N >= 3 and np.unique(X, axis=0).shape[0] >= 3:

        # perplexity debe ser < N; usaremos un valor seguro
        perplex = min(30, max(5, N - 1))
        try:
            tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                    random_state=random_state, perplexity=perplex)
            X_tsne = tsne.fit_transform(X)

            plt.figure(figsize=(8, 6))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=70)
            if annotate:
                for i, lbl in enumerate(clean_names):
                    plt.annotate(_truncate(lbl, label_max_chars), (X_tsne[i, 0], X_tsne[i, 1]),
                                xytext=(5, 3), textcoords="offset points", fontsize=9)
            plt.title("t-SNE 2D de embeddings")
            plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2"); plt.tight_layout()
            tsne_img = os.path.join(out_dir, f"{prefix}embeddings_tsne2d.png")
            plt.savefig(tsne_img, dpi=dpi)
            if show: plt.show()
            plt.close()
            
        except Exception as e:
            print(f"[WARN] t-SNE omitido: {e}")
    else:
        print("[EMB] t-SNE omitido (N<3 o muy poca variación).")

    # --- Heatmap de similitud coseno -----------------------------------------
    S = cosine_similarity(X)  # (N x N)
    plt.figure(figsize=(7.2, 6.6))
    im = plt.imshow(S, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_labels = [_truncate(n, label_max_chars) for n in clean_names]
    ticks = range(N)
    plt.xticks(ticks, tick_labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(ticks, tick_labels, fontsize=8)
    plt.title("Mapa de similitud coseno")
    plt.tight_layout()
    sim_img = os.path.join(out_dir, f"{prefix}embeddings_cosine_heatmap.png")
    plt.savefig(sim_img, dpi=dpi)
    if show: plt.show()
    plt.close()

    # --- Vecinos más similares por coseno ------------------------------------
    neighbors_out = []
    for i, lbl in enumerate(clean_names):
        order = np.argsort(-S[i])  # descendente
        neigh = [(clean_names[j], float(S[i, j])) for j in order if j != i][:max(0, neighbors_k)]
        neighbors_out.append((lbl, neigh))
        if verbose:
            pretty = ", ".join([f"{n} ({s:.3f})" for n, s in neigh]) or "—"
            print(f"[SIM] {lbl} -> {pretty}")

    if verbose:
        print(f"[EMB] shape candidatos = {X.shape} | query_plotted={query_plotted}")
        print(f"[EMB] CSV guardado en: {csv_path}")
        print(f"[EMB] PCA guardado en: {pca_img}")
        if tsne_img:
            print(f"[EMB] t-SNE guardado en: {tsne_img}")
        else:
            print("[EMB] t-SNE omitido (N < 3 o do_tsne=False).")
        print(f"[EMB] Heatmap guardado en: {sim_img}")

    if return_dict:
        return {
            "csv": csv_path,
            "pca_png": pca_img,
            "tsne_png": tsne_img,
            "sim_png": sim_img,
            "pca_var_explained": pca_var,
            "neighbors": neighbors_out,
            "n_samples": N,
            "n_features": D,
            "unique_rows": unique_rows,
            "query_plotted": query_plotted,
        }
    return None    
    
# --- 4️⃣ Ejecución ---
if __name__ == "__main__":
    pregunta_compleja = "Que unidades litoestratigráficas o poço 9-FZ-0024-AM atravessa?"
    # Que unidades litoestratigráficas o poço 1-BAS-0129-BA atravessa que são constituídas por rochas do tipo rudito?"
    #Que unidades litoestratigráficas o poço BIST-1 atravessa?"
    # Que unidades litoestratigráficas o poço 9-FZ-0007-AM atravessa?"
    # "Que unidades litoestratigráficas o poço 9-FZ-0024-AM atravessa?"
    #pregunta_compleja = "O que é um(a) sandstone?" Em que bacia está localizado o poço 1MD-0001-AM?
    #Que unidades litoestratigráficas o poço 2-CAST-0002-AM atravessa que são constituídas por rochas do tipo conglomerado?
    # #pregunta_compleja = "Descreva a unidade cronoestratigráfica Idade Bartoniana"

    respuesta_final, embeddings, node_names, texts_for_faiss, top_k_eff, query_emb = run_intelligent_hybrid_rag(pregunta_compleja, top_k=10)

    results = debug_plot_embeddings( embeddings=embeddings, node_names=node_names, texts=texts_for_faiss,
      out_dir="debug_emb", prefix="pozo_9-FZ-0024-AM", neighbors_k=top_k_eff, do_tsne=True, 
      show=False, verbose=True, query_embedding=query_emb, query_label="[QUERY]" )


    print("\n\n✅✅✅ Resposta Final ✅✅✅")
    print(respuesta_final)