# Avalia o desempenho do modelo RAG em um dataset de teste

import re, unicodedata
from collections import Counter
import json
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score
from difflib import SequenceMatcher
import sacrebleu
from main_neo4j import chain
from utils import base_utils as bu
from ragas.metrics import faithfulness, context_recall, context_precision
from datasets import Dataset
from ragas import evaluate

# Generar nombre de archivo con fecha actual
fecha_actual = datetime.now().strftime("%d_%m_%Y")
output_file = os.path.join("results", f"evaluation_results_{fecha_actual}.jsonl")
final_metrics_file = os.path.join("results", f"final_metrics_{fecha_actual}.json")
print("📁 Arquivos criados para guardar dados do teste")

# 1) Carrega o dataset
dataset_miniKGraph = bu.load_dataset()["MiniKGraph_teste.json"] # xt_dataset_balanced_1009 Dataset de teste MiniKGraph_teste.json
print("✅ Successfully load Dataset miniKGraph for Evaluation")

# 2) Funções auxiliares
# Normaliza as respostas, removendo espaços extras e convertendo para minúsculas
def normalize(text: str) -> str:
 
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(map(str, text))  # fallback defensivo
    # Reemplazar saltos de línea y tabulaciones por espacio
    text = text.replace("\n", " ").replace("\t", " ").replace("\xa0", " ")
    # Eliminar acentos
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Eliminar espacios extra
    # Colapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def flatten_answers(ans):
    # Ans vem como List[List[str]] ou List[str]
    if isinstance(ans, list) and ans and isinstance(ans[0], list):
        return [normalize(a) for sub in ans for a in sub]
    elif isinstance(ans, list):
        return [normalize(a) for a in ans]
    else:
        return [normalize(ans)]

def is_close_match(a: str, b: str, threshold: float = 0.85) -> bool:
    """Retorna True si a y b son suficientemente similares según el threshold."""
    return SequenceMatcher(None, a, b).ratio() >= threshold

# ---------- Utilidades ----------
def tokenize_norm(s: str): return normalize(s).split()

def token_f1(pred: str, gold: str) -> float:
    ptoks = tokenize_norm(pred)
    gtoks = tokenize_norm(gold)
    if not ptoks or not gtoks:  return 0.0
    pc, gc = Counter(ptoks), Counter(gtoks)
    overlap = sum((pc & gc).values())
    if overlap == 0: return 0.0
    precision = overlap / len(ptoks)
    recall    = overlap / len(gtoks)
    return 2 * precision * recall / (precision + recall)

def best_token_f1(pred: str, golds: list[str]) -> float:
    return max(token_f1(pred, g) for g in golds) if golds else 0.0

# def lcs_len(a_tokens, b_tokens):
#     # DP O(n*m) para ROUGE-L (cadenas cortas: ok)
#     n, m = len(a_tokens), len(b_tokens)
#     dp = [0]*(m+1)
#     for i in range(1, n+1):
#         prev = 0
#         for j in range(1, m+1):
#             tmp = dp[j]
#             if a_tokens[i-1] == b_tokens[j-1]:
#                 dp[j] = prev + 1
#             else:
#                 dp[j] = max(dp[j], dp[j-1])
#             prev = tmp
#     return dp[m]

def lcs_len(a, b):
    # a, b son listas de tokens
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]


def rouge_l_f1(pred: str, gold: str) -> float:
    pt, gt = tokenize_norm(pred), tokenize_norm(gold)
    if not pt or not gt: return 0.0
    lcs = lcs_len(pt, gt)
    prec = lcs/len(pt); rec = lcs/len(gt)
    if (prec+rec)==0: return 0.0
    return 2*prec*rec/(prec+rec)

# 3) Run e coleta de métricas
metrics = {
    "answer_em": 0,
    "answer_F1": [],
    "answer_bleu": [],
    "answer_Rouge/L": []
}

# Limpia archivo si ya existía
open(output_file, "w", encoding="utf-8").close()

for ex in tqdm(dataset_miniKGraph):
    id = ex["id"]
    question       = ex["question"]
    golds = sorted(set(flatten_answers(ex["answer"])))   # dedupPara
    out     = chain.invoke({"query": question})
    print("🛰️ Context Output del chain:")
    # Contexto
    print(json.dumps(out["intermediate_steps"][1]["context"], indent=2, ensure_ascii=False))
    contexts = out["intermediate_steps"][1]["context"]
    if isinstance(contexts, str):
        contexts = [contexts]
    # Normaliza a resposta do modelo
    pred    = normalize(out["result"]) 

    print(f"✅ Question: {question}")
    print(f"✅ Golds: {golds}")
    print(f"✅ Answer: {pred}")
    
    normalized_pred = normalize(pred)
    # Exact-Match: flexible con SequenceMatcher
    # exact_match = any(
    # is_close_match(normalize(gold), normalized_pred, threshold=0.9) 
    # for gold in golds
    # )
    exact_match = any(normalize(gold) in normalized_pred for gold in golds)
    if exact_match:
        metrics["answer_em"] += 1
    print(f"✅ Exact Match: {metrics['answer_em']}")
    
    # Token-F1 (estilo SQuAD)
    # Superposición de tokens entre la predicción y la(s) referencia(s) tras una normalización simple.
    best_f1 = best_token_f1(pred, golds)
    metrics["answer_F1"].append(best_f1)

    # BLEU (corpus-bleu por sentença)
    bleu = sacrebleu.corpus_bleu(pred, golds)
    metrics["answer_bleu"].append(bleu.score)

    # ROUGE-L
    best_rouge = 0.0
    for g in golds:
        best_rouge = max(best_rouge, rouge_l_f1(pred, g))
    metrics["answer_Rouge/L"].append(best_rouge)

    # Construir dataset en formato HuggingFace
    ragas_data = Dataset.from_dict({
        "question": [question],
        "contexts": [contexts],
        "answer": [pred],
        "ground_truth": [golds],
    })

    # Métricas de RAGAS
    result_RAGAS = evaluate(ragas_data,
        metrics=[faithfulness, context_recall, context_precision]
    )
    ragas_results.append(result_RAGAS)

    # Guarda resultado inmediatamente en JSONL
    result_data = {
        "id": id,
        "question": question,
        "gold": golds,
        "pred": pred,
        "exact_match": exact_match,
        "f1_score": best_f1,
        "bleu_score": bleu.score,
        "rouge_l": rouge_l,
        "ragas": result_RAGAS.to_dict()
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_data, indent=4, ensure_ascii=False) + "\n")

# 4) Agrega resultados
n = len(dataset_miniKGraph)
em_score = metrics['answer_em']/n
f1_score_avg = sum(metrics['answer_f1'])/n
bleu_score_avg = sum(metrics['answer_bleu'])/n
rouge_l_avg = sum(metrics['answer_Rouge/L'])/n

final_ragas = {}
for m in ["context_relevancy", "faithfulness", "answer_relevancy", "context_recall"]:
    final_ragas[m] = sum(r[m] for r in ragas_results) / len(ragas_results)

# Imprime métricas finales de RAGAS
print(f" * * * MÉTRICAS FINALES DE RAGAS * * *")
for m, v in final_ragas.items():
    print(f"{m}: {v:.2f}")

print(f" * * * MÉTRICAS FINALES * * *")
print(f"Answer EM (≥90% match):   {em_score:.2%}")
print(f"Answer F1:   {f1_score_avg:.2%}")
print(f"Answer BLEU: {bleu_score_avg:.2f}")
print(f"Answer ROUGE-L: {rouge_l_avg:.2%}")
print(f"📁 Resultados guardados progresivamente en {output_file}")

# 5) Guardar metricas finales en archivo JSONL
final_metrics = {
    "total_examples": n,
    "Answer EM ": em_score,
    "Answer F1": f1_score_avg,
    "Answer BLEU": bleu_score_avg,
    "Answer ROUGE-L": rouge_l_avg
}

with open(final_metrics_file, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=4, ensure_ascii=False)

print("📁 Métricas finales guardadas ✅")