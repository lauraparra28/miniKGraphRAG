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
from ragas.metrics import faithfulness, context_recall, context_precision, answer_relevancy
from datasets import Dataset
from ragas import evaluate
# import evaluate

# Generar nombre de archivo con fecha actual
fecha_actual = datetime.now().strftime("%d_%m_%Y")
output_file = os.path.join("results", f"Aggregation_evaluation_results_{fecha_actual}.jsonl")
final_metrics_file = os.path.join("results", f"Aggregation_final_metrics_{fecha_actual}.json")
print("📁 Arquivos criados para guardar dados do teste")

# 1) Carrega o dataset
dataset_miniKGraph = bu.load_dataset()["MiniKGraph_dataset_aggregation.json"]
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

def normalize_nested_list(nested_list):
    """Normaliza listas anidadas de strings conservando la estructura."""
    return [
        [normalize(item) for item in sublist]
        for sublist in nested_list
    ]
    
def tokenize_norm(s: str): return normalize(s).split()

# ---------- Utilidades ----------
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

# F1 a nivel de grupos (conceptual match)
def f1_group(pred: str, gold_group: list[str]) -> float:
    # Para un grupo de aliases, devuelve el mejor F1
    return max(token_f1(pred, alias) for alias in gold_group) if gold_group else 0.0

def avg_f1_per_question(pred: str, golds) -> float:
    """
    Calcula F1 promedio sobre todos los grupos de golds.
    
    pred: string con la predicción
    golds:
        - string
        - list[string] (aliases)
        - list[list[string]] (grupos de aliases)
    """
    # Caso gold simple
    if isinstance(golds, str):
        return token_f1(pred, golds)
    
    # Caso lista de aliases planos
    if isinstance(golds, list) and all(isinstance(g, str) for g in golds):
        gold_str = " ".join(golds)
        return token_f1(pred, gold_str)
    
    # Caso lista de listas de aliases (grupos)
    if isinstance(golds, list) and all(isinstance(g, list) for g in golds):
        f1_scores = []
        for alias_group in golds:
            # Para cada grupo, tomar el mejor F1 entre los aliases
            alias_f1s = [token_f1(pred, alias) for alias in alias_group]
            f1_scores.append(max(alias_f1s))
        # F1 promedio sobre todos los grupos
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    raise ValueError("Formato de golds no reconocido")

def to_group_format(ans):
    # Caso string
    if isinstance(ans, str):
        return [[normalize(ans)]]
    # Caso lista plana de strings
    if isinstance(ans, list) and all(isinstance(x, str) for x in ans):
        return [[x] for x in ans]
    # Caso lista de listas (correcto ya)
    if isinstance(ans, list) and all(isinstance(x, list) for x in ans):
        return ans
    # fallback
    return [[normalize(ans)]]

# ---------------------------
# F1 por grupos (any-match)
# ---------------------------
def group_f1(pred: str, gold_groups: list[list[str]]) -> float:
    ptoks = normalize(pred)
    pred_entities = set(ptoks.split())  # ⚠️ aquí podrías usar regex o parser más fino
    
    matched_groups = 0
    for group in gold_groups:
        group_norm = [normalize(x) for x in group]
        if any(g in ptoks for g in group_norm):  # basta con acertar 1 variante
            matched_groups += 1
    
    if not pred_entities or not gold_groups:
        return 0.0
    # --- Definiciones para Precisión y Recall ---
    tp = matched_groups  # Verdaderos Positivos
    # Para este enfoque extractivo, el número total de "predicciones positivas"
    # es simplemente la cantidad de grupos que encontramos. No hay falsos positivos.
    total_predicted_positives = tp
    total_actual_positives = len(gold_groups)

    precision = tp / total_predicted_positives if total_predicted_positives > 0 else 0.0
    recall = tp / total_actual_positives if total_actual_positives > 0 else 0.0
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    group_f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return group_f1_score

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

# def rougeL_max_hf(pred: str, golds: list[str]) -> dict:
#     """
#     Calcula ROUGE-L con evaluate y devuelve el mejor puntaje (F1, P, R) contra una lista de golds.
#     """
#     rouge = evaluate.load("rouge")
#     pred_n = normalize(pred)
    
#     best = {"rougeL_f": 0.0, "rougeL_p": 0.0, "rougeL_r": 0.0, "best_gold": None}
#     for g in golds:
#         g_n = normalize(g)
#         # evaluate.rouge espera listas paralelas de preds y refs
#         res = rouge.compute(predictions=[pred_n], references=[g_n], rouge_types=["rougeL"])
#         # 'rougeL' devuelve F1; precision y recall no vienen explícitos en evaluate
#         # (si necesitas P/R explícitos, usa la opción B de abajo)
#         f1 = res["rougeL"]
#         if f1 > best["rougeL_f"]:
#             best.update({"rougeL_f": f1, "best_gold": g})
#     return best

# 3) Run e coleta de métricas
metrics = {
    "answer_em": 0,
    "answer_f1_score": [],
    #"answer_avg_f1": [],
    "answer_group_f1": [],
    "answer_bleu": [],
    "answer_Rouge/L": [],
    #"answer_rougeL_HF": []
}

# Limpia archivo si ya existía
open(output_file, "w", encoding="utf-8").close()

ragas_results = {
    "faithfulness": [],
    "context_recall": [],
    "context_precision": [],
    "answer_relevancy": []
}

ragas_results_ = []

for ex in tqdm(dataset_miniKGraph):
    id = ex["id"]
    question       = ex["question"]
    golds = sorted(set(flatten_answers(ex["answer"]))) 
    print(f"Golds (normalized): {golds}")

    out     = chain.invoke({"query": question}) 
    contexts = out["intermediate_steps"][1]["context"]

    def to_ctx_strings(ctx):
        # SIEMPRE devolver List[str]
        if ctx is None:
            return []
        if isinstance(ctx, str):
            return [ctx]
        if isinstance(ctx, list):
            out = []
            for item in ctx:
                out.extend(to_ctx_strings(item))  # recursion: lista de cosas heterogéneas
            # quita vacíos y dedup
            out = [s for s in (x.strip() for x in out) if s]
            return list(dict.fromkeys(out))
        if isinstance(ctx, dict):
            # intenta recoger valores "textuales"
            parts = []
            for v in ctx.values():
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    parts.extend(v)
            if parts:
                return ["; ".join(p for p in parts if p)]
            # último recurso: volcar el dict como json
            return [json.dumps(ctx, ensure_ascii=False)]
        # último recurso para tipos raros
        return [str(ctx)]

    def to_ctx_strings_pretty(ctx):
        # Caso conocido de KG: [{"f.rdfs_label": ["JAPIIM", "IGARAPÉ MARIPÁ"]}, ...]
        if isinstance(ctx, list) and ctx and isinstance(ctx[0], dict) and "f.rdfs_label" in ctx[0]:
            # une todos los labels de todos los items, dedup y ordena para que quede natural
            labels = []
            for d in ctx:
                labels.extend(d.get("f.rdfs_label", []) or [])
            labels = sorted(set(x for x in labels if isinstance(x, str) and x.strip()))
            return ["; ".join(labels)] if labels else []
        # Para todo lo demás, fallback genérico
        return to_ctx_strings(ctx)


    contexts = to_ctx_strings_pretty(contexts)
    #print(f"✅ Contexts: {contexts}")
    # Normaliza a resposta do modelo
    pred    = normalize(out["result"]) 

    print(f"✅ Question: {question}")
    print(f"✅ Golds: {golds}")
    print(f"✅ Answer: {pred}")
    
    normalized_pred = normalize(pred)
    # Exact Match
    exact_match = any(normalize(gold) in normalized_pred for gold in golds)
    if exact_match:
        metrics["answer_em"] += 1
    print(f"✅ Exact Match: {metrics['answer_em']}")

    # Token-F1 (estilo SQuAD)
    # Superposición de tokens entre la predicción y la(s) referencia(s) tras una normalización simple.
    best_f1 = best_token_f1(pred, golds)
    metrics["answer_f1_score"].append(best_f1)

    # F1 promedio sobre todas las referencias
    #avg_f1 = avg_f1_per_question(pred, golds)
    #metrics["answer_avg_f1"].append(avg_f1)

    # Group F1 (any-match) ACTUALIZADA
    gold_groups = to_group_format(ex["answer"]) # conserva grupos para group_f1
    print(f"Golds (grouped): {gold_groups}")
    gold_groups_norm = normalize_nested_list(gold_groups)
    print(f"Gold groups (normalized): {gold_groups_norm}")
    group_f1_score = group_f1(pred, gold_groups_norm)
    metrics["answer_group_f1"].append(group_f1_score)
    print(f"✅ Group F1: {group_f1_score:.2%}")

    # BLEU (corpus-bleu por sentença)
    bleu = sacrebleu.sentence_bleu(pred, golds)
    metrics["answer_bleu"].append(bleu.score)

    # ROUGE-L
    best_rouge = 0.0
    for g in golds:
        best_rouge = max(best_rouge, rouge_l_f1(pred, g))
    metrics["answer_Rouge/L"].append(best_rouge)
    
    # ROUGE-L con evaluate (HF)
    # rougeL_hf = rougeL_max_hf(pred, golds)
    # metrics["answer_rougeL_HF"].append(rougeL_hf["rougeL_f"])
    # print(f"✅ ROUGE-L (HF): {rougeL_hf['rougeL_f']:.2%} (best gold: {rougeL_hf['best_gold']})")

    ground_truth_str = ", ".join(golds)
    print(f"✅ Gold String: {ground_truth_str}")
    # Construir dataset en formato HuggingFace
    ragas_data = Dataset.from_dict({
        "question": [question],
        "contexts": [contexts],
        "answer": [pred],
        "ground_truth": [ground_truth_str],
    })

    # Métricas de RAGAS
    result_RAGAS = evaluate(ragas_data,
        metrics=[faithfulness, context_recall, context_precision, answer_relevancy]
    )

    # convertir a dict simple
    ragas_metrics = result_RAGAS.to_pandas().iloc[0].to_dict()
    ragas_results_.append(ragas_metrics)

    ragas_results["faithfulness"].append(ragas_metrics["faithfulness"])
    ragas_results["context_recall"].append(ragas_metrics["context_recall"])
    ragas_results["context_precision"].append(ragas_metrics["context_precision"])
    ragas_results["answer_relevancy"].append(ragas_metrics["answer_relevancy"])

    # print(f"✅ RAGAS Metrics")
    # print("--------------------------------------------------")
    # print(f"Faithfulness (RAGAS): {ragas_results['faithfulness']}")
    # print(f"Context Recall (RAGAS): {ragas_results['context_recall']}")
    # print(f"Context Precision (RAGAS): {ragas_results['context_precision']}")

    # Guarda resultado inmediatamente en JSONL
    result_data = {
        "id": id,
        "question": question,
        "gold": golds,
        "pred": pred,
        "exact_match": exact_match,
        "f1_score": best_f1,
        #"avg_f1": avg_f1,
        "group_f1": group_f1_score,
        "bleu_score": bleu.score,
        "rouge_l": best_rouge,
        #"answer_rougeL_HF": rougeL_hf,
        "ragas": {
            "faithfulness": result_RAGAS["faithfulness"],
            "context_recall": result_RAGAS["context_recall"],
            "context_precision": result_RAGAS["context_precision"],
            "answer_relevancy": result_RAGAS["answer_relevancy"]
        }
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_data, indent=4, ensure_ascii=False) + "\n")

# 4) Agrega resultados
n = len(dataset_miniKGraph)
em_score = metrics['answer_em']/n
f1_score_avg = sum(metrics['answer_f1_score'])/n
#avg_f1 = sum(metrics['answer_avg_f1'])/n
group_f1_avg = sum(metrics['answer_group_f1'])/n
bleu_score_avg = sum(metrics['answer_bleu'])/n
rouge_l_avg = sum(metrics['answer_Rouge/L'])/n
#answer_rougeL_HF = sum(metrics['answer_rougeL_HF'])/n
faithfulness_avg = sum(ragas_results['faithfulness'])/n
context_recall_avg = sum(ragas_results['context_recall'])/n
context_precision_avg = sum(ragas_results['context_precision'])/n
answer_relevancy_avg = sum(ragas_results['answer_relevancy'])/n
# Calcula métricas finales de RAGAS

print(f" * * * MÉTRICAS FINALES * * *")
print(f"Answer EM:   {em_score:.2%}")
print(f"Answer F1:   {f1_score_avg:.2%}")
#print(f"Answer Avg F1: {avg_f1:.2%}")
print(f"Answer Group F1: {group_f1_avg:.2%}")
print(f"Answer BLEU: {bleu_score_avg:.2f}")
print(f"Answer ROUGE-L: {rouge_l_avg:.2%}")
#print(f"Answer ROUGE-L (HF): {answer_rougeL_HF:.2%}")

print(f" * * * MÉTRICAS FINALES DE RAGAS * * *")
print(f"Faithfulness: {faithfulness_avg:.3%}")
print(f"Answer Relevancy: {answer_relevancy_avg:.3%}")
print(f"Context Recall: {context_recall_avg:.3%}")
print(f"Context Precision: {context_precision_avg:.3%}")


print(f"📁 Resultados guardados progresivamente en {output_file}")

# 5) Guardar metricas finales en archivo JSONL
final_metrics = {
    "total_examples": n,
    "Answer EM ": em_score,
    "Answer F1": f1_score_avg,
    "Answer Group F1": group_f1_avg,
    "Answer BLEU": bleu_score_avg,
    "Answer ROUGE-L": rouge_l_avg,
    "RAGAS Faithfulness": faithfulness_avg,
    "RAGAS Answer Relevancy": answer_relevancy_avg,
    "RAGAS Context Recall": context_recall_avg,
    "RAGAS Context Precision": context_precision_avg
    
}

with open(final_metrics_file, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=4, ensure_ascii=False)

print("📁 Métricas finales guardadas ✅")