# Avalia o desempenho do modelo RAG em um dataset de teste

import unicodedata
import re
import json
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score
from difflib import SequenceMatcher
import sacrebleu
from main_neo4j import chain
from utils import base_utils as bu

# Generar nombre de archivo con fecha actual
fecha_actual = datetime.now().strftime("%d_%m_%Y")
output_file = os.path.join("results", f"evaluation_results_{fecha_actual}.jsonl")
final_metrics_file = os.path.join("results", f"final_metrics_{fecha_actual}.json")
print("📁 Arquivos criados para guardar dados do teste")

# 1) Carrega o dataset
dataset_miniKGraph = bu.load_dataset()["MiniKGraph_text_dataset_balanced_1009.json"] # Dataset de teste MiniKGraph_teste.json
test_examples = dataset_miniKGraph
print("✅ Successfully load Dataset miniKGraph for Evaluation")

# 2) Funções auxiliares
# Normaliza as respostas, removendo espaços extras e convertendo para minúsculas
def normalize(text: str) -> str:
    import unicodedata, re
    if text is None:
        return ""
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
    pt, gt = tokenize_norm(pred), tokenize_norm(gold)
    if not pt or not gt: return 0.0
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    if overlap == 0: return 0.0
    prec, rec = overlap/len(pt), overlap/len(gt)
    return 2*prec*rec/(prec+rec)

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
    "answer_f1": [],
    "answer_bleu": []
}

# Limpia archivo si ya existía
open(output_file, "w", encoding="utf-8").close()

for ex in tqdm(test_examples):
    id = ex["id"]
    question       = ex["question"]
    golds = sorted(set(flatten_answers(ex["answer"])))   # dedupPara
    out     = chain.invoke({"query": question})
    print("🛰️ Context Output del chain:")
    print(json.dumps(out["intermediate_steps"][1]["context"], indent=2, ensure_ascii=False))
    # Normaliza a resposta do modelo
    pred    = normalize(out["result"]) 

    print(f"✅ Question: {question}")
    print(f"✅ Golds: {golds}")
    print(f"✅ Answer: {pred}")
    
    # Exact-Match: flexible con SequenceMatcher
    normalized_pred = normalize(pred)
    exact_match = any(
    is_close_match(normalize(gold), normalized_pred, threshold=0.9) 
    for gold in golds
    )
    if exact_match:
        metrics["answer_em"] += 1
    print(f"✅ Exact Match: {metrics['answer_em']}")
    
    # F1 token-level: comparando com _cada_ gold e pegando o max
    best_f1 = 0
    pred_tokens = pred.split()
    for g in golds:
        gold_tokens = g.split()
        # cria rótulos binários: token presente em ambos?
        y_true  = [1]*len(gold_tokens) + [0]*len(pred_tokens)
        y_pred  = [1 if t in pred_tokens else 0 for t in gold_tokens] + [0]*len(pred_tokens)
        best_f1 = max(best_f1, f1_score(y_true, y_pred, zero_division=0))
    metrics["answer_f1"].append(best_f1)

    # BLEU (corpus-bleu por sentença)
    bleu = sacrebleu.sentence_bleu(pred, golds)
    metrics["answer_bleu"].append(bleu.score)

    # Guarda resultado inmediatamente en JSONL
    result_data = {
        "id": id,
        "question": question,
        "gold": golds,
        "pred": pred,
        "exact_match": exact_match,
        "f1_score": best_f1,
        "bleu_score": bleu.score
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_data, indent=4, ensure_ascii=False) + "\n")

# 4) Agrega resultados
n = len(test_examples)
em_score = metrics['answer_em']/n
f1_score_avg = sum(metrics['answer_f1'])/n
bleu_score_avg = sum(metrics['answer_bleu'])/n

print(f" * * * MÉTRICAS FINALES * * *")
print(f"Answer EM (≥90% match):   {em_score:.2%}")
print(f"Answer F1:   {f1_score_avg:.2%}")
print(f"Answer BLEU: {bleu_score_avg:.2f}")
print(f"📁 Resultados guardados progresivamente en {output_file}")

# 5) Guardar metricas finales en archivo JSONL
final_metrics = {
    "total_examples": n,
    "answer_em": metrics['answer_em']/n,
    "answer_f1": sum(metrics['answer_f1'])/n,
    "answer_bleu": sum(metrics['answer_bleu'])/n
}

with open(final_metrics_file, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=4, ensure_ascii=False)

print("📁 Métricas finales guardadas ✅")