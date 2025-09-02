# Avalia o desempenho do modelo RAG em um dataset de teste

import unicodedata
import re
import json
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score
import sacrebleu
from main_neo4j import chain
from utils import base_utils as bu

# Generar nombre de archivo con fecha actual
fecha_actual = datetime.now().strftime("%d_%m_%Y")
output_file = os.path.join("results", f"evaluation_results_{fecha_actual}.jsonl")
final_metrics_file = os.path.join("results", f"final_metrics_{fecha_actual}.json")
print("📁 Arquivos criados para guardar dados do teste")

# 1) Carrega o dataset
dataset_miniKGraph = bu.load_dataset()["MiniKGraph_text_dataset_balanced_2908.json"] # Dataset de teste MiniKGraph_teste.json
test_examples = dataset_miniKGraph
print("✅ Successfully load Dataset miniKGraph for Evaluation")

# 2) Funções auxiliares
# Normaliza as respostas, removendo espaços extras e convertendo para minúsculas
def normalize(text: str) -> str:
    # Reemplazar saltos de línea y tabulaciones por espacio
    text = text.replace("\n", " ").replace("\t", " ")
    # Eliminar acentos
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Eliminar espacios extra
    return text.strip()

def flatten_answers(ans):
    # Ans vem como List[List[str]] ou List[str]
    if isinstance(ans, list) and ans and isinstance(ans[0], list):
        return [normalize(a) for sub in ans for a in sub]
    elif isinstance(ans, list):
        return [normalize(a) for a in ans]
    else:
        return [normalize(ans)]

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
    golds   = flatten_answers(ex["answer"])
    out     = chain.invoke({"query": question})
    # Normaliza a resposta do modelo
    pred    = normalize(out["result"]) 

    print(f"✅ Question: {question}")
    print(f"✅ Golds: {golds}")
    print(f"✅ Answer: {pred}")
    
    # Exact-Match: pred exatamente igual a um dos golds?
    normalized_pred = normalize(pred)
    exact_match = any(normalize(gold) in normalized_pred for gold in golds)
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
print(f" * * * MÉTRICAS FINALES * * *")
print(f"Answer EM:   {metrics['answer_em']/n:.2%}")
print(f"Answer F1:   {sum(metrics['answer_f1'])/n:.2%}")
print(f"Answer BLEU: {sum(metrics['answer_bleu'])/n:.2f}")
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