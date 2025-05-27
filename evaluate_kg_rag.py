# Avalia o desempenho do modelo RAG em um dataset de teste

import json
from tqdm import tqdm
from sklearn.metrics import f1_score
import sacrebleu
from main_neo4j import chain
from utils import base_utils as bu

# 1) Carrega o dataset
dataset_miniKGraph = bu.load_dataset()["MiniKGraph_teste.json"]
test_examples = dataset_miniKGraph
print("✅ Successfully load Dataset miniKGraph")

# 2) Funções auxiliares
def normalize(text: str) -> str:
    return text.strip().lower()

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

for ex in tqdm(test_examples):
    question       = ex["question"]
    golds   = flatten_answers(ex["answer"])
    out     = chain.invoke({"query": question})
    print(f"✅ Question: {question}")
    print(f"✅ Golds: {golds}")
    print(f"Pred: {out['result']}")
    # Normaliza a resposta do modelo
    pred    = normalize(out["result"]) 
    
    # Exact-Match: pred exatamente igual a um dos golds?
    if any(gold in pred.lower() for gold in golds):
        metrics["answer_em"] += 1

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

# 4) Agrega resultados
n = len(test_examples)
print(f"Answer EM:   {metrics['answer_em']/n:.2%}")
print(f"Answer F1:   {sum(metrics['answer_f1'])/n:.2%}")
print(f"Answer BLEU: {sum(metrics['answer_bleu'])/n:.2f}")
