from sentence_transformers import SentenceTransformer, util
import pandas as pd

# ================================
# 1. Frases de prueba
# ================================
sentences_pt = [
    "Aplito é uma Rocha Hipabissal de cor clara (leucocrática) caracterizada por uma textura granular alotriomórfica de grão fino, geralmente com uma composição granítica consistindo de quartzo, feldspato alcalino e plagioclásio sódico.",
    "Arcósio é um Arenito com mais de 25 por cento de feldspato e quartzo como principais constituintes; também pode conter fragmentos líticos; feldspato alcalino domina o feldspato total; a maioria dos arcósios é derivada de granitos ou gnaisses.",
    "O Hastariano é uma SUB-IDADE DA EUROPA OCIDENTAL que ENCONTRA o Ivoriano."
]

sentences_en = [
    "Aplite is a light coloured (leucocratic) Hypabyssal Rock characterised by a fine-grained allotriomorphic-granular texture, usually with a granitic composition consisting of quartz, alkali feldspar, and sodic plagioclase.",
    "Arcose is a Sandstone with feldspar more than 25 percent and quartz as major constituents; it can also contain lithic fragments; alkali-feldspar dominates the total feldspar; most arkoses are derived from granites or gneisses.",
    "The Hastarian is a WESTERN EUROPEAN SUBAGE that MEETS the Ivorian."
]

# ================================
# 2. Modelos a comparar
# ================================
models = {
    "MiniLM (Inglés)": "all-MiniLM-L6-v2",
    "MiniLM Multilingüe": "paraphrase-multilingual-MiniLM-L12-v2",
    "DistilUSE Multilingüe": "distiluse-base-multilingual-cased-v2",
    "LaBSE": "sentence-transformers/LaBSE",
    "E5 Multilingüe Base": "intfloat/multilingual-e5-base"
}

# ================================
# 3. Ejecutar benchmark
# ================================
results = []

for name, model_name in models.items():
    model = SentenceTransformer(model_name)

    emb_pt = model.encode(sentences_pt, convert_to_tensor=True)
    emb_en = model.encode(sentences_en, convert_to_tensor=True)

    # Pares equivalentes (misma posición)
    sim_equiv = [util.cos_sim(emb_pt[i], emb_en[i]).item() for i in range(len(sentences_pt))]
    
    # Pares distintos (ej: pt[0] vs en[1])
    sim_diff = util.cos_sim(emb_pt[0], emb_en[1]).item()

    results.append({
        "Modelo": name,
        "Equivalente #1": round(sim_equiv[0], 4),
        "Equivalente #2": round(sim_equiv[1], 4),
        "Equivalente #3": round(sim_equiv[2], 4),
        "Distinto": round(sim_diff, 4)
    })

# ================================
# 4. Crear DataFrame y mostrar
# ================================
df = pd.DataFrame(results)
print("\n===== Resultados Benchmark =====\n")
print(df.to_string(index=False))
