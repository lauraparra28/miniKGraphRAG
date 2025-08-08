from neo4j import GraphDatabase
import numpy as np

# ==========================
# 1) Conexión a Neo4j
# ==========================
URI = "bolt://localhost:7687"   # Cambia si usas Aura o puerto distinto
USER = "neo4j"
PASSWORD = "diripar8$"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
print("✅ Successfully connection to Neo4j Graph")

# ==========================
# 2) Función para correr Cypher
# ==========================
def run_cypher(query, params=None):
    with driver.session() as session:
        return list(session.run(query, params or {}))

# ==========================
# 3) Proyectar el grafo en memoria
# ==========================
print("📌 Creando proyección del grafo...")
run_cypher("""
CALL gds.graph.project(
  'miniKGraph',
  ['well', 'field', 'basin', 'lithostratigraphic_unit', 'geological_time_interval', 'geological_structure', 'texture', 'rock_texture'],
  ['located_in', 'constituted_by', 'part_of', 'crosses', 'has_age', 'carrier_of', 'participates_in']
)
""")

# ==========================
# 4) Ejecutar Node2Vec
# ==========================
print("📌 Ejecutando Node2Vec...")
run_cypher("""
CALL gds.node2vec.mutate(
  'miniKGraph',
  {
    embeddingDimension: 128,
    walkLength: 80,
    iterations: 10,
    returnFactor: 1.0,
    inOutFactor: 1.0,
    mutateProperty: 'embedding'
  }
) YIELD nodeCount, computeMillis, embeddingDimension
""")

# ==========================
# 5) Extraer embeddings
# ==========================
print("📌 Extrayendo embeddings a NumPy...")
results = run_cypher("""
CALL gds.graph.streamNodeProperty('miniKGraph', 'embedding')
YIELD nodeId, propertyValue
WITH gds.util.asNode(nodeId).id AS id, propertyValue AS embedding
SET id.embedding = embedding;
""")

# Convertir a NumPy array
node_ids = []
embeddings = []

for record in results:
    node_ids.append(record["id"])
    embeddings.append(record["embedding"])
    
embeddings_array = np.array(embeddings)

print(f"✅ Shape de embeddings: {embeddings_array.shape}")
print(f"✅ Extracción completa: {len(node_ids)} nodos con embeddings de dimensión {embeddings_array.shape[1]}")
print("Embeddings shape:", embeddings_array.shape)
print("Primer embedding:", embeddings_array[0])

# ==========================
# 6) (Opcional) Guardar como archivo .npy
# ==========================
np.save("embeddings.npy", embeddings_array)
print("💾 Embeddings guardados en 'embeddings.npy'")

# ==========================
# 7) Cerrar conexión
# ==========================
driver.close()