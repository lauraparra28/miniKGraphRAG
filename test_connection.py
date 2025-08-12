from neo4j import GraphDatabase

# Datos de conexión
URI = "bolt://127.0.0.1:7687"  # Usa "bolt://" si prefieres, pero "neo4j://" es más compatible
USER = "neo4j"
PASSWORD = "diripar8$"
DBNAME = "neo4j"  # Nombre de tu base de datos (por defecto "neo4j" en Desktop)

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session(database=DBNAME) as session:
        result = session.run("RETURN 1 AS ok").single()
        print(f"✅ Conexión exitosa, respuesta: {result['ok']}")
    driver.close()

except Exception as e:
    print(f"❌ Error de conexión: {e}")
# Este script verifica la conexión a Neo4j y devuelve un mensaje de éxito o error.