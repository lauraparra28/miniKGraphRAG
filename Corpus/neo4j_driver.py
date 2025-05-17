from neo4j import GraphDatabase

TEMPLATES = {
    "located_in": "{source} está localizado em {target}.",
    "crosses": "{source} atravessa {target}.",
    "has_age": "{source} tem idade geológica {target}.",
    "part_of": "{source} faz parte de {target}.",
    "constituted_by": "{source} é constituído por {target}.",
    "carrier_of": "{source} é portador de {target}.",
    "labels": "{source} é chamado {target}.",
    
}

class KGToText:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def extract_triples(self):
        query = """
        MATCH (s)-[r]->(o)
        WHERE s.rdfs_label IS NOT NULL AND o.rdfs_label IS NOT NULL
        RETURN s.rdfs_label AS source, type(r) AS relation, o.rdfs_label AS target
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [{"source": r["source"], "relation": r["relation"], "target": r["target"]}
                    for r in result]

    def convert_to_text(self, triples):
        sentences = []
        for triple in triples:
            template = TEMPLATES.get(triple["relation"])
            if template:
                sentence = template.format(source=triple["source"], target=triple["target"])
                sentences.append(sentence)
        return sentences

# Example usage
kg = KGToText("bolt://localhost:7687", "neo4j", "diripar8$")
triples = kg.extract_triples()
sentences = kg.convert_to_text(triples)

with open("knowledge_corpus.txt", "w") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

for s in sentences:
    print(s)
kg.close()