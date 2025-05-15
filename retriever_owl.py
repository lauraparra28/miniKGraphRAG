from rdflib import Graph
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from pydantic import PrivateAttr

class OWLRetriever(BaseRetriever):
    _graph: Graph = PrivateAttr()

    def __init__(self, owl_path: str):
        super().__init__()
        self._graph = Graph()
        self._graph.parse(owl_path, format="xml")
        print("✅ Grafo cargado con éxito ")

    def _query_graph(self, query_text: str):
        query = """
        SELECT ?subject ?predicate ?object
        WHERE {
            ?subject ?predicate ?object .
            FILTER (
                
                regex(str(?predicate), "located_in", "i") &&
                regex(str(?subject), "CAMP_CD_CAMPO_0888", "i")
            )
        }
        """
        #&&
              #  regex(str(?object), "BASE_CD_BACIA_030", "i") 
              
                #regex(str(?predicate), "constituted_by", "i") ||
        #regex(str(?object), "sandstone", "i")

        results = self._graph.query(query)
        facts = [f"{s} {p} {o}." for s, p, o in results]
        return facts

    def _get_relevant_documents(self, query: str):
        facts = self._query_graph(query)
        return [Document(page_content="\n".join(facts))]

    async def aget_relevant_documents(self, query: str):
        return self._get_relevant_documents(query)