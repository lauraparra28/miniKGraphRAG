from retriever_owl import OWLRetriever
from llm_ollama import load_llm
from rag_chain import build_rag_chain

def main():
    owl_file = "/home/lauraparra85/ICA/miniKGraph-RAG/data/miniOntoGeoLogicaInstanciasRelacoes_v2.owl"
    question = "Em qual bacia está localizado o campo CAMP_CD_CAMPO_0888?"
    #Which fields are located in BASE_CD_BACIA_030?"
    #question2 = Which geological formations are composed of sandstone?
    retriever = OWLRetriever(owl_file)
    llm = load_llm()
    chain = build_rag_chain(llm, retriever)

    result = chain.invoke({"query": question})

    print("\n✅ Pregunta ")
    print(question)
    print("\n✅ Respuesta generada ")
    print(result['result'])

    print("\n✅ Hechos del grafo utilizados ")
    for doc in result['source_documents']:
        print(doc.page_content)

if __name__ == "__main__":
    main()
