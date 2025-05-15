from langchain.chains import RetrievalQA

def build_rag_chain(llm, retriever):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return chain