import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub

if __name__ == '__main__':
    path = "documents/alpha_zero.pdf"
    loader = PyPDFLoader(file_path=path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore_name = "faiss_index/" + path.split("/")[1].split(".")[0]
    vectorstore.save_local(vectorstore_name)

    new_vectorstore = FAISS.load_local(
        vectorstore_name, embeddings, allow_dangerous_deserialization=True
     )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    prompt = input(f"\n\nAsk the quesiton you would like to ask of your {path.split('/')[1]} Document:\n")
    res = retrieval_chain.invoke({"input": prompt})
    print(res["answer"])