import streamlit as st
import logging
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings  # Change to Ollama embeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os

EMBEDDING_MODEL = "mistral"  # Use same model as LLM for embeddings
LLM_MODEL = "mistral"
VECTORSTORE_DIR = "vectorstores"

@st.cache_resource
def get_embeddings():
    try:
        return OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        logging.error(f"Embedding model load error: {e}")
        st.stop()

@st.cache_resource
def get_llm():
    return OllamaLLM(model=LLM_MODEL, temperature=0.1, num_ctx=2048, top_k=20, top_p=0.9, repeat_penalty=1.1)

def get_existing_vectorstores():
    return [os.path.join(VECTORSTORE_DIR, f) for f in os.listdir(VECTORSTORE_DIR) 
            if os.path.isdir(os.path.join(VECTORSTORE_DIR, f))]

def get_qa_chain(docs=None, k=3):
    llm = get_llm()
    embeddings = get_embeddings()
    if docs:
        try:
            db = FAISS.from_documents(docs, embeddings)
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            logging.error(f"Embedding error: {e}")
            return None
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20})
    else:
        all_docs = []
        for vs_path in get_existing_vectorstores():
            try:
                db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
                all_docs.extend([doc for doc in db.docstore._dict.values()])
            except Exception as e:
                logging.warning(f"Error loading {vs_path}: {e}")
        if not all_docs:
            return None
        try:
            db = FAISS.from_documents(all_docs, embeddings)
        except _common.GoogleGenerativeAIError as e:
            st.error(f"Google Generative AI embedding error: {str(e)}")
            logging.error(f"Embedding error: {e}")
            return None
        retriever = db.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )