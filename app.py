import os
import sys
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import whisper
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from streamlit_mic_recorder import mic_recorder
from streamlit_autorefresh import st_autorefresh

import streamlit as st
from pdf_handler import load_and_process_pdf
from docx_handler import load_and_process_docx
from csv_handler import load_and_process_csv
from llm_utils import get_embeddings, get_llm, get_existing_vectorstores, get_qa_chain

import shutil

# Patch Streamlit watcher to ignore torch internal errors
if 'torch' in sys.modules:
    import streamlit.runtime.scriptrunner.script_runner
    import streamlit.watcher.local_sources_watcher
    original_get_module_paths = streamlit.watcher.local_sources_watcher.get_module_paths
    def patched_get_module_paths(module):
        if module.__name__.startswith("torch"):
            return []
        return original_get_module_paths(module)
    streamlit.watcher.local_sources_watcher.get_module_paths = patched_get_module_paths

# Set environment variable to resolve OpenMP DLL conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


ffmpeg_dir = r"C:\\Users\\SUDIPTA CHANDA\\models\\ffmpeg-7.1.1-essentials_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_dir

# Constants
VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s - %(levelname)s - %(message)s",
                   handlers=[logging.StreamHandler(), logging.FileHandler("app.log")])
logger = logging.getLogger(__name__)

# Page setup
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("🔍 Document Q&A")
st.markdown("Upload documents and ask questions about them.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = {}

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload document (PDF, DOCX, or CSV):", type=["pdf", "docx", "csv"])
    k_value = st.slider("Number of relevant chunks:", 1, 10, 3)
    
    # Document selection and management
    st.subheader("📚 Document Selection")
    docs = get_existing_vectorstores()
    selected_doc = st.selectbox(
        "Select document to query:",
        ["All"] + [os.path.basename(p) for p in docs]
    )
    
    # Document removal section
    if docs:
        st.subheader("🗑️ Document Management")
        doc_to_remove = st.selectbox(
            "Select document to remove:",
            [os.path.basename(p) for p in docs]
        )
        
        if st.button("Remove Selected Document"):
            try:
                doc_path = os.path.join(VECTORSTORE_DIR, doc_to_remove)
                status_path = doc_path + ".status"
                
                # Remove the vectorstore and status file
                if os.path.exists(doc_path):
                    shutil.rmtree(doc_path)
                if os.path.exists(status_path):
                    os.remove(status_path)
                    
                st.success(f"Successfully removed {doc_to_remove}")
                st.rerun()
            except Exception as e:
                st.error(f"Error removing document: {e}")
        
        if st.button("Remove All Documents"):
            try:
                # Remove all vectorstores and status files
                for doc in docs:
                    doc_path = os.path.join(VECTORSTORE_DIR, os.path.basename(doc))
                    status_path = doc_path + ".status"
                    
                    if os.path.exists(doc_path):
                        shutil.rmtree(doc_path)
                    if os.path.exists(status_path):
                        os.remove(status_path)
                
                st.success("Successfully removed all documents")
                st.rerun()
            except Exception as e:
                st.error(f"Error removing documents: {e}")

# Document processing
if uploaded_file:
    file_name = os.path.splitext(uploaded_file.name)[0]
    faiss_path = os.path.join(VECTORSTORE_DIR, file_name)
    
    if os.path.exists(faiss_path):
        st.info("Document already exists in database")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1].lower()) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            with st.spinner("Processing document..."):
                if uploaded_file.name.endswith('.pdf'):
                    chunks = load_and_process_pdf(tmp_path)
                elif uploaded_file.name.endswith('.csv'):
                    chunks = load_and_process_csv(tmp_path)
                else:
                    chunks = load_and_process_docx(tmp_path)
                db = FAISS.from_documents(chunks, get_embeddings())
                db.save_local(faiss_path)
                st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {e}")
            logger.error(f"Document processing error: {e}", exc_info=True)
        finally:
            os.remove(tmp_path)

# Q&A Interface
st.subheader("💬 Ask a Question")

# Initialize question in session state if not present
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

col1, col2 = st.columns([4, 1])

with col2:
    audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="⏹️ Stop")
    if audio and audio.get("bytes"):  # Check if audio data exists
        tmp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio["bytes"])
                tmp_audio_path = tmp_audio.name
            
            if "whisper_model" not in st.session_state:
                st.session_state.whisper_model = whisper.load_model("base")
            
            # Ensure file is properly closed before transcription
            import time
            time.sleep(0.1)  # Small delay to ensure file is written
            
            result = st.session_state.whisper_model.transcribe(tmp_audio_path)
            st.session_state.current_question = result["text"]
            st.success("Voice input captured!")
        except Exception as e:
            st.error(f"Voice transcription failed: {str(e)}")
            logger.error(f"Voice transcription error: {str(e)}", exc_info=True)
        finally:
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                try:
                    os.remove(tmp_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary audio file: {e}")

with col1:
    question = st.text_input("Type your question:", value=st.session_state.current_question, key="question_input")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            # Load selected document
            selected_docs = None
            if selected_doc != "All":
                db = FAISS.load_local(os.path.join(VECTORSTORE_DIR, selected_doc), get_embeddings())
                selected_docs = list(db.docstore._dict.values())
            
            # Get answer
            qa_chain = get_qa_chain(docs=selected_docs, k=k_value)
            if qa_chain:
                result = qa_chain({"query": question})
                answer = result["result"]
                
                # Display answer
                st.markdown("### Answer:")
                st.write(answer)
                
                # Update history
                st.session_state.history.insert(0, {
                    "question": question,
                    "answer": answer
                })
                st.session_state.history = st.session_state.history[:10]
            else:
                st.warning("No documents available to answer questions.")
        except Exception as e:
            st.error(f"Error getting answer: {e}")
            logger.error(f"QA error: {e}", exc_info=True)

# Display History
if st.session_state.history:
    st.markdown("---")
    with st.expander("📚 Question History", expanded=False):
        for i, entry in enumerate(st.session_state.history):
            st.markdown(f"**Q{i+1}:** {entry['question']}")
            st.markdown(f"**A{i+1}:** {entry['answer']}")
            st.markdown("---")