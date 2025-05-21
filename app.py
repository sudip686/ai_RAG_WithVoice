import os
import sys
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import shutil
import docx
import whisper
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from streamlit_mic_recorder import mic_recorder
from streamlit_autorefresh import st_autorefresh

import streamlit as st
from pdf_handler import load_and_process_pdf
from docx_handler import load_and_process_docx
from llm_utils import get_embeddings, get_llm, get_existing_vectorstores, get_qa_chain
from langchain_ollama import OllamaLLM

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

# Ensure FFMPEG is discoverable
ffmpeg_dir = r"C:\\Users\\SUDIPTA CHANDA\\models\\ffmpeg-7.1.1-essentials_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_dir

# App-wide constants
VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("document_qa.log")
    ]
)
logger = logging.getLogger(__name__)

# Page setup
st.set_page_config(page_title="Smart Document Q&A", layout="wide")
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .block-container { padding: 1.5rem; }
    .stTextInput input { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Smart Document Q&A")
st.markdown("Ask natural language questions based on the contents of your uploaded PDF or Word documents.")
st.divider()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = {}

# Sidebar - Document Management
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload document (PDF or DOCX):", type=["pdf", "docx"])
    k_value = st.slider("🔎 Top K results to retrieve:", 1, 10, 3)

    all_entries = get_existing_vectorstores()
    docs = []
    queue_statuses = []

    for path in all_entries:
        base = os.path.basename(path)
        status_file = os.path.join(VECTORSTORE_DIR, base + ".status")
        if os.path.exists(status_file):
            with open(status_file) as sf:
                status = sf.read().strip()
            if status == "done":
                docs.append(path)
            else:
                queue_statuses.append((base, status))
        else:
            docs.append(path)

    selected_doc = st.selectbox("Select document to query (or All):", ["All"] + [os.path.basename(p) for p in docs])
    if queue_statuses:
        st.write("**Documents being processed:**")
        for name, stat in queue_statuses:
            st.write(f"- {name} → _{stat}_")

    st_autorefresh(interval=5000, key="auto_refresh")
    st.markdown("---")

# Upload & process document
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_name, file_ext = os.path.splitext(uploaded_file.name)
    file_ext = file_ext.lower()
    faiss_path = os.path.join(VECTORSTORE_DIR, file_name)
    status_path = os.path.join(VECTORSTORE_DIR, file_name + ".status")

    if os.path.exists(faiss_path):
        st.info("Document already exists in database")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        try:
            with st.spinner("Processing document..."):
                chunks = load_and_process_pdf(tmp_path) if file_ext == ".pdf" else load_and_process_docx(tmp_path)
            with open(status_path, "w") as f:
                f.write("processing")
            def build_and_save_index():
                try:
                    db = FAISS.from_documents(chunks, get_embeddings(), normalize_L2=True)
                    db.save_local(faiss_path)
                    with open(status_path, "w") as f:
                        f.write("done")
                except Exception as e:
                    with open(status_path, "w") as f:
                        f.write(f"error: {str(e)}")
                    logger.error("Embedding error", exc_info=True)
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(build_and_save_index)
            st.success("Document is being processed in the background.")
        except Exception as e:
            st.error(f"Error processing document: {e}")
        finally:
            os.remove(tmp_path)

# Ask a question
st.subheader("💬 Ask a Question")
with st.form("qa_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Type your question:", placeholder="e.g. What is the summary of the document?", key="question_input")
    with col2:
        audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="⏹️ Stop")
        if audio:
            st.audio(audio["bytes"])
            audio_hash = hashlib.md5(audio["bytes"]).hexdigest()
            if audio_hash in st.session_state.transcriptions:
                st.session_state.question_input = st.session_state.transcriptions[audio_hash]
                st.success("Transcription retrieved from cache!")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    tmp_audio.write(audio["bytes"])
                    tmp_audio_path = tmp_audio.name
                try:
                    if "whisper_model" not in st.session_state:
                        st.session_state.whisper_model = whisper.load_model("base")
                    result = st.session_state.whisper_model.transcribe(tmp_audio_path, temperature=0)
                    st.session_state.transcriptions[audio_hash] = result["text"]
                    st.session_state.question_input = result["text"]
                    st.success("Transcription successful!")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    os.remove(tmp_audio_path)
    submitted = st.form_submit_button("Submit")

# Answer the question
if submitted and question.strip():
    st.session_state.question_input = ""
    with st.spinner("Searching knowledge base..."):
        selected_docs = None
        if selected_doc and selected_doc != "All":
            try:
                db = FAISS.load_local(os.path.join(VECTORSTORE_DIR, selected_doc), get_embeddings(), allow_dangerous_deserialization=True)
                selected_docs = list(db.docstore._dict.values())
            except Exception as e:
                st.error(f"Could not load selected document: {e}")

                # Ensure the LLM is warmed up by explicitly calling it once
        llm = get_llm()
        try:
            _ = llm.invoke("ping")  # prime the model
        except Exception as e:
            logger.warning(f"LLM warm-up failed: {e}")
        qa_chain = get_qa_chain(docs=selected_docs, k=k_value)
        if qa_chain:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = executor.submit(qa_chain, {"query": question}).result(timeout=60)
                answer = result.get("result", "")
                st.session_state.answer_text = answer
                st.session_state.history.insert(0, {"question": question, "answer": answer})
                st.session_state.history = st.session_state.history[:10]
            except Exception as e:
                st.error(f"Error in QA chain: {e}")
        else:
            st.warning("No relevant documents to answer this question.")

# Answer display
st.markdown("---")
with st.expander("📜 Answer", expanded=True):
    if 'answer_text' in st.session_state:
        st.write(st.session_state.answer_text)
        st.download_button("📄 Download Answer", st.session_state.answer_text, file_name="answer.txt", mime="text/plain")

# History
st.markdown("---")
with st.expander("🕘 Past Questions & Answers", expanded=False):
    if "history" in st.session_state and st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            st.markdown(f"**Q{i+1}:** {entry['question']}")
            st.markdown(f"**A{i+1}:** {entry['answer']}")
            st.markdown("---")
