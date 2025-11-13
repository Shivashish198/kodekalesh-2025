import streamlit as st
import os
import pandas as pd
from sqlalchemy import create_engine, text
import pdfplumber
import docx
from transformers import pipeline

# --- Setup directories ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MySQL Connection ---
MYSQL_USER = "lexiuser"
MYSQL_PASSWORD = "password123"
MYSQL_HOST = "localhost"
MYSQL_DB = "lexibrief_db"

engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}")

# --- Create table if not exists ---
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            filepath VARCHAR(500),
            file_text LONGTEXT,
            summary LONGTEXT,
            uploaded_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

# --- Initialize summarization pipeline ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# === Simple bcrypt-based authentication (drop-in replacement) ===
import bcrypt

# --- Define users and plaintext passwords (for dev/hackathon only) ---
USER_PLAINTEXT = {
    "admin": "admin@123",
    "judge": "judge@123"
}

# --- Create hashed password mapping once per session/run ---
if "user_password_hashes" not in st.session_state:
    st.session_state.user_password_hashes = {
        user: bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        for user, pw in USER_PLAINTEXT.items()
    }

# --- Initialize auth state ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "login_error" not in st.session_state:
    st.session_state.login_error = ""

# --- Sidebar login form ---
with st.sidebar:
    st.header("🔒 Login")
    if not st.session_state.authenticated:
        username_input = st.text_input("Username", key="username_input")
        password_input = st.text_input("Password", type="password", key="password_input")
        if st.button("Login", key="login_btn"):
            if username_input in st.session_state.user_password_hashes:
                hashed = st.session_state.user_password_hashes[username_input].encode("utf-8")
                if bcrypt.checkpw(password_input.encode("utf-8"), hashed):
                    st.session_state.authenticated = True
                    st.session_state.auth_user = username_input
                    st.session_state.login_error = ""
                    st.rerun()
                else:
                    st.session_state.login_error = "Incorrect username or password."
            else:
                st.session_state.login_error = "Incorrect username or password."
        if st.session_state.login_error:
            st.error(st.session_state.login_error)
    else:
        st.success(f"Welcome, {st.session_state.auth_user}!")
        if st.button("Logout", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.auth_user = None
            st.rerun()

# ============================================================
# 🔐 PROTECTED SECTION STARTS HERE
# ============================================================
if st.session_state.authenticated:

    # --- Streamlit UI ---
    st.set_page_config(page_title="LexiBrief+", layout="wide")
    st.title(" LexiBrief+ – AI Legal Document Management System")
    st.write("Upload judicial documents for AI-powered insights.")

    # --- Track uploaded files to prevent duplicates ---
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

    # --- Upload Section ---
    uploaded_file = st.file_uploader(
        "Upload a judicial document (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"],
        key="unique_file_uploader_1"
    )

    def extract_text(file_path, file_type):
        text = ""
        if file_type == "pdf":
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except:
                st.warning("Could not extract text from PDF.")
        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_type == "docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        return text.strip()

    if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.add(uploaded_file.name)
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_ext = uploaded_file.name.split('.')[-1].lower()
        file_text = extract_text(save_path, file_ext)
        if not file_text:
            st.warning("No text could be extracted from this file.")
            file_text = ""
        summary = ""
        if len(file_text) > 20:
            max_chunk = 1000
            chunks = [file_text[i:i+max_chunk] for i in range(0, len(file_text), max_chunk)]
            for chunk in chunks[:3]:
                summary_chunk = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                summary += summary_chunk[0]['summary_text'] + " "
            summary = summary.strip()
        else:
            summary = "Not enough text to summarize."

        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO files (filename, filepath, file_text, summary) VALUES (:fname, :fpath, :ftext, :summary)"),
                {"fname": uploaded_file.name, "fpath": save_path, "ftext": file_text, "summary": summary}
            )
        st.success(f"File '{uploaded_file.name}' uploaded, text extracted, and summarized!")

    # --- Display stored files with delete buttons ---
    st.subheader("Stored Files with Summaries")

    def load_files():
        with engine.connect() as conn:
            return pd.read_sql("SELECT filename, uploaded_on, summary FROM files ORDER BY uploaded_on DESC", conn)

    df = load_files()

    if not df.empty:
        for index, row in df.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{row['filename']}** (Uploaded: {row['uploaded_on']})")
            with col2:
                if st.button("Delete", key=f"delete_{row['filename']}"):
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, row['filename']))
                    except FileNotFoundError:
                        pass
                    with engine.begin() as conn:
                        conn.execute(
                            text("DELETE FROM files WHERE filename = :fname"),
                            {"fname": row['filename']}
                        )
                    st.success(f"File '{row['filename']}' deleted successfully!")
                    df = load_files()
                    break

        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No files uploaded yet.")
    else:
        st.info("No files uploaded yet.")

    # ==============================
    # 🔎 SEMANTIC SEARCH ON SUMMARIES
    # ==============================
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    st.subheader("Semantic Search on Summaries")

    @st.cache_resource
    def load_embedder():
        return SentenceTransformer("all-MiniLM-L6-v2")

    embedder = load_embedder()

    def build_faiss_index():
        with engine.connect() as conn:
            df = pd.read_sql("SELECT filename, summary FROM files", conn)
        if df.empty:
            return None, None
        texts = df["summary"].fillna("").tolist()
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        return index, df

    index, df_search = build_faiss_index()
    query = st.text_input("Enter a query to find related summaries : ")

    if query:
        if index is not None:
            query_vec = embedder.encode([query], convert_to_numpy=True)
            k = min(5, len(df_search))
            distances, indices = index.search(query_vec, k)
            st.markdown("### Top Matching Summaries")
            for i, idx in enumerate(indices[0]):
                if idx < len(df_search):
                    st.markdown(f"**{i+1}. {df_search.iloc[idx]['filename']}**")
                    st.write(df_search.iloc[idx]['summary'])
                    st.divider()
        else:
            st.warning("No summaries found in the database. Upload some documents first.")
    else:
        st.info("Type a query above to search across summaries.")

    # ==============================
    # ⚖️ JUDICIAL INSIGHT PANEL
    # ==============================
    from transformers import pipeline

    st.subheader("⚖️ Judicial Insight Panel")

    @st.cache_resource
    def load_insight_model():
        return pipeline("text2text-generation", model="google/flan-t5-small")

    insight_model = load_insight_model()

    if query and index is not None and len(df_search) > 0:
        st.markdown("### 🧠 AI-Generated Insights")
        top_indices = indices[0][:3]
        combined_summary_text = "\n".join(
            [df_search.iloc[i]["summary"][:500] for i in top_indices if i < len(df_search)]
        )
        prompt = (
            "Analyze the following judicial case summaries and provide insights:\n"
            "1. Identify the main legal themes.\n"
            "2. Mention case tone (favorable/unfavorable/neutral).\n"
            "3. Give possible societal or policy impact.\n\n"
            f"Summaries:\n{combined_summary_text}\n\nInsights:"
        )
        with st.spinner("Generating judicial insights..."):
            insight_output = insight_model(prompt, max_new_tokens=100, do_sample=False)
        st.write(insight_output[0]["generated_text"])
    else:
        st.info("Run a search above to view judicial insights.")

else:
    st.info("Please login from the sidebar to access LexiBrief+.")
