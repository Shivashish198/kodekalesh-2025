import streamlit as st
import os
import pandas as pd
from sqlalchemy import create_engine, text
import pdfplumber
import docx
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import bcrypt

# ============================================
# 🌓 THEME TOGGLE (Light / Dark Mode)
# ============================================
st.sidebar.markdown("### 🌓 Theme")
theme_choice = st.sidebar.radio("Choose Theme:", ["Light", "Dark"], index=0)

if theme_choice == "Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #0e1117 !important; color: white !important; }
        .stButton button { background-color: #4a4a4a !important; color: white !important; border-radius: 8px; }
        .stTextInput > div > div > input { background-color: #262730 !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: white !important; color: black !important; }
        .stButton button { background-color: #e0e0e0 !important; color: black !important; border-radius: 8px; }
        .stTextInput > div > div > input { background-color: #fafafa !important; color: black !important; }
        </style>
    """, unsafe_allow_html=True)

# ============================================
# DIRECTORY + DATABASE INITIALIZATION
# ============================================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MYSQL_USER = "lexiuser"
MYSQL_PASSWORD = "password123"
MYSQL_HOST = "localhost"
MYSQL_DB = "lexibrief_db"

engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
)

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            filepath VARCHAR(500),
            file_text LONGTEXT,
            summary LONGTEXT,
            summary_translated LONGTEXT,
            tags VARCHAR(500),
            language VARCHAR(20),
            uploaded_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

# ============================================
# ROLE-BASED USERS
# ============================================
USER_ROLES = {
    "client": {"password": "client@123", "role": "client"},
    "lawyer": {"password": "lawyer@123", "role": "lawyer"},
    "judge": {"password": "judge@123", "role": "judge"},
    "admin": {"password": "admin@123", "role": "admin"},
}

# Hash passwords only once
if "password_hashes" not in st.session_state:
    st.session_state.password_hashes = {
        user: bcrypt.hashpw(info["password"].encode(), bcrypt.gensalt()).decode()
        for user, info in USER_ROLES.items()
    }

# Authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "login_error" not in st.session_state:
    st.session_state.login_error = ""

# ============================================
# LANGUAGE DETECT + TRANSLATE HELPERS
# ============================================
def detect_language(text):
    try:
        if not text.strip():
            return "unknown"
        return detect(text)
    except:
        return "unknown"

def translate_text(text, dest):
    try:
        return GoogleTranslator(source="auto", target=dest).translate(text)
    except:
        return text
# ============================================
# SIDEBAR: UI LANGUAGE + LOGIN FORM (Centralized)
# ============================================
with st.sidebar:
    st.markdown("## ⚖️ LexiBrief+ — Login & Settings")

    # UI language selector
    ui_lang = st.selectbox("🌐 UI Language", [
        "English", "Hindi", "Marathi", "Tamil", "Telugu",
        "Kannada", "Bengali", "Gujarati", "Punjabi", "Urdu"
    ], index=0)

    lang_code_map = {
        "English":"en","Hindi":"hi","Marathi":"mr","Tamil":"ta","Telugu":"te",
        "Kannada":"kn","Bengali":"bn","Gujarati":"gu","Punjabi":"pa","Urdu":"ur"
    }
    ui_lang_code = lang_code_map.get(ui_lang, "en")

    st.markdown("---")
    st.markdown("### 🔒 Login")
    if not st.session_state.authenticated:
        username_input = st.text_input("Username", key="login_username")
        password_input = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if username_input in st.session_state.password_hashes:
                stored_hash = st.session_state.password_hashes[username_input].encode()
                if bcrypt.checkpw(password_input.encode(), stored_hash):
                    # success
                    st.session_state.authenticated = True
                    st.session_state.auth_user = username_input
                    st.session_state.role = USER_ROLES[username_input]["role"]
                    st.session_state.login_error = ""
                    # rerun safely
                    try:
                        st.experimental_rerun()
                    except Exception:
                        try:
                            st.rerun()
                        except Exception:
                            pass
                else:
                    st.session_state.login_error = "Incorrect username or password."
            else:
                st.session_state.login_error = "User not found."
        if st.session_state.login_error:
            st.error(st.session_state.login_error)
    else:
        st.success(f"Logged in as: {st.session_state.auth_user} ({st.session_state.role})")
        if st.button("Logout", key="logout_button"):
            st.session_state.authenticated = False
            st.session_state.auth_user = None
            st.session_state.role = None
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    pass

# ============================================
# PERMISSION HELPER
# ============================================
def allow(action: str) -> bool:
    """Return True if current role is allowed to perform action."""
    role = st.session_state.get("role", None)
    permissions = {
        "client": {
            "upload": False,
            "delete": False,
            "export_pdf": False,
            "insight": False,
            "compare": False,
            "search": True,
            "view": True
        },
        "lawyer": {
            "upload": True,
            "delete": False,
            "export_pdf": True,
            "insight": True,
            "compare": True,
            "search": True,
            "view": True
        },
        "judge": {
            "upload": True,
            "delete": True,
            "export_pdf": True,
            "insight": True,
            "compare": True,
            "search": True,
            "view": True
        },
        "admin": {
            "upload": True,
            "delete": True,
            "export_pdf": True,
            "insight": True,
            "compare": True,
            "search": True,
            "view": True
        }
    }
    return permissions.get(role, {}).get(action, False)

# ============================================
# Summarizer loader (cached)
# ============================================
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.warning("Summarizer model could not be loaded locally. Make sure transformers & model are available.")
        return None

summarizer = load_summarizer()
# ============================================================
# MAIN APP — Only visible after authentication
# ============================================================
if st.session_state.authenticated:

    st.set_page_config(page_title="LexiBrief+", layout="wide")
    st.title("⚖️ LexiBrief+ — AI Legal Document System")
    st.markdown(f"Logged in as **{st.session_state.auth_user}** ({st.session_state.role})")

    # Track uploads to prevent duplicates
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

    # ==============================
    # UPLOAD SECTION
    # ==============================
    st.subheader("📤 Upload Judicial Document (PDF / TXT / DOCX)")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx"],
        key="file_uploader_unique"
    )

    # Function to extract text
    def extract_text(path, ext):
        text = ""
        if ext == "pdf":
            try:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        ptext = page.extract_text()
                        if ptext:
                            text += ptext + "\n"
            except:
                st.error("PDF parsing failed.")

        elif ext == "txt":
            text = open(path, "r", encoding="utf-8", errors="ignore").read()

        elif ext == "docx":
            doc = docx.Document(path)
            for p in doc.paragraphs:
                text += p.text + "\n"

        return text.strip()

    # ==============================
    # FILE PROCESSING
    # ==============================
    if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.add(uploaded_file.name)

        # Save file
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Extract text
        file_text = extract_text(save_path, file_ext)
        if not file_text:
            st.warning("No text extracted.")
            file_text = ""

        # Detect language
        detected_lang = detect_language(file_text)

        # Prepare text for summarization
        to_summarize = file_text
        if detected_lang not in ["en", "unknown"]:
            try:
                to_summarize = translate_text(file_text, "en")
            except:
                pass

        # ==============================
        # SUMMARY GENERATION
        # ==============================
        summary_en = ""
        if summarizer and len(to_summarize) > 20:
            chunks = [to_summarize[i:i+900] for i in range(0, len(to_summarize), 900)]
            for chunk in chunks[:3]:
                out = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                summary_en += out[0]["summary_text"] + " "
            summary_en = summary_en.strip()
        else:
            summary_en = "Not enough content to summarize."

        # ==============================
        # TAG GENERATION
        # ==============================
        tags_en = generate_tags(summary_en)

        # Translate summary to UI language
        summary_translated = translate_text(summary_en, ui_lang_code)
        tags_translated = ", ".join([translate_text(t, ui_lang_code) for t in tags_en.split(",")]) if tags_en else ""

        # ==============================
        # INSERT INTO DATABASE
        # ==============================
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO files (filename, filepath, file_text, summary, summary_translated, tags, language)
                VALUES (:a,:b,:c,:d,:e,:f,:g)
            """), {
                "a": uploaded_file.name,
                "b": save_path,
                "c": file_text,
                "d": summary_en,
                "e": summary_translated,
                "f": tags_en,
                "g": detected_lang
            })

        st.success(f"Uploaded, summarized & tagged: **{uploaded_file.name}**")
    # ============================================================
    # 📁 STORED FILES LIST
    # ============================================================
    st.subheader("Stored Files")

    def load_files():
        with engine.connect() as conn:
            return pd.read_sql(
                "SELECT id, filename, uploaded_on, summary, summary_translated, tags, language FROM files ORDER BY uploaded_on DESC",
                conn,
            )

    df = load_files()

    # ============================================================
    # 🔍 TAG FILTER
    # ============================================================
    st.markdown("### 🔎 Filter by Tag")

    try:
        tag_data = pd.read_sql("SELECT tags FROM files", engine)

        all_tags = set()
        for t in tag_data["tags"]:
            if t:
                all_tags.update([x.strip() for x in t.split(",") if x.strip()])

        tag_options = ["All"] + sorted(all_tags)

    except Exception:
        tag_options = ["All"]

    selected_tag = st.selectbox("Select Tag", tag_options)

    if selected_tag != "All" and not df.empty:
        df = df[df["tags"].str.contains(selected_tag, na=False)]

    # ============================================================
    # 🧾 DISPLAY FILE LIST
    # ============================================================
    if df.empty:
        st.info("No files uploaded yet.")
    else:
        for idx, row in df.iterrows():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### 📌 {row['filename']}")
                st.caption(f"Uploaded: {row['uploaded_on']}")
                st.caption(f"Detected Language: {row['language']}")

                summary_display = row["summary_translated"] or row["summary"]
                st.write(summary_display)

                if row["tags"]:
                    st.markdown(f"**Tags:** {row['tags']}")

            with col2:
                # Delete button (Judge Only)
                if st.session_state.role == "Judge":
                    if st.button(f"Delete {row['filename']}", key=f"delete_{row['id']}"):
                        try:
                            os.remove(row["filepath"])
                        except:
                            pass
                        with engine.begin() as conn:
                            conn.execute(text("DELETE FROM files WHERE id = :a"), {"a": row["id"]})
                        st.success("File deleted successfully.")
                        st.rerun()

        st.dataframe(df)

    # ============================================================
    # 📤 EXPORT SUMMARY TO PDF
    # ============================================================
    st.subheader("📄 Export Summary to PDF")

    export_df = load_files()
    file_choices = export_df["filename"].tolist()

    if file_choices:
        selected_export_file = st.selectbox("Choose a file to export:", file_choices)

        if st.button("Export to PDF"):
            row = export_df[export_df["filename"] == selected_export_file].iloc[0]

            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()

            pdf.add_font("ArialUnicode", "", "fonts/arial-unicode-ms.ttf", uni=True)
            pdf.set_font("ArialUnicode", size=12)

            pdf.multi_cell(0, 10, f"Filename: {row['filename']}")
            pdf.ln(4)

            pdf.set_font("ArialUnicode", "B", 14)
            pdf.multi_cell(0, 10, "Summary:")
            pdf.set_font("ArialUnicode", size=12)
            pdf.multi_cell(0, 8, row["summary_translated"] or row["summary"])

            pdf.ln(4)
            pdf.set_font("ArialUnicode", "B", 14)
            pdf.multi_cell(0, 10, "Tags:")
            pdf.set_font("ArialUnicode", size=12)
            pdf.multi_cell(0, 8, row["tags"] or "None")

            export_path = f"{row['filename']}_summary.pdf"
            pdf.output(export_path)

            with open(export_path, "rb") as f:
                st.download_button(
                    label="⬇ Download PDF",
                    data=f,
                    file_name=export_path,
                    mime="application/pdf",
                )

            st.success("PDF Exported Successfully!")
    else:
        st.info("Upload a file first to enable PDF export.")
    # ============================================================
    # 🔎 SEMANTIC SEARCH — FAISS VECTOR SEARCH
    # ============================================================
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    st.subheader("🔍 Semantic Search on Summaries")

    @st.cache_resource
    def load_embedder():
        return SentenceTransformer("all-MiniLM-L6-v2")

    embedder = load_embedder()

    def build_faiss_index():
        with engine.connect() as conn:
            df_vec = pd.read_sql("SELECT filename, summary FROM files", conn)

        if df_vec.empty:
            return None, None

        texts = df_vec["summary"].fillna("").tolist()
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

        return index, df_vec

    index, df_search = build_faiss_index()

    # ============================================================
    # 🔤 QUERY BOX (supports any language)
    # ============================================================
    query_input = st.text_input("Enter your search query:")

    if query_input:

        # Translate query to English
        try:
            query_en = GoogleTranslator(source="auto", target="en").translate(query_input)
        except:
            query_en = query_input

        if index is None:
            st.warning("No files in database to search. Upload some documents.")
        else:
            # Encode the query
            query_vec = embedder.encode([query_en], convert_to_numpy=True)

            k = min(5, len(df_search))
            distances, indices = index.search(query_vec, k)

            st.markdown("### 🔎 Top Matching Results")

            for rank, idx in enumerate(indices[0]):
                if idx < len(df_search):

                    fname = df_search.iloc[idx]["filename"]
                    summary_en = df_search.iloc[idx]["summary"]

                    # Translate summary to UI language for display
                    try:
                        summary_disp = GoogleTranslator(source="auto", target=ui_lang_code).translate(summary_en)
                    except:
                        summary_disp = summary_en

                    st.markdown(f"#### {rank+1}. 📄 **{fname}**")
                    st.write(summary_disp)
                    st.caption(f"Distance Score: {distances[0][rank]:.4f}")

                    st.divider()
    else:
        st.info("Type something above to search across summaries.")
    # ============================================================
    # ⚖️ JUDICIAL INSIGHT PANEL — AI LEGAL ANALYSIS
    # ============================================================
    from transformers import pipeline

    st.subheader("⚖️ Judicial Insight Panel")

    @st.cache_resource
    def load_insight_model():
        return pipeline("text2text-generation", model="google/flan-t5-small")

    insight_model = load_insight_model()

    if query_input and index is not None and len(df_search) > 0:

        st.markdown("### 🧠 AI-Generated Legal Insights")

        top_indices = indices[0][:3]  # best 3 matches

        combined_text = "\n".join(
            [
                df_search.iloc[i]["summary"][:500]
                for i in top_indices
                if i < len(df_search)
            ]
        )

        prompt = f"""
        Analyze the following judicial case summaries and provide structured insights:
        1. Identify the main legal issues involved.
        2. Describe the tone of the judgment (favorable / neutral / unfavorable).
        3. Summarize possible social or legal impact.
        4. Suggest any policy implications or areas needing reform.

        Case Summaries:
        {combined_text}

        Insights:
        """

        with st.spinner("Generating judicial insight..."):
            insight_raw = insight_model(prompt, max_new_tokens=200, do_sample=False)

        insight_text_en = insight_raw[0]["generated_text"]

        # Translate insights to UI language if needed
        if ui_lang_code != "en":
            try:
                insight_text = GoogleTranslator(source="auto", target=ui_lang_code).translate(insight_text_en)
            except:
                insight_text = insight_text_en
        else:
            insight_text = insight_text_en

        st.write(insight_text)

    else:
        st.info("Run a semantic search to enable judicial insights.")

# ============================================================
# 🚪 If user NOT logged in
# ============================================================
else:
    st.info("Please log in from the sidebar to access LexiBrief+.") 
