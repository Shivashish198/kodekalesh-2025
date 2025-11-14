import streamlit as st
import os
import pandas as pd
from sqlalchemy import create_engine, text
import pdfplumber
import docx
from transformers import pipeline

# --- Translation (deep_translator) and language detection (langdetect) ---
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# --- Setup directories ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MySQL Connection ---
MYSQL_USER = "lexiuser"
MYSQL_PASSWORD = "password123"
MYSQL_HOST = "localhost"
MYSQL_DB = "lexibrief_db"

engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}")

# --- Create table if not exists (now includes tags, language, summary_translated) ---
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

# --- Initialize summarization pipeline ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# === Tagging setup: try KeyBERT, fallback if not available ===
try:
    from keybert import KeyBERT
    kw_model = KeyBERT()
    def generate_tags(summary):
        if not summary or summary.strip() == "" or summary.startswith("Not enough"):
            return ""
        try:
            keywords = kw_model.extract_keywords(
                summary,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=5
            )
            tags = [kw for kw, score in keywords]
            return ", ".join(tags)
        except Exception:
            pass
except Exception:
    kw_model = None
    import re
    def generate_tags(summary):
        if not summary or summary.strip() == "" or summary.startswith("Not enough"):
            return ""
        stopwords = set([
            "the","and","of","to","in","a","is","for","that","on","with","as","by","be","this","an",
            "are","or","it","from","which","at","have","has","was","were","but","not","their","such"
        ])
        words = re.findall(r"\b[a-zA-Z]{3,}\b", summary.lower())
        freq = {}
        for w in words:
            if w in stopwords:
                continue
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        tags = [w for w, _ in top]
        return ", ".join(tags)

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

# --- Translation helper functions using deep_translator + langdetect ---
def detect_language(text):
    try:
        if not text or text.strip() == "":
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"

def translate_text(text, dest_lang):
    """
    Translate text to dest_lang using deep_translator.GoogleTranslator.
    dest_lang should be language code like 'en', 'hi', 'ta', etc.
    On any failure returns original text.
    """
    try:
        if not text or dest_lang is None:
            return text
        # deep_translator expects language codes; set source='auto'
        return GoogleTranslator(source='auto', target=dest_lang).translate(text)
    except Exception:
        return text

# --- Sidebar login form + UI language selector ---
with st.sidebar:
    st.header("🔒 Login")
    # UI language selector (affects displayed labels/text)
    ui_lang = st.selectbox("🌐 UI Language", [
        "English", "Hindi", "Marathi", "Tamil", "Telugu",
        "Kannada", "Bengali", "Gujarati", "Punjabi", "Urdu"
    ], index=0)

    # Simple mapping to language codes for deep_translator
    lang_code_map = {
        "English":"en","Hindi":"hi","Marathi":"mr","Tamil":"ta","Telugu":"te",
        "Kannada":"kn","Bengali":"bn","Gujarati":"gu","Punjabi":"pa","Urdu":"ur"
    }
    ui_lang_code = lang_code_map.get(ui_lang, "en")

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
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.rerun()
                        except Exception:
                            pass
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
            try:
                st.rerun()
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    pass

# ============================================================
# 🔐 PROTECTED SECTION STARTS HERE
# ============================================================
if st.session_state.authenticated:

    # --- Streamlit UI (main) ---
    st.set_page_config(page_title="LexiBrief+", layout="wide")
    st.title("⚖️ LexiBrief+ – AI Legal Document Management System")
    st.markdown(f"**UI language:** {ui_lang}  |  Logged in as: **{st.session_state.auth_user}**")

    # --- Track uploaded files to prevent duplicates ---
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

    # --- Upload Section ---
    upload_header = "Upload a judicial document (PDF, TXT, DOCX)"
    if ui_lang != "English":
        upload_header = translate_text(upload_header, ui_lang_code)
    st.subheader(upload_header)

    uploaded_file = st.file_uploader(
        translate_text("Choose a file", ui_lang_code),
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
            except Exception:
                st.warning(translate_text("Could not extract text from PDF.", ui_lang_code))
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
            st.warning(translate_text("No text could be extracted from this file.", ui_lang_code))
            file_text = ""

        # detect language of extracted text
        detected_lang = detect_language(file_text) if file_text else "unknown"

        # If not English, translate to English for summarization
        text_for_summary = file_text
        if detected_lang != "en" and detected_lang != "unknown":
            try:
                text_for_summary = translate_text(file_text, "en")
            except Exception:
                text_for_summary = file_text  # fallback

        # --- Summarization (English) ---
        summary_en = ""
        if len(text_for_summary) > 20:
            max_chunk = 1000
            chunks = [text_for_summary[i:i+max_chunk] for i in range(0, len(text_for_summary), max_chunk)]
            for chunk in chunks[:3]:
                summary_chunk = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                summary_en += summary_chunk[0]['summary_text'] + " "
            summary_en = summary_en.strip()
        else:
            summary_en = "Not enough text to summarize."

        # Generate tags (in English)
        tags_en = generate_tags(summary_en)

        # Translate summary and tags into UI language for display if needed
        summary_translated = summary_en
        tags_translated = tags_en
        if ui_lang_code != "en":
            try:
                summary_translated = translate_text(summary_en, ui_lang_code)
                if tags_en:
                    tag_list = [t.strip() for t in tags_en.split(",") if t.strip()]
                    translated_tags = [translate_text(t, ui_lang_code) for t in tag_list]
                    tags_translated = ", ".join(translated_tags)
            except Exception:
                summary_translated = summary_en
                tags_translated = tags_en

        # Store metadata + text + summary_en + summary_translated + tags_en + language in MySQL DB
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO files (filename, filepath, file_text, summary, summary_translated, tags, language) VALUES (:fname, :fpath, :ftext, :summary, :summary_translated, :tags, :lang)"),
                {
                    "fname": uploaded_file.name,
                    "fpath": save_path,
                    "ftext": file_text,
                    "summary": summary_en,
                    "summary_translated": summary_translated,
                    "tags": tags_en,
                    "lang": detected_lang
                }
            )

        st.success(translate_text(f"File '{uploaded_file.name}' uploaded, text extracted, summarized and tagged!", ui_lang_code))

    # --- Display stored files with delete buttons ---
    files_header = "Stored Files with Summaries"
    if ui_lang != "English":
        files_header = translate_text(files_header, ui_lang_code)
    st.subheader(files_header)

    def load_files():
        with engine.connect() as conn:
            return pd.read_sql("SELECT id, filename, uploaded_on, summary, summary_translated, tags, language FROM files ORDER BY uploaded_on DESC", conn)

    df = load_files()

    # --- Tag filter UI ---
    tag_filter_label = "Filter by tag"
    if ui_lang != "English":
        tag_filter_label = translate_text(tag_filter_label, ui_lang_code)
    st.markdown("### " + tag_filter_label)

    try:
        all_tags_df = pd.read_sql("SELECT tags FROM files", engine)
        all_tags = set()
        for t in all_tags_df["tags"]:
            if t and isinstance(t, str):
                all_tags.update([tag.strip() for tag in t.split(",") if tag.strip()])
        # For ui display, translate tags if necessary for the selector
        if ui_lang_code != "en":
            display_tag_map = {}
            for tag in sorted(all_tags):
                try:
                    display_tag_map[tag] = translate_text(tag, ui_lang_code)
                except Exception:
                    display_tag_map[tag] = tag
            tag_options_display = ["All"] + [display_tag_map[t] for t in sorted(all_tags)]
            display_to_en_tag = {display_tag_map[t]: t for t in sorted(all_tags)}
        else:
            tag_options_display = ["All"] + sorted(all_tags)
            display_to_en_tag = {}
        tag_options = tag_options_display
    except Exception:
        tag_options = ["All"]
        display_to_en_tag = {}

    selected_tag_display = st.selectbox(translate_text("Select tag to filter", ui_lang_code), tag_options, index=0)
    if selected_tag_display != "All":
        if ui_lang_code != "en" and selected_tag_display in display_to_en_tag:
            selected_tag = display_to_en_tag[selected_tag_display]
        else:
            selected_tag = selected_tag_display
        if not df.empty:
            df = df[df["tags"].str.contains(selected_tag, na=False)]

    if not df.empty:
        for index, row in df.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{row['filename']}** (Uploaded: {row['uploaded_on']})")
                lang_label = f"Language: {row.get('language', 'unknown')}"
                st.caption(lang_label)
                disp_summary = row.get('summary_translated') if row.get('summary_translated') else row.get('summary')
                if ui_lang_code == "en":
                    disp_summary = row.get('summary') or disp_summary
                st.write(disp_summary)
                tags_en = row.get('tags') or ""
                tags_display = tags_en
                if ui_lang_code != "en" and tags_en:
                    try:
                        tag_list = [t.strip() for t in tags_en.split(",") if t.strip()]
                        translated_tags = [translate_text(t, ui_lang_code) for t in tag_list]
                        tags_display = ", ".join(translated_tags)
                    except Exception:
                        tags_display = tags_en
                if tags_display:
                    st.markdown(f"**{translate_text('Tags', ui_lang_code)}:** {tags_display}")
            with col2:
                if st.button(translate_text("Delete", ui_lang_code), key=f"delete_{row['filename']}"):
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, row['filename']))
                    except FileNotFoundError:
                        pass
                    with engine.begin() as conn:
                        conn.execute(
                            text("DELETE FROM files WHERE filename = :fname"),
                            {"fname": row['filename']}
                        )
                    st.success(translate_text(f"File '{row['filename']}' deleted successfully!", ui_lang_code))
                    df = load_files()
                    break

        if not df.empty:
            st.dataframe(df)
        else:
            st.info(translate_text("No files uploaded yet.", ui_lang_code))
    else:
        st.info(translate_text("No files uploaded yet.", ui_lang_code))

    # ==============================
    # 🔎 SEMANTIC SEARCH ON SUMMARIES
    # ==============================
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    search_header = "Semantic Search on Summaries"
    if ui_lang != "English":
        search_header = translate_text(search_header, ui_lang_code)
    st.subheader(search_header)

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

    query_label = "Enter a query to find related summaries : "
    if ui_lang != "English":
        query_label = translate_text(query_label, ui_lang_code)
    query = st.text_input(query_label)

    if query:
        query_en = query
        try:
            if ui_lang_code != "en":
                query_en = translate_text(query, "en")
        except Exception:
            query_en = query

        if index is not None:
            query_vec = embedder.encode([query_en], convert_to_numpy=True)
            k = min(5, len(df_search))
            distances, indices = index.search(query_vec, k)
            st.markdown("### " + translate_text("Top Matching Summaries", ui_lang_code))
            for i, idx in enumerate(indices[0]):
                if idx < len(df_search):
                    filename = df_search.iloc[idx]['filename']
                    summary_en = df_search.iloc[idx]['summary']
                    summary_disp = summary_en
                    if ui_lang_code != "en":
                        try:
                            summary_disp = translate_text(summary_en, ui_lang_code)
                        except Exception:
                            summary_disp = summary_en
                    st.markdown(f"**{i+1}. {filename}**")
                    st.write(summary_disp)
                    st.divider()
        else:
            st.warning(translate_text("No summaries found in the database. Upload some documents first.", ui_lang_code))
    else:
        st.info(translate_text("Type a query above to search across summaries.", ui_lang_code))

    # ==============================
    # ⚖️ JUDICIAL INSIGHT PANEL
    # ==============================
    from transformers import pipeline

    insight_header = "Judicial Insight Panel"
    if ui_lang != "English":
        insight_header = translate_text(insight_header, ui_lang_code)
    st.subheader("⚖️ " + insight_header)

    @st.cache_resource
    def load_insight_model():
        return pipeline("text2text-generation", model="google/flan-t5-small")

    insight_model = load_insight_model()

    if query and index is not None and len(df_search) > 0:
        st.markdown("### " + translate_text("AI-Generated Insights", ui_lang_code))
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
        with st.spinner(translate_text("Generating judicial insights...", ui_lang_code)):
            insight_output = insight_model(prompt, max_new_tokens=100, do_sample=False)
        insight_text_en = insight_output[0]["generated_text"]
        insight_text_disp = insight_text_en
        if ui_lang_code != "en":
            try:
                insight_text_disp = translate_text(insight_text_en, ui_lang_code)
            except Exception:
                insight_text_disp = insight_text_en
        st.write(insight_text_disp)
    else:
        st.info(translate_text("Run a search above to view judicial insights.", ui_lang_code))

else:
    st.info("Please login from the sidebar to access LexiBrief+.")
