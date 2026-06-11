# kodekalesh-2025
⚖️ LexiBrief+ — AI-Powered Judicial Document Analysis System
Smart Summaries • Semantic Search • Multilingual Support • AI Insights • Secure Roles

LexiBrief+ is an AI-driven legal document management and analysis platform built for Judges, Lawyers, and Clients.
It helps users upload judicial files, extract text, auto-summarize, tag, search semantically, and generate AI insights — all with multilingual support and role-based authentication.

Designed for hackathons, courts, law firms, legal tech innovators, and researchers, LexiBrief+ provides fast, accurate, and secure judicial intelligence.

🚀 Core Features
📝 1. Smart Document Processing

Upload PDF, DOCX, TXT

Automatic text extraction

Language detection (Hindi, Marathi, Tamil, Telugu, Kannada, Bengali, Gujarati, Punjabi, Urdu, English)

Automatic translation to English for processing

🧠 2. AI-Generated Summaries

Uses Facebook BART Large-CNN transformer model

Splits long documents into chunks

Produces clear and concise summaries

Summaries translated automatically to the user's selected language

🏷️ 3. Auto-Tagging Using NLP

Uses KeyBERT (or fallback keyword extraction)

Generates up to 5 relevant tags per summary

Supports multilingual tag translation

🔎 4. Semantic Search using FAISS

Searches understanding the meaning, not just keywords

Uses:

SentenceTransformer all-MiniLM-L6-v2

FAISS vector index

Returns the most relevant case summaries

⚖️ 5. Judicial Insight Panel

AI model: FLAN-T5

Generates:

Main legal themes

Case tone (favorable/neutral/unfavorable)

Policy & social impact analysis

🌍 6. Full Multi-Language UI

User can choose display language:

English

Hindi

Marathi

Tamil

Telugu

Kannada

Gujarati

Bengali

Punjabi

Urdu

🔐 7. Role-Based Authentication

Login roles:

Judge Dashboard

Lawyer Dashboard

Client Dashboard

Secure authentication uses bcrypt hashing (no plaintext storage).

🌓 8. Light / Dark Mode

Full dark mode

CSS-based UI theming

Automatically changes colors of buttons and inputs

📂 9. File Management

Prevents duplicate uploads

Displays summary, tags, language

Delete file option

MySQL storage for:

filename

text

summary

translated summary

tags

uploaded timestamp

📄 10. Export Summary to PDF

Clean formatted PDF

Summary + Tags + Filename

Download directly from browser

🏛️ Tech Stack
Frontend

Streamlit (UI)

Custom CSS (Dark/Light mode)

Backend

Python 3.10+

BART, FLAN-T5, MiniLM (Transformers)

FAISS (Semantic Search)

KeyBERT (Tagging)

Database

MySQL (file metadata + summaries)

AI Libraries

Transformers

Sentence Transformers

FAISS

PDFPlumber

Deep Translator

LangDetect

🗄️ Project Structure
lexibrief/
│── app.py                  # Main Streamlit app
│── uploads/                # Uploaded files saved here
│── README.md               # Documentation
│── requirements.txt        # Dependency list
└── resources/              # (optional) images, logos

⚙️ Setup Instructions
1️⃣ Install Python 3.10

Download from
https://www.python.org/downloads/release/python-3100/

2️⃣ Create Virtual Environment
py -3.10 -m venv lexi_env
lexi_env\Scripts\activate

3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt


If no requirements.txt, install manually:

pip install streamlit pdfplumber python-docx deep-translator langdetect bcrypt keybert transformers sentence-transformers faiss-cpu fpdf2 pymysql sqlalchemy

4️⃣ Start MySQL Database

Create database:

CREATE DATABASE lexibrief_db;


Your code automatically creates the files table.

5️⃣ Run the App
streamlit run app.py

🔒 Role-Based Access
Role	Permissions
Judge	Full Access (upload, delete, insights, search)
Lawyer	Upload, search, read summaries
Client	View summaries only

(Handled in UI / dashboard behavior)

🧠 AI Models Used
Purpose	Model
Summarization	facebook/bart-large-cnn
Embeddings	all-MiniLM-L6-v2
Insights	google/flan-t5-small
Keyword Tagging	KeyBERT

🏁 Future Improvements

OCR for scanned PDFs

ChatGPT-powered case Q&A

Automatic case classification (civil/criminal/writ etc.)

Timeline extraction

Workflow automation for courts

Contributors:-
1) [Shivashish Mehrotra](https://github.com/Shivashish198)
2) [Suryansh Singh](https://github.com/Suryansh-2412)
3) [Prakhar Dixit](https://github.com/prakhar54dixit)
