# AnswerMind AI ⚖️🤖

AnswerMind AI is a Retrieval-Augmented Generation (RAG) web application tailored for legal queries on Indian legal frameworks—including the **Constitution of India**, **Indian Penal Code (IPC)**, and **Bharatiya Nyaya Sanhita (BNS)**.

Built with Flask, ChromaDB, Ollama, and LangChain, it provides streaming, advocate-style answers backed by real-time source citations and voice interaction.

---

## ✨ Features

- 📜 **Indian Legal Focus**: Specialized in answering queries with awareness of both historical statutes (IPC) and current frameworks (BNS) alongside the Constitution.
- ⚡ **Real-Time SSE Streaming**: Token-by-token streaming responses delivered over Server-Sent Events (SSE).
- 📌 **Source Citation Panel**: Dynamically displays extracted source sections, articles, and context excerpts alongside responses.
- 🎙️ **Voice Integration**: Hands-free voice query input (Speech Recognition) and response read-aloud capabilities (Text-to-Speech).
- 🔒 **User Authentication**: Built-in signup, login, and secure session handling backed by SQLite and Werkzeug password hashing.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Werkzeug, SQLite
- **RAG & AI Framework**: LangChain, ChromaDB, Ollama (`llama3`, `nomic-embed-text`)
- **Document Processing**: PyPDF, LangChain Text Splitters
- **Frontend**: HTML5, Modern CSS, Vanilla JavaScript, Marked.js, DOMPurify

---

## 📁 Repository Structure

```text
├── app.py                  # Main Flask application and routes
├── ingest_legal_docs.py    # Document ingestion script for ChromaDB
├── logic/
│   └── rag.py              # Core RAG pipeline, vector search, and Ollama integration
├── data/                   # Directory to place legal PDF documents
├── templates/              # HTML views (dashboard, chat, login, signup)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # MIT License
```

---

## 🚀 Getting Started
1. Prerequisites
Python 3.9+
Ollama installed and running locally.
Pull the necessary Ollama models:
Bash
ollama pull llama3:latest
ollama pull nomic-embed-text
2. Installation
Clone the repository and set up a virtual environment:
```bash
git clone [https://github.com/your-username/answermind-ai.git](https://github.com/your-username/answermind-ai.git)
cd answermind-ai

python3 -m venv venv
source venv/bin/venv/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Ingest Legal Documents
Place your legal reference PDF files into the data/ folder, then run the ingestion script to create vector embeddings:
Bash
python ingest_legal_docs.py
4. Run the Application
Start the Flask server:
Bash
python app.py
Open your browser and navigate to http://localhost:5000.


## 🔐 Environment Variables (Optional)
You can customize runtime settings via environment variables or a .env file:
|Variable |	Default Value |	Description |
|----------|--------------|-------------|
| FLASK_SECRET_KEY |	dev-secret-change-this |  Flask session key |
| OLLAMA_BASE_URL |	http://localhost:11434 |	Base URL for Ollama instance |
| OLLAMA_MODEL |	llama3:latest |	Chat LLM model name |
| OLLAMA_EMBED_MODEL |	nomic-embed-text |	Embedding model name |
| CHROMA_PERSIST_DIR |	chroma_db |	Path to persistent Chroma DB directory|


---

## ⚠️ Disclaimer
AnswerMind AI provides informational assistance only and does not constitute formal legal advice. Always consult a qualified advocate for official legal guidance.
