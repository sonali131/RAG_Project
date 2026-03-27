# RAG_Project
Using Streamlit and GenAI LLM RAG Application
# 📚 RAG Project — Chat with Your PDF using GenAI

An interactive **Retrieval-Augmented Generation (RAG)** application built with **Streamlit + LangChain + Mistral AI**, allowing users to upload PDFs and ask questions in natural language.

---

## 🚀 Live Demo

👉https://ragproject-5j9qpoecxvgu5paydyw8qn.streamlit.app/

---

## ✨ Features

* 📄 Upload any PDF document
* 🔍 Smart text chunking & embeddings
* 🧠 Context-aware question answering (RAG)
* 💬 Chat-style interface
* ⚡ Fast semantic search using Chroma DB
* 🤖 Powered by Mistral LLM

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Mistral AI
* **Framework:** LangChain
* **Embeddings:** HuggingFace (MiniLM)
* **Vector Store:** ChromaDB

---

## 📂 Project Structure

```
RAG_Project/
│
├── app.py                 # Streamlit UI
├── create_database.py     # DB creation logic
├── main.py                # CLI RAG pipeline
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore sensitive files
└── README.md              # Project documentation
```

---

## ⚙️ Installation (Local Setup)

### 1. Clone Repository

```bash
git clone https://github.com/sonali131/RAG_Project.git
cd RAG_Project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Deploy `app.py`
4. Add API key in **Secrets**:

```toml
MISTRAL_API_KEY = "your_api_key_here"
```

---

## 🧠 How It Works

1. Upload PDF
2. Text is split into chunks
3. Chunks → converted to embeddings
4. Stored in Chroma vector DB
5. User query → similarity search
6. Relevant context → passed to LLM
7. LLM generates accurate answer

---

## 🔒 Security Notes

* `.env` file is ignored (API keys protected)
* Do NOT expose secrets in code
* Use Streamlit Secrets for deployment

---

## 📸 Screenshots (Optional)

*<img width="951" height="440" alt="Screenshot 2026-03-27 195712" src="https://github.com/user-attachments/assets/57235562-fdc6-4cb1-bf23-9f2a8d152ed7" />
*

---

## 🚀 Future Improvements

* 📄 Multiple PDF support
* 📊 Source citation display
* ⚡ Streaming responses
* 🧠 Conversational memory
* 🎨 Enhanced UI (ChatGPT-style)

---

## 👩‍💻 Author

**Sonali Mishra**
GitHub: https://github.com/sonali131

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share it

---

## 📜 License

This project is open-source and available under the MIT License.
