# 📚 Local Multi-Document RAG Chatbot
### AI-Powered Retrieval-Augmented Generation using Ollama, Pinecone, MiniLM & Whisper

A production-ready **terminal-based Retrieval-Augmented Generation (RAG) chatbot** built entirely in Python.

The chatbot allows users to upload and query **multiple PDF documents simultaneously** using **text or voice**, while ensuring every answer is **strictly grounded** in the uploaded documents.

Unlike generic chatbots, this system **refuses to hallucinate**. If the requested information does not exist inside the uploaded PDFs, the assistant explicitly states that it cannot answer.

---

## ✨ Features

- 📄 Multi-PDF document ingestion
- 🔍 Semantic search using Pinecone
- 🤖 Local LLM inference with Ollama (Llama 3)
- 🎤 Voice queries using Faster-Whisper
- 💬 Conversational memory with question reformulation
- 📖 Source citations
  - Document Name
  - Page Number
  - Paragraph Number
- 📊 Confidence score for every answer
- 🚫 Hallucination prevention using relevance threshold
- ⚡ GPU acceleration (automatic CPU fallback)
- 🧠 Production-style modular architecture
- 🖥️ Fully terminal-based interface

---

# 🏗️ System Architecture

```
                   User
               (Text / Voice)
                      │
                      ▼
          Faster-Whisper (Speech → Text)
                      │
                      ▼
          Conversation Memory Manager
                      │
                      ▼
        Question Reformulation (Ollama)
                      │
                      ▼
         SentenceTransformer Embeddings
        (all-MiniLM-L6-v2 | 384 Dimensions)
                      │
                      ▼
             Pinecone Vector Database
                      │
         Similarity Search + Filtering
                      │
         Reject Low-Relevance Documents
                      │
                      ▼
             Ollama (Llama 3)
       Strict Grounded Prompt Generation
                      │
                      ▼
      Grounded Answer + Source Citations
```

---

# ⚙️ Technical Architecture

## 1. PDF Processing

- Extracts text page-by-page using **PyMuPDF (fitz)**
- Maintains document structure
- Supports multiple PDF uploads

---

## 2. Intelligent Chunking

Each PDF is processed into logical chunks while preserving metadata.

Every chunk stores:

- Document Name
- Page Number
- Paragraph Number
- Chunk ID

Default configuration

| Parameter | Value |
|-----------|------|
| Chunk Size | 500 characters |
| Chunk Overlap | 50 characters |

---

## 3. Embedding Model

Uses

> **Sentence Transformers**
>
> `all-MiniLM-L6-v2`

Features

- 384-dimensional embeddings
- Unit normalized vectors
- Fast inference
- High semantic retrieval accuracy

---

## 4. Vector Database

Uses **Pinecone** with

- Cosine Similarity
- Real-time vector search
- Cloud-hosted index
- Fast nearest-neighbor retrieval

---

## 5. Voice Input

Uses

- Faster-Whisper
- Whisper Base Model

Features

- Live microphone recording
- CUDA acceleration
- Automatic CPU fallback

---

## 6. Conversational Memory

Maintains session history.

If follow-up questions contain pronouns such as

- he
- she
- it
- they
- this
- that

the chatbot asks the local LLM to rewrite the question into a standalone query before retrieval.

Example

```
User:
Who chaired the Drafting Committee?

User:
Where was he born?
```

Rewritten internally

```
Where was Dr. B. R. Ambedkar born?
```

---

## 7. Retrieval Pipeline

```
User Question
      │
      ▼
Generate Embedding
      │
      ▼
Search Pinecone
      │
      ▼
Retrieve Top-k Chunks
      │
      ▼
Filter by Similarity Score
      │
      ▼
Ground Prompt
      │
      ▼
Ollama
      │
      ▼
Grounded Response
```

---

## 8. Hallucination Prevention

The chatbot never fabricates information.

Workflow

- Retrieve relevant chunks
- Check similarity score
- Reject chunks below threshold
- Refuse to answer if no relevant evidence exists

Default threshold

```
0.40
```

Example

```
Question

Who is Shah Rukh Khan?
```

Response

```
This question does not appear to be related to the uploaded documents.
Please ask questions based on the uploaded PDFs.
```

---

## 📂 Project Structure

```
project/
│
├── data/                       # Raw PDF documents
├── logs/                       # Runtime logs
│
├── src/
│   ├── pdf_loader.py           # PDF extraction
│   ├── chunker.py              # Chunk generation
│   ├── embeddings.py           # MiniLM embeddings
│   ├── vector_store.py         # Pinecone integration
│   ├── retriever.py            # Retrieval pipeline
│   ├── memory.py               # Chat session memory
│   └── chatbot.py              # Main RAG chatbot
│
├── .env.example
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone <repository-url>

cd project
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Install Ollama

Download from

https://ollama.com

Pull Llama 3

```bash
ollama pull llama3
```

Start Ollama

```bash
ollama serve
```

---

## 4. Configure Environment Variables

Create a `.env`

```env
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=your_index

PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

---

# ▶️ Running the Project

```bash
python main.py
```

---

# 💻 Usage

When started, the chatbot will

- Load embedding model
- Connect to Pinecone
- Check existing vectors
- Upload PDFs if necessary
- Generate embeddings
- Store vectors
- Start chat session

Commands

| Command | Description |
|----------|------------|
| clear | Reset conversation history |
| exit | Exit chatbot |

---

# 💬 Example Session

```
=========================================================
      Local Multi-Document RAG Chatbot
=========================================================

Loading embedding model...

Enter PDF paths

data/constitution.pdf,data/history.pdf

Reading PDFs...

Generating embeddings...

Uploading vectors...

Ready!
```

---

### Example 1

```
Question

Who chaired the Drafting Committee?
```

Answer

```
Dr. B. R. Ambedkar chaired the Drafting Committee.
```

Source

```
Document:
constitution.pdf

Page:
2

Paragraph:
3

Confidence:
92%
```

---

### Example 2

```
Question

Compare the Drafting Committee and the Constituent Assembly.
```

Answer

```
The Constituent Assembly was responsible for framing the Constitution,
while the Drafting Committee prepared the final draft under
Dr. B. R. Ambedkar.
```

Sources

```
constitution.pdf
Page 2
Paragraph 3

constitutional_history.pdf
Page 7
Paragraph 2
```

Confidence

```
89%
```

---

### Example 3

```
Question

Who is Shah Rukh Khan?
```

Response

```
This question does not appear to be related to the uploaded documents.

Please ask questions based on the uploaded PDFs.
```

Confidence

```
15%
```

---

# 🔧 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| LLM | Ollama (Llama 3) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | Pinecone |
| Speech-to-Text | Faster-Whisper |
| PDF Parsing | PyMuPDF |
| Similarity Metric | Cosine Similarity |
| Environment | Python 3.11+ |

---

# 🌟 Future Improvements

- Web UI
- FastAPI backend
- Authentication
- Docker support
- Streaming responses
- Hybrid Search (BM25 + Dense Retrieval)
- Reranking using Cross Encoder
- Metadata filtering
- OCR support for scanned PDFs
- Image understanding
- Table extraction
- Multi-user chat sessions

---

# 🌐 FastAPI Extension

The architecture follows strict OOP principles, making it easy to expose as a REST API.

Suggested endpoints

```
POST /upload-pdf
```

Uploads and indexes PDF documents.

```
POST /chat
```

Accepts

```json
{
  "session_id": "123",
  "question": "Who chaired the Drafting Committee?"
}
```

Returns

```json
{
  "answer": "...",
  "confidence": 92,
  "sources": [
    {
      "document": "constitution.pdf",
      "page": 2,
      "paragraph": 3
    }
  ]
}
```

---

# 📌 Key Highlights

- Multi-document semantic search
- Local LLM inference
- Voice-enabled RAG
- Hallucination-resistant responses
- Source-grounded answers
- Production-ready modular architecture
- Pinecone cloud vector database
- Conversation-aware retrieval
- GPU accelerated speech transcription

---

## ⭐ If you found this project useful, consider giving it a star!
-Anshul Verma
