# Local Multi-Document Retrieval-Augmented Generation (RAG) Chatbot Documentation

This document provides a comprehensive overview of the Local Multi-Document RAG Chatbot project. It explains the system architecture, data flow, setup instructions, hardware optimization details, pros, cons, and real-world applications.

---

## 1. Introduction to RAG
**Retrieval-Augmented Generation (RAG)** is an architectural pattern used to optimize the output of a Large Language Model (LLM) by referencing an authoritative, external knowledge base (in this case, your uploaded PDF documents) before generating a response. 

Instead of relying on the LLM's pre-trained general knowledge—which can be outdated or prone to hallucinations—RAG retrieves relevant segments of the user's documents and feeds them directly to the LLM as the sole source of truth.

---

## 2. System Architecture & Component Breakdown

The project is designed as a modular, object-oriented system in Python. It consists of the following core components:

```
                   ┌────────────────────────────────────────┐
                   │         Raw PDF Documents (Multiple)   │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ 1. PDF Loader (PyMuPDF)                │
                   └───────────────────┬────────────────────┘
                                       │ (Page-wise text)
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ 2. Chunker (Paragraph + Char Window)   │
                   └───────────────────┬────────────────────┘
                                       │ (Doc, Page, Paragraph Chunks)
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ 3. Embedding Manager (all-MiniLM-L6-v2)│ ◄── [CUDA/GPU Accelerated]
                   └───────────────────┬────────────────────┘
                                       │ (Unit-Normalized Vectors)
                                       ▼
                   ┌────────────────────────────────────────┐
                   │ 4. Vector Store Manager (Pinecone)     │ ──► Upserts: Cloud Database (AWS/GCP)
                   └───────────────────┬────────────────────┘
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           │                  Query/Chat Pipeline                  │
           ▼                                                       ▼
  ┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
  │ 5. Chat Memory  ├───────►│  Query Rewriter  ├───────►│ 6. Retriever     │
  │ (Rolling Turns) │        │ (Llama 3.2 Stand-│        │ (Cosine Simi-    │
  └─────────────────┘        │  alone Question) │        │  larity Check)   │
                             └──────────────────┘        └────────┬─────────┘
                                                                  │
                                                     ┌────────────┴────────────┐
                                                     │ Passes Threshold?       │
                                                     └─────┬──────────────┬────┘
                                                       Yes │           No │
                                                           ▼              ▼
                                                 ┌───────────┐  ┌───────────────────┐
                                                 │ 7. LLM    │  │ Refuse & Return:  │
                                                 │ Grounding │  │ "This question    │
                                                 │ (Context) │  │  does not appear  │
                                                 └─────┬─────┘  │  to be related..."│
                                                       │        └───────────────────┘
                                                       ▼
                                             ┌───────────────────────────┐
                                             │ Grounded Response         │
                                             │ + Citations (Doc, Page,   │
                                             │   Paragraph, Confidence%) │
                                             └───────────────────────────┘
```

### 1. Ingestion Pipeline
* **PDF Loader (`src/pdf_loader.py`)**: Uses PyMuPDF (`fitz`) to read one or more documents. It parses text page-by-page and stores metadata associated with the text.
* **Document Chunker (`src/chunker.py`)**: Split text page-wise, separates paragraphs logically (via double newlines `\n\n`), assigns paragraph numbers, and chunks them into character-sized windows (default size 500 characters, 50 characters overlap) while maintaining comprehensive metadata (`document_name`, `page_number`, `paragraph_number`, `chunk_id`).
* **Embedding Manager (`src/embeddings.py`)**: Converts each text chunk into a 384-dimensional mathematical vector using Sentence Transformers (`all-MiniLM-L6-v2`).
  * **GPU/CUDA Acceleration**: Automatically runs on NVIDIA GPUs using PyTorch CUDA if available, accelerating the indexing process.
  * **Online/Offline Fallback**: If Hugging Face is offline or there is a connection reset, it automatically loads the model locally from the cache directory.
  * **Normalization**: Vectors are unit-normalized to allow Pinecone to calculate exact cosine similarity.
* **Vector Store Manager (`src/vector_store.py`)**: Integrates with the cloud-hosted **Pinecone** vector database (using the `cosine` similarity metric).
  * When documents are processed, chunks are upserted along with metadata (`document_name`, `page_number`, `paragraph_number`, `chunk_text`, `chunk_id`) directly to the Pinecone index.
  * Checks for index existence and automatically creates a serverless index if one is missing, polling until it becomes ready.
  * Implements dynamic zero-vector metadata querying to find out if there are existing documents and identify the document name, allowing the CLI's `"reuse existing vector index"` flow to continue working transparently.

### 2. Retrieval & Generation Pipeline
* **Conversational Memory (`src/memory.py`)**: Maintains a rolling history of the last 5 turns of conversation in the current session.
* **RAG Chatbot Coordinator (`src/chatbot.py`)**:
  * **Query Reformulation**: When you ask a follow-up question (e.g., *"What was his role?"*), the chatbot sends the chat history and the question to the LLM first. The LLM reformulates it into a standalone question (e.g., *"What was the role of Dr. B. R. Ambedkar?"*).
  * **Retrieval & Relevance Check (`src/retriever.py`)**: The standalone question is embedded and queried against the Pinecone index to retrieve the top-4 closest chunks. 
    * If the highest similarity score is below the threshold (default `0.40`), it blocks generation and returns: *"This question does not appear to be related to the uploaded documents. Please ask questions based on the uploaded PDFs."*
  * **Strict Grounding Prompt**: If chunks pass the threshold, they are formatted into a prompt which instructs the model to answer **only** using the context and output: *"I could not find this information in the uploaded documents."* if the answer isn't present.
  * **Ollama Integration**: Talks to a local Ollama instance (running Llama 3.2 on the GPU) with temperature set to `0.0` for deterministic, factual outputs.
  * **Citations & Confidence**: Extracts and formats unique source citations (Document name, Page number, and Paragraph number) and displays a confidence score (0-100%) calculated directly from the Pinecone similarity score.

---

## 3. Step-by-Step Guide to Run the Project

### Prerequisites
1. **Python 3.11+** installed.
2. An **NVIDIA GPU** with up-to-date graphics drivers.
3. [Ollama](https://ollama.com/) installed and running.

### 1. Download Llama 3.2 (3B)
Llama 3.2 (3B) is the optimal model size (~2.0 GB) for 4GB VRAM GPUs (like the RTX 3050 Laptop). Run the following command in your terminal to pull it:
```bash
& "C:\Users\ACER\AppData\Local\Programs\Ollama\ollama.exe" pull llama3.2
```

### 2. Configure Environment & Start the App

1. Create a `.env` file in the project directory (`D:\RAG Chatbot`) with your Pinecone credentials:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

2. Navigate to `D:\RAG Chatbot` in your PowerShell terminal and run:
```powershell
$env:OLLAMA_MODEL="llama3.2"
python main.py
```

### 3. Usage
1. Enter the comma-separated paths to your PDF files (e.g., `doc1.pdf, doc2.pdf`) when prompted.
2. Ask questions about the documents' contents.
3. Type `clear` to reset the chat memory or `exit` to close the program.

---

## 4. Pros & Cons of This Architecture

### Pros
* **Local LLM Processing & Privacy**: The Large Language Model (Llama 3.2 via Ollama) runs entirely on your local machine, keeping raw conversation history private. Vector embeddings are stored securely in Pinecone cloud.
* **Multi-Document Support**: Seamlessly queries across multiple PDF files simultaneously and synthesizes answers drawing context from multiple sources.
* **GPU-Accelerated Speed**: By using a 3B model (Llama 3.2) that fits entirely in the RTX 3050's 4GB VRAM, token generation runs at maximum speed (taking only seconds).
* **Strict Factuality (No Hallucinations)**: The custom retriever blocks unrelated queries from reaching the LLM, and the prompt forbids the LLM from using its pre-trained external knowledge.
* **Traceability**: Every answer includes document, page-level, and paragraph-level citations showing exactly where the source text was found, along with a retrieval confidence score.
* **Conversational Context**: It handles pronouns and context-sensitive follow-ups natively via its LLM-based query reformulation.

### Cons
* **Hardware Bound**: Performance is heavily dependent on your computer's hardware. Running models larger than Llama 3.2 (like Llama 3 8B or 70B) requires expensive graphics cards (8GB-40GB VRAM).
* **CLI Limitations**: The current user interface is terminal-based, which isn't as user-friendly as a web GUI.
* **Session Bound Memory**: The conversational history is kept in memory and reset once the application is closed.
* **Dependency on Out-Of-Domain Thresholds**: Cosine similarity thresholds (e.g., `0.40`) are heuristic. If a document uses highly unique vocabulary, some relevant questions might get blocked, or irrelevant ones might sneak in.

---

## 5. Applications

This local RAG chatbot architecture is ideal for:
1. **Confidential Business Documents**: Querying internal financials, product specifications, or strategic plans without violating data protection standards.
2. **Academic & Research Papers**: Chatting with research PDFs to extract definitions, page references, and summaries without manual page-skimming.
3. **Legal Document Review**: Querying contracts, lease agreements, or court filings to find specific terms and clauses.
4. **Offline Operations**: Running internal knowledge bases in remote locations, on flights, or during internet outages.
