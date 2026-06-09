# Local Multi-Document RAG Chatbot (Ollama + Pinecone + MiniLM)

A production-quality, terminal-based Retrieval-Augmented Generation (RAG) chatbot in Python. This chatbot allows users to query multiple uploaded PDF documents simultaneously and receive answers that are **strictly grounded** in the document context. It implements absolute grounding rules (refusing to answer when information is missing), multi-level output citations (Document name, Page number, and Paragraph number), confidence scores, conversational session memory, and Pinecone vector database integration.

---

## Technical Architecture

- **PDF Text Extraction**: Uses **PyMuPDF** (`fitz`) to extract page-wise text from multi-page documents.
- **Paragraph & Text Chunking**: Splits extracted text page-wise, separates paragraphs logically, assigns paragraph numbers, and chunks them into character-sized windows (default size 500 characters, 50 characters overlap) while maintaining comprehensive metadata (`document_name`, `page_number`, `paragraph_number`, `chunk_id`).
- **Embeddings**: Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) to generate 384-dimensional unit-normalized embeddings.
- **Vector Database**: Implements **Pinecone** (cloud vector database using the `cosine` similarity metric) to index and search vector embeddings in real time.
- **Conversational Memory**: Implements a session message history queue. If history is present, the chatbot uses the local LLM to reformulate the question (resolving pronouns) into a standalone search query.
- **Relevance Score Check**: Queries Pinecone and discards matches below a similarity threshold (default `0.40`). If all retrieved chunks are below this threshold, the chatbot refuses to answer to prevent hallucination.
- **LLM Engine**: Queries **Ollama** running `llama3.2` locally using a strict zero-temperature grounding prompt.

---

## Setup & Installation

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally.
- A **Pinecone** account, API Key, and an Index created (or let the app create it).

### 2. Configure Environment Variables

Create a `.env` file in the project root directory with the following variables:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Optional configuration defaults
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 2. Download Llama 3 in Ollama

Open your terminal and pull the `llama3` model:

```bash
ollama pull llama3
```

Ensure Ollama is running in the background. By default, it runs on `http://localhost:11434`.

### 3. Install Python Dependencies

Clone or copy the project files to your workspace, navigate to the directory, and run:

```bash
pip install -r requirements.txt
```

---

## Usage Instructions

To run the chatbot, simply run `main.py`:

```bash
python main.py
```

### Flow description:

1. **Model Loading**: The application will load the Sentence Transformers embedding model on startup.
2. **PDF Load/Ingestion**: 
   - If existing vectors are found in your Pinecone index, the CLI will ask if you want to query it.
   - If the index is empty (or you choose not to query the existing index), you will be prompted to enter the path to a PDF file (e.g. `data/constitution.pdf`) to clear the index and ingest new vectors.
3. **Chat Session**: 
   - Ask questions based on the PDF contents.
   - Outputs display the grounded answer and page-level source citations.
   - Type `clear` to reset the chat memory.
   - Type `exit` to quit.

---

## Code Base Organization

```
project/
│
├── data/                       # Directory to store raw PDFs
├── logs/                       # Application run-time logs (e.g., logs/chatbot.log)
│
├── src/
│   ├── pdf_loader.py           # Loads PDFs and parses text page-by-page
│   ├── chunker.py              # Splits text into chunks with metadata
│   ├── embeddings.py           # Generates Sentence Transformers embeddings
│   ├── vector_store.py         # Pinecone vector store manager
│   ├── retriever.py            # Retrieves and filters chunks based on relevance
│   ├── memory.py               # Maintains session conversational memory
│   └── chatbot.py              # Integrates memory, retriever, and local LLM
│
├── .env.example                # Example configuration template
├── main.py                     # CLI loop and setup entrypoint
├── requirements.txt            # Package dependencies
└── README.md                   # Setup and usage guide
```

---

## Example Chat Session

```
=================================================================
    Local Document RAG Chatbot (Ollama + Pinecone + MiniLM)   
=================================================================
Loading embeddings model (all-MiniLM-L6-v2)...

Enter the path(s) to the PDF document(s), separated by commas: data/constitution.pdf, data/constitutional_history.pdf

[1/3] Reading 2 PDF(s)...
  - Reading: constitution.pdf
  - Chunking text content for: constitution.pdf
  - Reading: constitutional_history.pdf
  - Chunking text content for: constitutional_history.pdf

[2/3] Generating embeddings for 215 chunks (this may take a moment)...

[3/3] Upserting to Pinecone...
Vector store created and stored successfully!

--------------------------------------------------
PDFs Loaded:
 - constitution.pdf
 - constitutional_history.pdf
--------------------------------------------------
Type 'exit' to quit.
Type 'clear' to reset chat memory.
--------------------------------------------------

Ask Question:
Who chaired the Drafting Committee?

Thinking...

Answer:
Dr. B. R. Ambedkar chaired the Drafting Committee.

Source:
Document: constitution.pdf
Page: 2
Paragraph: 3

Confidence: 92%

--------------------------------------------------

Ask Question:
Compare the Drafting Committee and the Constituent Assembly.

Thinking...

Answer:
The Constituent Assembly was responsible for drafting the Constitution, while the Drafting Committee prepared the final draft under the chairmanship of Dr. B. R. Ambedkar.

Sources:

Document: constitution.pdf
Page: 2
Paragraph: 3

Document: constitutional_history.pdf
Page: 7
Paragraph: 2

Confidence: 89%

--------------------------------------------------

Ask Question:
Who is Shah Rukh Khan?

Thinking...

Answer:
This question does not appear to be related to the uploaded documents. Please ask questions based on the uploaded PDFs.

Confidence: 15%

--------------------------------------------------

Ask Question:
exit
Goodbye!
```

---

## Extension to FastAPI Backend

The codebase has been designed with strict OOP principles, separating ingestion, vector storage, retrieval, memory, and LLM query tasks. 

To turn this into a REST API:
1. Create a `FastAPI` app (e.g., using `uvicorn`).
2. Implement a `/upload-pdf` endpoint that accepts a PDF file upload, calls `PDFLoader` and `DocumentChunker`, generates embeddings, and saves the index.
3. Implement a `/chat` endpoint that accepts a JSON payload `{ "session_id": str, "question": str }`, retrieves or initializes a `ChatMemory` instance for that `session_id`, and calls `RAGChatbot.ask(question)` to return the answer and citations as a JSON response.
