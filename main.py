import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.speech.speech_to_text import SpeechToTextManager


from src.pdf_loader import PDFLoader
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.memory import ChatMemory
from src.chatbot import RAGChatbot

# Configure professional dual logging
# - Detailed debug/info logs go to logs/chatbot.log
# - Only warnings/errors display on stdout to keep the terminal CLI pristine
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File logger
file_handler = logging.FileHandler("logs/chatbot.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console logger (errors only)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def main():
    print("=" * 65)
    print("    Local Document RAG Chatbot (Ollama + Pinecone + MiniLM)   ")
    print("=" * 65)
    
    # 1. Initialize Embedding Manager
    try:
        print("Loading embeddings model (all-MiniLM-L6-v2)...")
        embeddings_mgr = EmbeddingManager()
        
        # Initialize Vector Store Manager
        vector_store_mgr = VectorStoreManager(
            vectorstore_dir="vectorstore", 
            dimension=embeddings_mgr.dimension
        )
        
        # Initialize Speech-to-Text Manager placeholder (lazy-loaded on demand)
        stt_mgr = None
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize core model: {e}")
        logging.exception("Core model loading failure")
        sys.exit(1)

    # 2. Check for existing indexed documents
    index_exists = False
    existing_docs = []
    try:
        vector_store_mgr.create_or_load_index()
        if not vector_store_mgr.is_empty():
            index_exists = True
            existing_docs = sorted(list(set(
                chunk.get("document_name") or chunk.get("source_document")
                for chunk in vector_store_mgr.chunks_metadata
                if chunk.get("document_name") or chunk.get("source_document")
            )))
    except Exception as e:
        logging.exception("Error checking for existing index database")

    pdf_paths = []

    if index_exists and existing_docs:
        print("\nFound existing vector index containing the following documents:")
        for doc in existing_docs:
            print(f"- {doc}")
        choice = input("\nWould you like to search these existing documents? (y/n, default 'y'): ").strip().lower()
        if choice not in ['n', 'no']:
            print(f"Loading existing vector store. Ready to query {len(existing_docs)} documents.")
            pdf_paths = existing_docs
        else:
            index_exists = False

    if not index_exists:
        while True:
            paths_input = input("\nEnter the path(s) to the PDF document(s), separated by commas: ").strip()
            if not paths_input:
                print("Paths cannot be empty. Please enter a valid path.")
                continue
            
            # Split paths by comma
            raw_paths = [p.strip() for p in paths_input.split(",")]
            valid_paths = []
            all_valid = True
            
            for path in raw_paths:
                if not path:
                    continue
                if not os.path.exists(path):
                    print(f"File not found at '{path}'. Please check the path and try again.")
                    all_valid = False
                    break
                if not path.lower().endswith(".pdf"):
                    print(f"Target file '{path}' must be a PDF document (.pdf extension).")
                    all_valid = False
                    break
                valid_paths.append(path)
                
            if not all_valid or not valid_paths:
                continue
                
            pdf_paths = valid_paths
            break

        # Process and ingest each document
        all_chunks = []
        print(f"\n[1/3] Reading {len(pdf_paths)} PDF(s)...")
        try:
            for path in pdf_paths:
                filename = os.path.basename(path)
                print(f"  - Reading: {filename}")
                loader = PDFLoader(path)
                pages_data = loader.load()
                
                print(f"  - Chunking text content for: {filename}")
                chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
                chunks = chunker.split_pages(pages_data)
                
                if chunks:
                    all_chunks.extend(chunks)
                    
            if not all_chunks:
                print("\n[ERROR] No extractable text found in any of the PDFs. Are they scanned or image-only?")
                sys.exit(1)

            print(f"\n[2/3] Generating embeddings for {len(all_chunks)} chunks (this may take a moment)...")
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = embeddings_mgr.get_embeddings(texts)
            
            print(f"\n[3/3] Upserting to Pinecone...")
            # Reset/Save new index
            vector_store_mgr._initialize_empty_index()
            vector_store_mgr.add_documents(all_chunks, embeddings)
            vector_store_mgr.save()
            print("Vector store created and stored successfully!")
            
        except Exception as e:
            print(f"\n[ERROR] PDF parsing and indexing failed: {e}")
            logging.exception("PDF Ingestion pipeline crashed")
            sys.exit(1)

    # 3. Setup Retriever, Memory, and Chatbot Components
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")
    
    # We use a default cosine threshold of 0.40, which works well with MiniLM-L6-v2 normalized embeddings
    retriever = Retriever(
        vector_store=vector_store_mgr, 
        embeddings=embeddings_mgr, 
        top_k=4, 
        relevance_threshold=0.40
    )
    
    memory = ChatMemory(max_history_turns=5)
    
    chatbot = RAGChatbot(
        retriever=retriever, 
        memory=memory, 
        ollama_url=ollama_host, 
        model_name=ollama_model
    )
    
    print("\n" + "-" * 50)
    print("PDFs Loaded:")
    for path in pdf_paths:
        print(f" - {os.path.basename(path)}")
    print("-" * 50)
    print("Type 'exit' to quit.")
    print("Type 'clear' to reset chat memory.")
    print("-" * 50 + "\n")

    # 4. Interactive Chat Loop
    while True:
        print("\nChoose input mode:")
        print("1. Ask by typing")
        print("2. Ask by voice")
        print("Type 'exit' to quit or 'clear' to reset chat memory.")
        
        try:
            choice = input("Selection [1 or 2]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not choice:
            continue
            
        if choice.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if choice.lower() == "clear":
            memory.clear()
            print("Chat memory cleared.")
            print("-" * 40 + "\n")
            continue
            
        question = ""
        if choice == "1":
            try:
                question = input("\nAsk Question (Type):\n").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
            if not question:
                continue
        elif choice == "2":
            if stt_mgr is None:
                try:
                    whisper_model = os.getenv("WHISPER_MODEL", "tiny")
                    print(f"Loading Faster-Whisper transcription model ({whisper_model})...")
                    stt_mgr = SpeechToTextManager(model_size=whisper_model)
                except Exception as stt_init_err:
                    print(f"\n[SPEECH ERROR] Failed to load speech transcription model: {stt_init_err}")
                    print("You can still use typed input mode (Option 1).")
                    continue

            print("\nPress [Enter] to start recording...")
            try:
                input()
                temp_filename = "temp_recording.wav"
                stt_mgr.record_audio(temp_filename)
                question = stt_mgr.transcribe_audio(temp_filename)
                print(f"\nRecognized Text: \"{question}\"")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
            except Exception as stt_err:
                print(f"\n[SPEECH ERROR] {stt_err}")
                continue
            if not question:
                print("No recognized text. Please try again.")
                continue
        else:
            print("Invalid selection. Please choose 1 or 2.")
            continue
            
        print("\nThinking...")
        try:
            answer, sources, confidence = chatbot.ask(question)
            
            print("\nAnswer:")
            print(answer)
            
            is_fallback = answer in [
                "This question does not appear to be related to the uploaded documents. Please ask questions based on the uploaded PDFs.",
                "I could not find this information in the uploaded documents."
            ]
            
            if not is_fallback:
                if confidence is not None:
                    print("\nConfidence:")
                    print(f"{confidence}%")
                
                if sources:
                    print("\nSources:\n")
                    for src in sources:
                        print(f"* Document: {src['document_name']}")
                        print(f"  Page: {src['page_number']}")
                        print(f"  Paragraph: {src['paragraph_number']}")
                        print()
            print("\n" + "-" * 50 + "\n")
            
        except ConnectionError as ce:
            print(f"\n[OLLAMA ERROR] {ce}")
            print("Ensure Ollama is running and the model is pulled.")
            print("\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"\n[SYSTEM ERROR] {e}")
            logging.exception("Error processing question in loop")
            print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
