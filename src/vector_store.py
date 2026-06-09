import os
import logging
import time
import numpy as np
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages cloud-hosted Pinecone index creation, initialization, 
    upserting, and similarity search operations.
    """
    def __init__(self, vectorstore_dir: str = "vectorstore", dimension: int = 384):
        """
        Initializes the VectorStoreManager.
        
        Args:
            vectorstore_dir (str): Kept for signature compatibility; not used for local index files.
            dimension (int): Dimension of the embedding vectors (default 384 for all-MiniLM-L6-v2).
        """
        self.vectorstore_dir = os.path.abspath(vectorstore_dir)
        self.dimension = dimension
        
        # Load credentials from environment
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        
        if not self.api_key or not self.index_name:
            raise ValueError(
                "Missing Pinecone configuration. Ensure PINECONE_API_KEY and "
                "PINECONE_INDEX_NAME are set in your environment or .env file."
            )
            
        self.pc = None
        self.index = None
        self.chunks_metadata = []

    def create_or_load_index(self) -> None:
        """
        Connects to the Pinecone index. Creates it if it doesn't exist.
        Retrieves existing source document metadata if vectors are present.
        """
        logger.info("Connecting to Pinecone client...")
        try:
            self.pc = Pinecone(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise RuntimeError(f"Pinecone initialization error: {e}")

        # Check if index exists, create if not
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Index '{self.index_name}' not found. Creating serverless index...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region)
                )
                # Wait for index to be ready
                logger.info(f"Waiting for index '{self.index_name}' to become ready...")
                while True:
                    desc = self.pc.describe_index(self.index_name)
                    status = desc.status
                    if isinstance(status, dict):
                        is_ready = status.get("ready", False)
                    else:
                        is_ready = getattr(status, "ready", False)
                    if is_ready:
                        break
                    time.sleep(5)
                logger.info("Index created and ready.")
            else:
                logger.info(f"Connected to existing Pinecone index: '{self.index_name}'")
        except Exception as e:
            logger.error(f"Error checking/creating Pinecone index: {e}")
            raise RuntimeError(f"Index check/creation failure: {e}")

        self.index = self.pc.Index(self.index_name)

        # Check if index is empty or has existing vectors
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            logger.info(f"Index '{self.index_name}' has {total_vectors} total vectors.")
            
            if total_vectors > 0:
                # Query index with a dummy vector to retrieve unique document names
                dummy_vector = [0.0] * self.dimension
                results = self.index.query(vector=dummy_vector, top_k=500, include_metadata=True)
                
                unique_docs = set()
                sample_chunks = []
                for match in results.matches:
                    meta = match.metadata
                    doc_name = meta.get("document_name") or meta.get("source_document")
                    if doc_name and doc_name not in unique_docs:
                        unique_docs.add(doc_name)
                        sample_chunks.append({
                            "document_name": doc_name,
                            "source_document": doc_name,
                            "page_number": int(meta.get("page_number", 1)),
                            "paragraph_number": int(meta.get("paragraph_number", 1)),
                            "chunk_id": match.id,
                            "text": meta.get("text") or meta.get("chunk_text") or ""
                        })
                
                self.chunks_metadata = sample_chunks
                logger.info(f"Found existing indexed documents: {list(unique_docs)}")
            else:
                self.chunks_metadata = []
        except Exception as e:
            logger.warning(f"Failed to query index stats/sample metadata: {e}. Starting with empty metadata.")
            self.chunks_metadata = []

    def _initialize_empty_index(self) -> None:
        """Wipes the Pinecone index / deletes all vectors."""
        if self.index is None:
            self.create_or_load_index()
            
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
        except Exception as e:
            logger.warning(f"Could not retrieve index stats before deletion: {e}")
            total_vectors = -1
            
        if total_vectors == 0:
            logger.info("Pinecone index is already empty. Skipping delete operation.")
            self.chunks_metadata = []
            return

        logger.info(f"Wiping all vectors in index '{self.index_name}' to start fresh...")
        try:
            # We delete all vectors in the default namespace
            self.index.delete(delete_all=True, namespace="")
            self.chunks_metadata = []
            logger.info("Successfully wiped Pinecone index.")
        except Exception as e:
            err_msg = str(e).lower()
            if "not found" in err_msg or "404" in err_msg:
                logger.info("Pinecone index namespace is empty. Wiping skipped.")
                self.chunks_metadata = []
            else:
                logger.error(f"Failed to wipe Pinecone index: {e}")
                raise RuntimeError(f"Index wipe failure: {e}")

    def add_documents(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Adds chunks and their embeddings to the Pinecone index.
        
        Args:
            chunks (list[dict]): Chunks list matching embeddings rows.
            embeddings (np.ndarray): 2D numpy array of shape (len(chunks), dimension).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: Received {len(chunks)} document chunks and {len(embeddings)} embeddings."
            )
            
        if self.index is None:
            self.create_or_load_index()

        logger.info(f"Upserting {len(chunks)} vectors to Pinecone index '{self.index_name}'...")
        
        # Prepare data for upsert
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            # Support both old and new field names for maximum flexibility
            doc_name = chunk.get("source_document") or chunk.get("document_name", "Unknown")
            metadata = {
                "document_name": doc_name,
                "source_document": doc_name,
                "page_number": int(chunk.get("page_number", 1)),
                "paragraph_number": int(chunk.get("paragraph_number", 1)),
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "text": chunk.get("text", ""),
                "chunk_text": chunk.get("text", "")
            }
            vectors_to_upsert.append({
                "id": chunk.get("chunk_id", f"chunk_{i}"),
                "values": embeddings[i].tolist(),
                "metadata": metadata
            })

        # Upsert in batches of 100
        batch_size = 100
        for idx in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[idx : idx + batch_size]
            try:
                logger.info(f"Upserting batch {idx//batch_size + 1} ({len(batch)} vectors)...")
                self.index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
                raise RuntimeError(f"Pinecone upsert failure: {e}")
                
        # Populate chunks_metadata with all the unique documents we now have in the index
        unique_docs = set(meta["metadata"]["document_name"] for meta in vectors_to_upsert)
        self.chunks_metadata = [{"document_name": doc, "source_document": doc} for doc in unique_docs]
            
        logger.info("Successfully completed Pinecone upsert.")

    def save(self) -> None:
        """
        No-op method kept for compatibility. Pinecone operations are written in real-time.
        """
        logger.info("Pinecone operates in the cloud in real-time. Save operations are persisted dynamically.")

    def search(self, query_embedding: np.ndarray, k: int = 4) -> list[tuple[dict, float]]:
        """
        Searches the Pinecone index for the top k most similar chunks.
        
        Args:
            query_embedding (np.ndarray): Query embedding array.
            k (int): Number of nearest neighbors to retrieve.
            
        Returns:
            list[tuple[dict, float]]: A list of tuples containing:
                (chunk_metadata_dict, similarity_score_float).
        """
        if self.index is None:
            logger.warning("Search query invoked but index is not initialized.")
            return []

        # Convert query embedding to list
        if isinstance(query_embedding, np.ndarray):
            query_list = query_embedding.tolist()
        else:
            query_list = list(query_embedding)

        try:
            # Execute query
            response = self.index.query(
                vector=query_list,
                top_k=k,
                include_metadata=True
            )
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            raise RuntimeError(f"Pinecone search error: {e}")

        results = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            score = float(match.get("score", 0.0))
            
            # Reconstruct chunk dict structure with both new and old keys
            doc_name = meta.get("document_name") or meta.get("source_document") or "Unknown"
            chunk_dict = {
                "text": meta.get("text") or meta.get("chunk_text") or "",
                "chunk_text": meta.get("chunk_text") or meta.get("text") or "",
                "page_number": int(meta.get("page_number", 1)),
                "paragraph_number": int(meta.get("paragraph_number", 1)),
                "chunk_id": match.get("id") or meta.get("chunk_id", ""),
                "document_name": doc_name,
                "source_document": doc_name
            }
            results.append((chunk_dict, score))
            
        return results

    def is_empty(self) -> bool:
        """Checks if the index contains any vectors."""
        if self.index is None:
            return True
        try:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0) == 0
        except Exception as e:
            logger.error(f"Failed to check if index is empty: {e}")
            return True
