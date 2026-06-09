import logging
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class Retriever:
    """
    Retrieves document chunks matching a text query, evaluating 
    relevance against a cosine similarity threshold.
    """
    def __init__(
        self, 
        vector_store: VectorStoreManager, 
        embeddings: EmbeddingManager, 
        top_k: int = 4, 
        relevance_threshold: float = 0.40
    ):
        """
        Initializes the Retriever.
        
        Args:
            vector_store (VectorStoreManager): The vector database manager.
            embeddings (EmbeddingManager): The embedding model manager.
            top_k (int): Number of top documents to fetch from FAISS.
            relevance_threshold (float): Minimum cosine similarity to consider relevant.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(self, query: str) -> tuple[list[dict], float, bool]:
        """
        Retrieves matching chunks, evaluates their relevance score, and filters
        out those below the similarity threshold.
        
        Args:
            query (str): The search query.
            
        Returns:
            tuple[list[dict], float, bool]:
                - list[dict]: Chunks that exceeded the threshold.
                - float: The highest similarity score found.
                - bool: Whether at least one chunk exceeded the threshold.
        """
        if self.vector_store.is_empty():
            logger.warning("Attempted retrieval on an empty vector store.")
            return [], 0.0, False

        logger.info(f"Retrieving matching chunks for query: '{query}'")
        
        # 1. Convert question to embedding
        query_embedding = self.embeddings.get_embedding(query)
        
        # 2. Retrieve top-k most relevant chunks from FAISS
        search_results = self.vector_store.search(query_embedding, k=self.top_k)
        
        if not search_results:
            logger.info("No matching chunks found in FAISS.")
            return [], 0.0, False
            
        # 3. Evaluate retrieval relevance score (based on the highest similarity)
        top_chunk, top_score = search_results[0]
        logger.info(f"Top retrieved chunk score: {top_score:.4f} (Threshold: {self.relevance_threshold})")
        
        # Filter chunks that are above or equal to the relevance threshold
        relevant_chunks = [chunk for chunk, score in search_results if score >= self.relevance_threshold]
        
        is_relevant = len(relevant_chunks) > 0
        if not is_relevant:
            logger.info(
                f"Top score {top_score:.4f} is below the threshold of {self.relevance_threshold}. "
                "Query marked as unrelated."
            )
            return [], top_score, False
            
        logger.info(f"Retrieved {len(relevant_chunks)} chunks above the relevance threshold.")
        return relevant_chunks, top_score, True
