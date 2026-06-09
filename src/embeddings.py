import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages the loading of the Sentence Transformer model and generating 
    normalized embeddings for text chunks and queries.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the EmbeddingManager and loads the SentenceTransformer model.
        
        Args:
            model_name (str): The name of the sentence-transformers model to use.
        """
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading SentenceTransformer on device: {device}")
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.warning(
                f"Failed to load SentenceTransformer online ({e}). "
                "Attempting to load from local cache with local_files_only=True..."
            )
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = SentenceTransformer(model_name, device=device, model_kwargs={"local_files_only": True})
            except Exception as cache_err:
                logger.error(f"Failed to load model offline: {cache_err}")
                raise RuntimeError(
                    f"Model initialization failure: Could not load '{model_name}' online or from cache. "
                    f"Original error: {e}. Cache error: {cache_err}"
                )
                
        try:
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = self.model.get_embedding_dimension()
            else:
                self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer loaded. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to verify model dimension: {e}")
            raise RuntimeError(f"Model configuration check failed: {e}")

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generates unit-normalized embeddings for a list of strings.
        
        Args:
            texts (list[str]): List of texts to generate embeddings for.
            
        Returns:
            np.ndarray: A 2D numpy array of shape (len(texts), dimension) of float32 embeddings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            # We set normalize_embeddings=True so that the generated embeddings 
            # have a unit L2 norm. This allows us to use Inner Product (IP) 
            # in FAISS to get exact cosine similarity.
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation error: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a unit-normalized embedding for a single text query.
        
        Args:
            text (str): Query string.
            
        Returns:
            np.ndarray: 1D numpy array representing the query embedding.
        """
        embeddings = self.get_embeddings([text])
        if len(embeddings) > 0:
            return embeddings[0]
        raise RuntimeError("Failed to generate embedding for the single text query.")
