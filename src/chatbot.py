import os
import requests
import logging
from src.retriever import Retriever
from src.memory import ChatMemory
from src.prompts.system_prompt import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RAGChatbot:
    """
    Coordinates the RAG chatbot pipeline. Integrates the retriever, 
    conversational memory, query reformulation, and Ollama LLM execution.
    """
    def __init__(
        self, 
        retriever: Retriever, 
        memory: ChatMemory, 
        ollama_url: str = "http://localhost:11434", 
        model_name: str = "llama3"
    ):
        """
        Initializes the RAGChatbot.
        
        Args:
            retriever (Retriever): The document retriever.
            memory (ChatMemory): The conversational memory.
            ollama_url (str): The local Ollama server address.
            model_name (str): The name of the LLM model inside Ollama.
        """
        self.retriever = retriever
        self.memory = memory
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name

    def _call_ollama(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Sends a prompt to the local Ollama API.
        
        Args:
            prompt (str): Text prompt to feed the LLM.
            temperature (float): Sampling temperature. Defaults to 0.0 for factual responses.
            
        Returns:
            str: Generated text from the LLM.
        """
        endpoint = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            logger.debug(f"Requesting Ollama '{self.model_name}' at {endpoint}...")
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("response", "").strip()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama connection error: {e}")
            raise ConnectionError(
                f"Failed to connect to local Ollama server at {self.ollama_url}. "
                "Please ensure Ollama is running (`ollama serve`)."
            )
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            raise RuntimeError(f"Ollama integration error: {e}")

    def reformulate_query(self, question: str) -> str:
        """
        Reformulates a follow-up query into a standalone question using Ollama
        if conversational history is present.
        
        Args:
            question (str): The user's latest follow-up question.
            
        Returns:
            str: The standalone reformulated question.
        """
        if self.memory.is_empty():
            return question

        history_str = self.memory.get_formatted_history()
        
        prompt = f"""Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question (i.e., self-contained, resolving all pronoun references to their original nouns) in the context of the conversation. Do not add any conversational filler, notes, or explanations; return ONLY the rewritten question.

Conversation History:
{history_str}

Follow-up Question:
{question}

Standalone Question:"""

        logger.info("Attempting query reformulation based on chat memory...")
        try:
            standalone_query = self._call_ollama(prompt, temperature=0.0)
            if standalone_query:
                logger.info(f"Query reformulated: '{question}' -> '{standalone_query}'")
                return standalone_query
        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}. Falling back to original question.")
            
        return question

    def ask(self, question: str) -> tuple[str, list[dict], int]:
        """
        Processes a user question through the complete RAG workflow:
        1. Reformulate question if chat history exists.
        2. Retrieve top-k chunks from Pinecone.
        3. Evaluate relevance threshold.
        4. If below threshold, return fallback message.
        5. Build strict prompt template and request LLM.
        6. Append source citations.
        7. Record the Q&A turn in memory.
        
        Args:
            question (str): The user's query.
            
        Returns:
            tuple[str, list[dict], int]: 
                - str: The generated answer or fallback message.
                - list[dict]: List of unique source dictionaries containing document_name, page_number, paragraph_number.
                - int: Confidence score (0-100).
        """
        # 1. Convert/reformulate query if memory contains context
        search_query = self.reformulate_query(question)
        
        # 2. Retrieve chunks & evaluate relevance score
        relevant_chunks, max_score, is_relevant = self.retriever.retrieve(search_query)
        
        # Compute confidence score
        confidence = max(0, min(100, int(max_score * 100)))
        
        # 3. If relevance is below threshold, return the standard fallback message
        if not is_relevant:
            fallback_msg = "This question does not appear to be related to the uploaded documents. Please ask questions based on the uploaded PDFs."
            return fallback_msg, [], confidence

        # 4. Format context from retrieved chunks
        context_blocks = []
        for chunk in relevant_chunks:
            doc_name = chunk.get("document_name") or chunk.get("source_document") or "Unknown"
            context_blocks.append(
                f"[Source: {doc_name}, Page: {chunk['page_number']}, Paragraph: {chunk['paragraph_number']}]\n{chunk['text']}"
            )
        retrieved_context = "\n\n".join(context_blocks)
        
        # 5. strict grounding prompt template using centralized system prompt
        prompt = f"""{RAG_SYSTEM_PROMPT}

Context:
{retrieved_context}

Question:
{question}"""

        # 6. Generate answer
        logger.info("Generating response from context...")
        answer = self._call_ollama(prompt, temperature=0.0)
        
        # Check if the LLM output indicates the answer was not found
        not_found_indicators = [
            "could not find this information", 
            "i could not find this", 
            "not present in the context",
            "not present in the uploaded document",
            "not present in the uploaded documents"
        ]
        is_not_found = any(indicator in answer.lower() for indicator in not_found_indicators)
        
        if is_not_found:
            clean_not_found = "I could not find this information in the uploaded documents."
            return clean_not_found, [], confidence

        # Extract unique sources
        seen_sources = set()
        sources_list = []
        for chunk in relevant_chunks:
            doc_name = str(chunk.get("document_name") or chunk.get("source_document") or "Unknown").strip()
            try:
                page_num = int(chunk.get("page_number", 1))
            except (ValueError, TypeError):
                page_num = 1
            try:
                para_num = int(chunk.get("paragraph_number", 1))
            except (ValueError, TypeError):
                para_num = 1
                
            source_key = (doc_name, page_num, para_num)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources_list.append({
                    "document_name": doc_name,
                    "page_number": page_num,
                    "paragraph_number": para_num
                })
        
        # 7. Record to memory only if a valid answer is generated
        self.memory.add_message("user", question)
        self.memory.add_message("assistant", answer)
        
        return answer, sources_list, confidence
