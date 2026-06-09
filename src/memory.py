import logging

logger = logging.getLogger(__name__)

class ChatMemory:
    """
    Manages conversational memory within the current session, keeping track of 
    a rolling window of user queries and assistant responses.
    """
    def __init__(self, max_history_turns: int = 5):
        """
        Initializes ChatMemory.
        
        Args:
            max_history_turns (int): Maximum number of conversation turns (Q&A pairs) to keep.
        """
        self.history = []  # Stores message dicts: {'role': 'user'|'assistant', 'content': str}
        self.max_history_turns = max_history_turns

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the session history.
        
        Args:
            role (str): The speaker, either 'user' or 'assistant'.
            content (str): The text message content.
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'")
            
        self.history.append({"role": role, "content": content})
        
        # A turn is a pair of (user, assistant) messages.
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
            logger.debug(f"Memory trimmed to last {self.max_history_turns} turns.")

    def get_history(self) -> list[dict]:
        """
        Retrieves the list of raw history messages.
        
        Returns:
            list[dict]: List of messages.
        """
        return self.history

    def get_formatted_history(self) -> str:
        """
        Formats history into a single string for prompting.
        
        Returns:
            str: Conversation transcript formatted as "User: ... \n Bot: ...".
        """
        formatted_turns = []
        for msg in self.history:
            speaker = "User" if msg["role"] == "user" else "Bot"
            formatted_turns.append(f"{speaker}: {msg['content']}")
        return "\n".join(formatted_turns)

    def is_empty(self) -> bool:
        """Checks if memory contains any messages."""
        return len(self.history) == 0

    def clear(self) -> None:
        """Clears the conversational memory."""
        self.history = []
        logger.info("Conversational memory cleared.")
