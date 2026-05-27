import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory


class MemoryService:
    def __init__(self):
        # Dictionary mapping session_id -> ConversationBufferMemory
        self._memories = {}

    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        """
        Retrieves or creates a ConversationBufferMemory instance for a session.
        """
        if not session_id:
            # Fallback to a default memory if session_id is not provided
            session_id = "default_session"

        if session_id not in self._memories:
            self._memories[session_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                input_key="input",
                output_key="output"
            )
        return self._memories[session_id]

    def get_history(self, session_id: str):
        """
        Retrieves the chat history string for a session.
        """
        memory = self.get_memory(session_id)
        chat_history = memory.load_memory_variables({})
        return chat_history.get("history", "")

    def clear_memory(self, session_id: str):
        """
        Clears the conversation memory for a session.
        """
        if session_id in self._memories:
            self._memories[session_id].clear()


memory_service = MemoryService()
