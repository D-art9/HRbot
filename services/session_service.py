import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SessionService:
    def __init__(self):
        # Maps session_id -> { "user_id": str, "metadata": dict }
        self._sessions = {}

    def create_session(self, session_id: str, user_id: str, metadata: dict = None) -> dict:
        """
        Creates or updates a session with a user_id and optional metadata.
        """
        self._sessions[session_id] = {
            "user_id": user_id,
            "metadata": metadata or {}
        }
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> dict:
        """
        Gets the session object for the given session_id.
        """
        return self._sessions.get(session_id)

    def get_user_id(self, session_id: str) -> str:
        """
        Retrieves the user_id associated with a session.
        """
        session = self.get_session(session_id)
        return session.get("user_id") if session else None

    def session_exists(self, session_id: str) -> bool:
        """
        Checks if a session exists.
        """
        return session_id in self._sessions

    def delete_session(self, session_id: str):
        """
        Removes a session.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]


session_service = SessionService()
