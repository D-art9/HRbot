import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.policy_tools import policy_rag_tool, generate_policy_response


class RAGService:
    def query_policy(self, query: str):
        """
        Executes the policy query against the RAG vector store and returns raw dictionary results.
        """
        return policy_rag_tool(query)

    def query_policy_formatted(self, query: str) -> str:
        """
        Executes the policy query and returns a formatted response with source attribution.
        """
        return generate_policy_response(query)


rag_service = RAGService()
