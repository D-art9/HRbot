import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from langchain_core.tools import tool
except ImportError:
    try:
        from langchain.tools import tool
    except ImportError:
        from langchain_classic.tools import tool

from modules.rag_module import ask_question


@tool
def policy_rag_tool(query: str) -> str:
    """
    Connects to the company policy RAG database to answer specific employee policy questions (e.g. leaves, code of conduct, payroll policies).
    """
    try:
        answer, sources = ask_question(query)
        return {
            "success": True,
            "answer": answer,
            "sources": [
                f"{doc.metadata.get('source_file')} (Page {doc.metadata.get('page')})"
                for doc in sources
            ],
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


def generate_policy_response(query):
    """
    Generates and formats the policy response for the user.
    """
    result = policy_rag_tool.invoke({"query": query})
    if not result["success"]:
        return f"I encountered an error while searching for the policy: {result['message']}"

    # The RAG system already returns a professional formatted response
    # We just ensure it's presented cleanly with source attribution
    response = result["answer"]

    if result["sources"]:
        response += "\n\nSources:"
        for source in set(
            result["sources"]
        ):  # Use set to avoid duplicate source listings
            response += f"\n- {source}"

    return response


if __name__ == "__main__":
    # Test blocks
    print("--- Policy Query ---")
    print(generate_policy_response("What is the leave policy?"))
