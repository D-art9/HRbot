# LLM Provider configuration and factory
import sys

from langchain_groq import ChatGroq

from core.config import settings

if not settings.GROQ_API_KEY:
    print("\n[!] WARNING: GROQ_API_KEY is not set in your .env file.")
    print(
        "Please add GROQ_API_KEY=your_key_here to your .env file to use the HR Agent.\n"
    )

try:
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.MODEL_NAME,
        temperature=0,
    )
except Exception as e:
    print(f"\n[!] Error initializing ChatGroq: {e}")
    # Provide a dummy LLM or exit depending on how you want to handle it
    # For now, we'll let it fail but with a clearer message if possible
    llm = None
