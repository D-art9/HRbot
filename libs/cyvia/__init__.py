from .adapters.claude import CLAUDE_AGENT_SDK, observe
from .claude_web_hooks import (
    build_claude_web_hooks,
    fetch_claude_web_tool_hooks,
    merge_claude_hooks,
)
from .client import CyviaClient
from .trace import Trace, new_id
from .web_tool_access_policy import WebToolAccessPolicy, fetch_web_tool_access_policy

__all__ = [
    "CLAUDE_AGENT_SDK",
    "build_claude_web_hooks",
    "CyviaClient",
    "fetch_claude_web_tool_hooks",
    "merge_claude_hooks",
    "LANGCHAIN",
    "LangChainCallbackHandler",
    "Trace",
    "WebToolAccessPolicy",
    "fetch_web_tool_access_policy",
    "new_id",
    "observe",
]


def __getattr__(name: str):
    if name == "LANGCHAIN":
        try:
            from cyvia.adapters.langchain import LANGCHAIN as v
        except ImportError as e:
            raise ImportError(
                "LangChain support requires langchain-core. "
                "Install it in the same environment as cyvia (e.g. pip install langchain-core)."
            ) from e
        return v
    if name == "LangChainCallbackHandler":
        try:
            from cyvia.adapters.langchain import LangChainCallbackHandler as v
        except ImportError as e:
            raise ImportError(
                "LangChain support requires langchain-core. "
                "Install it in the same environment as cyvia (e.g. pip install langchain-core)."
            ) from e
        return v
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
