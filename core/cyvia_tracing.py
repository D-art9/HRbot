"""
Cyvia observability for HRbot (LangGraph + LangChain + Groq).

Root traces are created per HTTP chat request; LangChainCallbackHandler is passed
into LangGraph via config["callbacks"] so child LLM/tool runs are recorded.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Optional

from cyvia.adapters.langchain import LANGCHAIN, LangChainCallbackHandler
from cyvia.client import CyviaClient
from cyvia.trace import Trace

from core.config import settings
from core.logger import logger

_client: CyviaClient | None = None


@dataclass(frozen=True)
class CyviaTraceContext:
    trace: Trace
    handler: LangChainCallbackHandler


def is_cyvia_enabled() -> bool:
    return bool(settings.CYVIA_API_KEY and settings.CYVIA_API_KEY.strip())


def get_cyvia_client() -> CyviaClient | None:
    """Singleton Cyvia HTTP client (disabled when CYVIA_API_KEY is unset)."""
    global _client
    if not is_cyvia_enabled():
        return None
    if _client is None:
        _client = CyviaClient(
            api_key=settings.CYVIA_API_KEY,
            base_url=settings.CYVIA_BASE_URL,
        )
        logger.info("CYVIA: Client initialized (base_url=%s)", settings.CYVIA_BASE_URL)
        
        # Temporary debug logging for Cyvia troubleshooting
        import importlib.metadata
        try:
            sdk_version = importlib.metadata.version("cyvia")
        except Exception:
            sdk_version = "unknown"
        logger.info("CYVIA DEBUG: SDK Version: %s", sdk_version)
        logger.info("CYVIA DEBUG: Resolved Base URL: %s", settings.CYVIA_BASE_URL)
        logger.info("CYVIA DEBUG: Client Config _base URL: %s", _client._base)
        logger.info("CYVIA DEBUG: Final Trace Start Endpoint: %s/agent/traces", _client._base)
        logger.info("CYVIA DEBUG: Final Ingest Endpoint: %s/agent/traces/<trace_id>/ingest", _client._base)
    return _client


def create_chat_trace(
    *,
    session_id: str,
    user_id: Optional[str] = None,
    query_preview: Optional[str] = None,
) -> Optional[CyviaTraceContext]:
    """
    Create and register a root Cyvia trace for one user chat request.

    Call close_trace() when the request finishes (including after SSE streaming).
    """
    client = get_cyvia_client()
    if client is None:
        return None

    description = "HR chat request"
    if query_preview:
        description = f"HR chat: {query_preview[:200]}"

    trace = Trace(
        client=client,
        adapter=LANGCHAIN,
        runtime="langgraph",
        agent_name="hrbot",
        external_agent_id=session_id,
        agent_display_name="SVYIA HR AI Agent",
        agent_description=description,
        model_name=settings.MODEL_NAME or "unknown",
    )
    try:
        trace.start()
        handler = LangChainCallbackHandler(trace)
        logger.info(
            "CYVIA: Root trace started trace_id=%s session_id=%s user_id=%s",
            trace.trace_id,
            session_id,
            user_id,
        )
        return CyviaTraceContext(trace=trace, handler=handler)
    except Exception as exc:
        logger.warning("CYVIA: Failed to start trace: %s", exc)
        return None


def build_langgraph_config(
    session_id: Optional[str],
    handler: Optional[LangChainCallbackHandler] = None,
) -> dict[str, Any]:
    """
    LangGraph RunnableConfig preserving thread_id and optional Cyvia callbacks.

    Callbacks on this config propagate to graph nodes (agent, ToolNode) and
    LangChain runnables invoked with inherited config.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id or "default_session"},
    }
    if handler is not None:
        config["callbacks"] = [handler]
    return config


def close_trace(trace: Optional[Trace]) -> None:
    """Flush pending spans/events and end the Cyvia session (sync, idempotent)."""
    if trace is None:
        return
    try:
        trace.flush()
        trace.close()
        logger.info("CYVIA: Trace closed trace_id=%s", trace.trace_id)
    except Exception as exc:
        logger.warning("CYVIA: Trace close failed: %s", exc)


async def aclose_trace(trace: Optional[Trace]) -> None:
    """Async-safe trace.close() for FastAPI streaming paths."""
    if trace is None:
        return
    await asyncio.to_thread(close_trace, trace)


async def wrap_sse_stream(
    stream: AsyncIterator[str],
    trace: Optional[Trace],
) -> AsyncIterator[str]:
    """Ensure trace.close() runs after the SSE generator completes or errors."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await aclose_trace(trace)


def iter_sync_stream_with_trace(
    stream: Iterator[str],
    trace: Optional[Trace],
) -> Iterator[str]:
    """Sync variant for non-async iterators (unused by default API path)."""
    try:
        yield from stream
    finally:
        close_trace(trace)
