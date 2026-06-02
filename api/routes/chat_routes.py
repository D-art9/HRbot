import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.hr_agent import stream_hr_query
from core.cyvia_tracing import (
    aclose_trace,
    create_chat_trace,
    wrap_sse_stream,
)
from services.memory_service import memory_service
from services.session_service import session_service

router = APIRouter(prefix="/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """Supports HRbot SSE clients and JSON clients (e.g. axios with messages array)."""

    query: Optional[str] = None
    message: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: Optional[bool] = None

    def resolved_query(self) -> str:
        if self.query and self.query.strip():
            return self.query.strip()
        if self.message and self.message.strip():
            return self.message.strip()
        if self.messages:
            for item in reversed(self.messages):
                if item.role == "user" and item.content.strip():
                    return item.content.strip()
        return ""

    def should_stream(self) -> bool:
        if self.stream is not None:
            return self.stream
        # Clients sending OpenAI-style message arrays typically expect JSON, not SSE.
        if self.messages:
            return False
        return True


def _resolve_session(session_id: Optional[str], user_id: Optional[str]) -> Tuple[str, Optional[str]]:
    session_id = session_id or "default_session"

    if session_service.session_exists(session_id):
        if not user_id:
            user_id = session_service.get_user_id(session_id)
    elif user_id:
        session_service.create_session(session_id, user_id)

    return session_id, user_id


async def _collect_stream_response(
    query: str,
    user_id: Optional[str],
    session_id: str,
    cyvia_handler=None,
) -> Dict[str, Any]:
    tokens: List[str] = []
    intent: Optional[str] = None
    error: Optional[str] = None

    async for chunk in stream_hr_query(
        query, user_id, session_id, cyvia_handler=cyvia_handler
    ):
        if not chunk.startswith("data: "):
            continue
        try:
            payload = json.loads(chunk[6:].strip())
        except json.JSONDecodeError:
            continue

        event_type = payload.get("type")
        if event_type == "token":
            tokens.append(payload.get("content", ""))
        elif event_type == "metadata" and payload.get("intent"):
            intent = payload["intent"]
        elif event_type == "error":
            error = payload.get("message", "Unknown error")
        elif event_type == "done":
            break

    response_text = "".join(tokens)
    if error:
        return {
            "success": False,
            "response": response_text,
            "intent": intent,
            "session_id": session_id,
            "error": error,
        }

    return {
        "success": True,
        "response": response_text,
        "intent": intent,
        "session_id": session_id,
    }


@router.post("")
async def chat(request: ChatRequest):
    query = request.resolved_query()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty. Send 'query', 'message', or a user entry in 'messages'.",
        )

    session_id, user_id = _resolve_session(request.session_id, request.user_id)

    # --- Cyvia root trace (one per HTTP chat request) ---
    cyvia_ctx = create_chat_trace(
        session_id=session_id,
        user_id=user_id,
        query_preview=query,
    )
    cyvia_handler = cyvia_ctx.handler if cyvia_ctx else None
    cyvia_trace = cyvia_ctx.trace if cyvia_ctx else None

    if not request.should_stream():
        try:
            result = await _collect_stream_response(
                query, user_id, session_id, cyvia_handler=cyvia_handler
            )
            if not result.get("success"):
                raise HTTPException(
                    status_code=500, detail=result.get("error", "Chat failed")
                )
            return JSONResponse(
                {
                    "success": True,
                    "data": result["response"],
                    "response": result["response"],
                    "intent": result.get("intent"),
                    "session_id": session_id,
                }
            )
        finally:
            await aclose_trace(cyvia_trace)

    async def event_stream():
        async for chunk in wrap_sse_stream(
            stream_hr_query(
                query,
                user_id,
                session_id,
                cyvia_handler=cyvia_handler,
            ),
            cyvia_trace,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/json")
async def chat_json(request: ChatRequest):
    """Non-streaming JSON endpoint for clients that call response.json()."""
    json_request = request.model_copy(update={"stream": False})
    return await chat(json_request)


@router.get("/stream")
async def chat_stream_get(
    query: str = Query(..., min_length=1),
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """GET + SSE for browsers using EventSource (POST + fetch also supported)."""
    session_id, user_id = _resolve_session(session_id, user_id)

    cyvia_ctx = create_chat_trace(
        session_id=session_id,
        user_id=user_id,
        query_preview=query,
    )
    cyvia_handler = cyvia_ctx.handler if cyvia_ctx else None
    cyvia_trace = cyvia_ctx.trace if cyvia_ctx else None

    async def event_stream():
        async for chunk in wrap_sse_stream(
            stream_hr_query(
                query,
                user_id,
                session_id,
                cyvia_handler=cyvia_handler,
            ),
            cyvia_trace,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/session/clear")
async def clear_session(session_id: str):
    memory_service.clear_memory(session_id)
    session_service.delete_session(session_id)
    return {"message": f"Session {session_id} cleared successfully"}
