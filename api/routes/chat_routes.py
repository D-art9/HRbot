import os
import sys
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.hr_agent import stream_hr_query
from services.session_service import session_service
from services.memory_service import memory_service

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@router.post("")
async def chat(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = request.session_id or "default_session"
    user_id = request.user_id

    # Session mapping logic
    if session_id:
        if session_service.session_exists(session_id):
            if not user_id:
                user_id = session_service.get_user_id(session_id)
        else:
            if user_id:
                session_service.create_session(session_id, user_id)

    return StreamingResponse(
        stream_hr_query(request.query, user_id, session_id),
        media_type="text/event-stream"
    )

@router.post("/session/clear")
async def clear_session(session_id: str):
    memory_service.clear_memory(session_id)
    session_service.delete_session(session_id)
    return {"message": f"Session {session_id} cleared successfully"}
