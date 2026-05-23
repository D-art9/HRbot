import os
import sys
from fastapi import APIRouter, HTTPException

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from services.rag_service import rag_service

router = APIRouter(prefix="/policy", tags=["policy"])


@router.get("/query")
async def query_policy(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    try:
        response = rag_service.query_policy_formatted(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_policy(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    try:
        result = rag_service.query_policy(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
