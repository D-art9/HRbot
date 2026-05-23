import os
import sys
from fastapi import APIRouter, HTTPException

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.onboarding_tools import (
    generate_onboarding_status_response,
    generate_pending_documents_response,
)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


@router.get("/status/{user_id}")
async def get_onboarding_status(user_id: str):
    try:
        response = generate_onboarding_status_response(user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{user_id}")
async def get_pending_documents(user_id: str):
    try:
        response = generate_pending_documents_response(user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
