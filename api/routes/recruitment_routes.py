import os
import sys
from fastapi import APIRouter, HTTPException

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.recruitment_tools import (
    generate_candidate_status_response,
    generate_interview_schedule_response,
    generate_offer_status_response,
    generate_recruiter_contact_response,
)

router = APIRouter(prefix="/recruitment", tags=["recruitment"])


@router.get("/status/{user_id}")
async def get_candidate_status(user_id: str):
    try:
        response = generate_candidate_status_response(user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interview/{user_id}")
async def get_interview_schedule(user_id: str):
    try:
        response = generate_interview_schedule_response(user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/offer/{user_id}")
async def get_offer_status(user_id: str):
    try:
        response = generate_offer_status_response(user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recruiter/{user_id}")
async def get_recruiter_contact(user_id: str):
    try:
        response = generate_recruiter_contact_response(user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
