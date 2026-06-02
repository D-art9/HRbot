import json
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_provider import llm
try:
    from langchain_core.tools import tool
except ImportError:
    try:
        from langchain.tools import tool
    except ImportError:
        from langchain_classic.tools import tool

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANDIDATES_FILE = os.path.join(_project_root, "data", "mock_data", "candidates.json")
EMPLOYEES_FILE = os.path.join(_project_root, "data", "mock_data", "employees.json")


def get_candidate_status_tool(candidate_id):
    """
    Fetch candidate details using candidate ID.
    """
    try:
        with open(CANDIDATES_FILE, "r") as file:
            candidates = json.load(file)

        for candidate in candidates:
            if candidate["candidate_id"] == candidate_id:
                return {"success": True, "candidate": candidate}

        return {"success": False, "message": "Candidate not found."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_interview_schedule_tool(candidate_id):
    """
    Fetch interview schedule for candidate.
    """
    try:
        with open(CANDIDATES_FILE, "r") as file:
            candidates = json.load(file)

        for candidate in candidates:
            if candidate["candidate_id"] == candidate_id:
                return {
                    "success": True,
                    "candidate_name": candidate["full_name"],
                    "role": candidate["position_applied"],
                    "application_status": candidate["application_status"],
                    "interview_date": candidate.get("interview_date", "Not Scheduled"),
                    "interview_time": candidate.get("interview_time", "N/A"),
                    "interview_mode": candidate.get("interview_mode", "N/A"),
                    "recruiter": candidate["recruiter"],
                }

        return {"success": False, "message": "Candidate not found."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_offer_status_tool(candidate_id):
    """
    Fetch offer and onboarding status for a candidate.
    """
    try:
        with open(CANDIDATES_FILE, "r") as file:
            candidates = json.load(file)

        for candidate in candidates:
            if candidate["candidate_id"] == candidate_id:
                return {
                    "success": True,
                    "full_name": candidate["full_name"],
                    "offer_status": candidate.get("offer_status", "N/A"),
                    "role": candidate["position_applied"],
                    "recruiter": candidate["recruiter"],
                    "onboarding_status": candidate.get("onboarding_status", "N/A"),
                }

        return {"success": False, "message": "Candidate not found."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_recruiter_contact_tool(candidate_id):
    """
    Fetch recruiter contact details for a candidate.
    """
    try:
        with open(CANDIDATES_FILE, "r") as file:
            candidates = json.load(file)

        recruiter_name = None
        role_applied = None
        for candidate in candidates:
            if candidate["candidate_id"] == candidate_id:
                recruiter_name = candidate["recruiter"]
                role_applied = candidate["position_applied"]
                break

        if not recruiter_name:
            return {"success": False, "message": "Candidate or recruiter not found."}

        # Try to find recruiter email in employees.json
        recruiter_email = "hr@svyia.com"
        recruiter_phone = "+91-0000000000"

        if os.path.exists(EMPLOYEES_FILE):
            with open(EMPLOYEES_FILE, "r") as file:
                employees = json.load(file)
            for emp in employees:
                if emp["full_name"] == recruiter_name:
                    recruiter_email = emp.get("email", recruiter_email)
                    # phone not in employees.json, using default
                    break

        return {
            "success": True,
            "recruiter_name": recruiter_name,
            "recruiter_email": recruiter_email,
            "recruiter_phone": recruiter_phone,
            "role_applied": role_applied,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


@tool
def get_candidate_status(candidate_id: str) -> str:
    """
    Retrieves the recruitment status (e.g. Applied, Interviewing, Offer Sent, Rejected) 
    and recruiter info for a candidate using their Candidate ID (e.g., 'CAND-101').
    """
    data = get_candidate_status_tool(candidate_id)
    if not data["success"]:
        return f"Error fetching status: {data['message']}"
    return json.dumps(data["candidate"])


@tool
def get_interview_schedule(candidate_id: str) -> str:
    """
    Retrieves the interview date, time, mode (Online/Offline), and panelist 
    for a candidate using their Candidate ID (e.g., 'CAND-101').
    """
    data = get_interview_schedule_tool(candidate_id)
    if not data["success"]:
        return f"Error fetching schedule: {data['message']}"
    return json.dumps(data)


@tool
def get_offer_status(candidate_id: str) -> str:
    """
    Retrieves the final offer status (e.g., Sent, Accepted, Declined) 
    and onboarding start details for a candidate using their Candidate ID (e.g., 'CAND-101').
    """
    data = get_offer_status_tool(candidate_id)
    if not data["success"]:
        return f"Error fetching offer details: {data['message']}"
    return json.dumps(data)


@tool
def get_recruiter_contact(candidate_id: str) -> str:
    """
    Retrieves contact details (Name, Email, Phone) of the assigned HR recruiter 
    responsible for a candidate's application, using the Candidate ID (e.g., 'CAND-101').
    """
    data = get_recruiter_contact_tool(candidate_id)
    if not data["success"]:
        return f"Error fetching recruiter contact: {data['message']}"
    return json.dumps(data)


def generate_candidate_status_response(candidate_id):
    """
    Generates professional HR response for candidate status.
    """
    result = get_candidate_status_tool(candidate_id)
    if not result["success"]:
        return result["message"]

    candidate = result["candidate"]
    prompt = f"""
You are a professional HR recruitment assistant for SVYIA Cybersecurity Solutions Pvt. Ltd.

Generate a concise and professional recruitment status update for the candidate.

Candidate Details:
- Name: {candidate["full_name"]}
- Role Applied: {candidate["position_applied"]}
- Application Status: {candidate["application_status"]}
- Recruiter: {candidate["recruiter"]}

Keep the response professional and employee-facing.
"""
    response = llm.invoke(prompt)
    return response.content


def generate_interview_schedule_response(candidate_id):
    """
    Generates professional interview schedule response.
    """
    result = get_interview_schedule_tool(candidate_id)
    if not result["success"]:
        return result["message"]

    prompt = f"""
You are a professional HR recruitment assistant for SVYIA Cybersecurity Solutions Pvt. Ltd.

Generate a professional interview schedule update for the candidate.

Candidate Details:
- Candidate Name: {result["candidate_name"]}
- Role Applied: {result["role"]}
- Application Status: {result["application_status"]}
- Interview Date: {result["interview_date"]}
- Interview Time: {result["interview_time"]}
- Interview Mode: {result["interview_mode"]}
- Recruiter: {result["recruiter"]}

Keep the response concise, professional, and employee-facing.
"""
    response = llm.invoke(prompt)
    return response.content


def generate_offer_status_response(candidate_id):
    """
    Generates professional HR offer response.
    """
    result = get_offer_status_tool(candidate_id)
    if not result["success"]:
        return result["message"]

    prompt = f"""
You are a professional HR recruitment assistant for SVYIA Cybersecurity Solutions Pvt. Ltd.

Generate a professional and welcoming update regarding the candidate's offer status.

Candidate Details:
- Name: {result["full_name"]}
- Role: {result["role"]}
- Offer Status: {result["offer_status"]}
- Onboarding Status: {result["onboarding_status"]}
- Assigned Recruiter: {result["recruiter"]}

Maintain an enterprise HR tone.
"""
    response = llm.invoke(prompt)
    return response.content


def generate_recruiter_contact_response(candidate_id):
    """
    Generates professional recruiter contact response.
    """
    result = get_recruiter_contact_tool(candidate_id)
    if not result["success"]:
        return result["message"]

    prompt = f"""
You are a professional HR recruitment assistant for SVYIA Cybersecurity Solutions Pvt. Ltd.

Provide the recruiter's contact details to the candidate in a professional manner.

Details:
- Candidate Applied for: {result["role_applied"]}
- Recruiter Name: {result["recruiter_name"]}
- Recruiter Email: {result["recruiter_email"]}
- Recruiter Phone: {result["recruiter_phone"]}

Keep the response helpful and professional.
"""
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    # Test blocks
    print("--- Candidate Status ---")
    print(generate_candidate_status_response("CAND-101"))
    print("\n--- Interview Schedule ---")
    print(generate_interview_schedule_response("CAND-101"))
    print("\n--- Offer Status ---")
    print(generate_offer_status_response("CAND-103"))
    print("\n--- Recruiter Contact ---")
    print(generate_recruiter_contact_response("CAND-101"))
