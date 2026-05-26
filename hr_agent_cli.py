# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-core",
#     "langchain-groq",
#     "python-dotenv",
# ]
# ///
"""
Cyvia HR Agent CLI.

A single-file LangChain agent that uses manual tool-calling (no AgentExecutor,
no create_agent) to answer recruitment and onboarding questions for Cyvia
Cybersecurity Solutions Pvt. Ltd.

Run with:
    uv run hr_agent_cli.py

Requires GROQ_API_KEY (and optionally MODEL_NAME) in a .env file at the repo
root. Must be run from the repo root so the data/mock_data/*.json paths
resolve.
"""

# Sample output from an execution
"""
============================================================
                    SVYIA HR AGENT (CLI)
============================================================
Type your question. Use /reset to clear history, /exit to quit.

Enter Candidate/Employee ID (e.g. CAND-101 or ONB-201): CAND-101

You: what is my status?
  [iter 1] model requested 1 tool call(s):
    -> get_candidate_status({'candidate_id': 'CAND-101'})
    <- {"success": true, "candidate": {"candidate_id": "CAND-101", "full_name": "Aarav Mehta", "email": "aarav.mehta@outlook.com", "phone": "+91-9876543210", "position_applied": "SOC Analyst", "experience_ye...

AI: It appears that your application status is "Interview Scheduled" for the position of SOC Analyst. Your interview is scheduled on June 10, 2026, at 11:00 AM via Google Meet.

You: what else can you tell me about me
  [iter 1] model requested 1 tool call(s):
    -> get_candidate_status({'candidate_id': 'CAND-101'})
    <- {"success": true, "candidate": {"candidate_id": "CAND-101", "full_name": "Aarav Mehta", "email": "aarav.mehta@outlook.com", "phone": "+91-9876543210", "position_applied": "SOC Analyst", "experience_ye...

AI: You are Aarav Mehta, a candidate for the SOC Analyst position at SVYIA Cybersecurity Solutions Pvt. Ltd. You have 3 years of experience and are currently based in Bengaluru. Your recruiter is Priya Sharma, and your interview is scheduled on June 10, 2026, at 11:00 AM via Google Meet.

You: what other tools do you access to

AI: I have access to the following tools:

1. `get_candidate_status`: This tool provides information about a candidate's recruitment application status, including their full name, role applied for, current application stage, and assigned recruiter.
2. `get_interview_schedule`: This tool provides information about a candidate's scheduled interview details, including the date, time, mode (e.g. Google Meet / Microsoft Teams), recruiter, and current application stage.
3. `get_offer_status`: This tool provides information about a candidate's offer and onboarding status, including the offer state, role, assigned recruiter, and onboarding progress.
4. `get_recruiter_contact`: This tool provides the recruiter contact details (name, email, phone) for the recruiter assigned to a given candidate.
5. `get_onboarding_status`: This tool provides information about a new hire's onboarding status, including orientation progress, joining date, background check state, laptop allocation, document verification, and payroll setup.
6. `get_pending_documents`: This tool provides the list of pending onboarding documents and tasks for a new hire.

These tools allow me to provide accurate and up-to-date information to candidates and new hires about their recruitment and onboarding status.

You: what is my interview schedule and who do i contac
  [iter 1] model requested 2 tool call(s):
    -> get_interview_schedule({'candidate_id': 'CAND-101'})
    <- {"success": true, "candidate_name": "Aarav Mehta", "role": "SOC Analyst", "application_status": "Interview Scheduled", "interview_date": "2026-06-10", "interview_time": "11:00 AM", "interview_mode": "...
    -> get_recruiter_contact({'candidate_id': 'CAND-101'})
    <- {"success": true, "recruiter_name": "Priya Sharma", "recruiter_email": "priya.sharma@svyia.com", "recruiter_phone": "+91-0000000000", "role_applied": "SOC Analyst"}

AI: Your interview schedule is as follows:

* Date: June 10, 2026
* Time: 11:00 AM
* Mode: Google Meet

You can contact your recruiter, Priya Sharma, at the following details:

* Email: priya.sharma@svyia.com
* Phone: +91-0000000000

Please note that the phone number is not a real number and is just a placeholder. You should contact Priya Sharma at the email address provided.
""" 

import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_FILE = os.path.join(SCRIPT_DIR, "data", "mock_data", "candidates.json")
EMPLOYEES_FILE = os.path.join(SCRIPT_DIR, "data", "mock_data", "employees.json")
ONBOARDING_FILE = os.path.join(SCRIPT_DIR, "data", "mock_data", "onboarding.json")

DEBUG = True
MAX_ITERATIONS = 6


# ---------------------------------------------------------------------------
# Tools (copied & lightly adapted from tools/recruitment_tools.py and
# tools/onboarding_tools.py).
# ---------------------------------------------------------------------------


@tool
def get_candidate_status(candidate_id: str) -> dict:
    """Get a candidate's recruitment application status: full name, role
    applied for, current application stage, and assigned recruiter.

    Use when the user asks about their application status, where they are in
    the recruitment process, or for general candidate information.

    Args:
        candidate_id: Candidate ID, e.g. 'CAND-101'. Candidate IDs always
            start with the 'CAND-' prefix.
    """
    try:
        with open(CANDIDATES_FILE, "r") as f:
            candidates = json.load(f)
        for candidate in candidates:
            if candidate["candidate_id"] == candidate_id:
                return {"success": True, "candidate": candidate}
        return {"success": False, "message": "Candidate not found."}
    except Exception as e:
        return {"success": False, "message": str(e)}


@tool
def get_interview_schedule(candidate_id: str) -> dict:
    """Get a candidate's scheduled interview details: date, time, mode (e.g.
    Google Meet / Microsoft Teams), recruiter, and current application stage.

    Use when the user asks about their interview, when it is scheduled, the
    meeting link/mode, or related logistics.

    Args:
        candidate_id: Candidate ID, e.g. 'CAND-101'.
    """
    try:
        with open(CANDIDATES_FILE, "r") as f:
            candidates = json.load(f)
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


@tool
def get_offer_status(candidate_id: str) -> dict:
    """Get a candidate's offer and onboarding status: offer state, role,
    assigned recruiter, and onboarding progress.

    Use when the user asks whether their offer has been released, the offer
    status, or how onboarding is progressing post-offer.

    Args:
        candidate_id: Candidate ID, e.g. 'CAND-101'.
    """
    try:
        with open(CANDIDATES_FILE, "r") as f:
            candidates = json.load(f)
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


@tool
def get_recruiter_contact(candidate_id: str) -> dict:
    """Get the recruiter contact details (name, email, phone) for the
    recruiter assigned to a given candidate.

    Use when the user asks who their recruiter is, how to reach HR, or for
    a recruiter's email or phone number.

    Args:
        candidate_id: Candidate ID, e.g. 'CAND-101'.
    """
    try:
        with open(CANDIDATES_FILE, "r") as f:
            candidates = json.load(f)

        recruiter_name = None
        role_applied = None
        for candidate in candidates:
            if candidate["candidate_id"] == candidate_id:
                recruiter_name = candidate["recruiter"]
                role_applied = candidate["position_applied"]
                break

        if not recruiter_name:
            return {"success": False, "message": "Candidate or recruiter not found."}

        recruiter_email = "hr@Cyvia.com"
        recruiter_phone = "+91-0000000000"

        if os.path.exists(EMPLOYEES_FILE):
            with open(EMPLOYEES_FILE, "r") as f:
                employees = json.load(f)
            for emp in employees:
                if emp["full_name"] == recruiter_name:
                    recruiter_email = emp.get("email", recruiter_email)
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
def get_onboarding_status(employee_id: str) -> dict:
    """Get a new hire's onboarding status: orientation progress, joining
    date, background check state, laptop allocation, document verification,
    and payroll setup.

    Use when the user asks about their onboarding progress, joining date,
    laptop assignment, or background verification.

    Args:
        employee_id: Onboarding/employee ID, e.g. 'ONB-201'. Onboarding IDs
            always start with the 'ONB-' prefix.
    """
    try:
        with open(ONBOARDING_FILE, "r") as f:
            onboarding_data = json.load(f)
        for record in onboarding_data:
            if record["employee_id"] == employee_id:
                return {
                    "success": True,
                    "employee_name": record["employee_name"],
                    "onboarding_status": record.get("orientation_status", "N/A"),
                    "pending_documents": "Required"
                    if not record.get("documents_verified", False)
                    else "None",
                    "joining_date": record.get("joining_date", "N/A"),
                    "verification_status": record.get("background_check", "N/A"),
                    "laptop_allocation": record.get("laptop_assigned", "N/A"),
                    "payroll_setup": "Pending"
                    if not record.get("documents_verified", False)
                    else "Completed",
                }
        return {"success": False, "message": "Onboarding record not found."}
    except Exception as e:
        return {"success": False, "message": str(e)}


@tool
def get_pending_documents(employee_id: str) -> dict:
    """Get the list of pending onboarding documents and tasks for a new hire
    (e.g. identity proof, educational certificates, orientation, IT asset
    collection).

    Use when the user asks what documents or tasks they still need to
    submit/complete for onboarding.

    Args:
        employee_id: Onboarding/employee ID, e.g. 'ONB-201'.
    """
    try:
        with open(ONBOARDING_FILE, "r") as f:
            onboarding_data = json.load(f)
        for record in onboarding_data:
            if record["employee_id"] == employee_id:
                pending = []
                if not record.get("documents_verified", False):
                    pending.append("Identity Proof")
                    pending.append("Educational Certificates")
                    pending.append("Relieving Letter")
                if record.get("orientation_status") == "Pending":
                    pending.append("Orientation Session")
                if record.get("laptop_assigned") == "Not Assigned":
                    pending.append("IT Asset Collection")
                return {
                    "success": True,
                    "employee_name": record["employee_name"],
                    "pending_list": pending if pending else ["No pending items"],
                }
        return {"success": False, "message": "Onboarding record not found."}
    except Exception as e:
        return {"success": False, "message": str(e)}


TOOLS = [
    get_candidate_status,
    get_interview_schedule,
    get_offer_status,
    get_recruiter_contact,
    get_onboarding_status,
    get_pending_documents,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Cyvia HR Assistant, an AI helping candidates and \
new hires at Cyvia Cybersecurity Solutions Pvt. Ltd.

The user you are speaking to has ID: {user_id}.
- IDs starting with 'CAND-' (e.g. CAND-101) are candidates in the recruitment pipeline.
- IDs starting with 'ONB-' (e.g. ONB-201) are new hires in onboarding.

You have tools to look up candidate and onboarding records. ALWAYS use the \
appropriate tool to fetch data — never invent or guess candidate, interview, \
offer, recruiter, or onboarding details. If a tool returns success=False, \
relay the error message to the user politely.

When the user asks a question without specifying whose record to look at, \
default to their own ID: {user_id}. If they explicitly ask about another \
person, ask for that person's ID.

Topics you may answer (using tools):
- Recruitment: application status, interview schedule, offer status, recruiter contact.
- Onboarding: onboarding progress, pending documents/tasks.

You MUST politely decline any question outside those topics — including \
questions about company policies, leave, benefits, compensation, the \
employee handbook, training programs, code of conduct, or general HR \
policy. For those, respond with something like: "I can only help with \
recruitment and onboarding questions today. For policy queries, please \
reach out to HR directly." Do NOT answer policy questions from your own \
knowledge.

Keep responses concise, professional, and corporate in tone.
"""


def build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0,
    )
    return llm.bind_tools(TOOLS)


def run_turn(llm_with_tools, messages: list) -> str:
    """Run the manual tool-calling loop until the model produces a final
    answer (no more tool_calls) or we hit MAX_ITERATIONS.
    """
    for iteration in range(MAX_ITERATIONS):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return ai_msg.content or "(no content)"

        if DEBUG:
            print(f"  [iter {iteration + 1}] model requested {len(ai_msg.tool_calls)} tool call(s):")

        for tool_call in ai_msg.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            if DEBUG:
                print(f"    -> {name}({args})")

            tool_fn = TOOLS_BY_NAME.get(name)
            if tool_fn is None:
                from langchain_core.messages import ToolMessage
                messages.append(
                    ToolMessage(
                        content=json.dumps({"success": False, "message": f"Unknown tool: {name}"}),
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            tool_result = tool_fn.invoke(tool_call)
            if DEBUG:
                preview = str(tool_result.content)[:200]
                print(f"    <- {preview}{'...' if len(str(tool_result.content)) > 200 else ''}")
            messages.append(tool_result)

    return "(reached MAX_ITERATIONS without a final answer)"


def main() -> None:
    print("\n" + "=" * 60)
    print("Cyvia HR AGENT (CLI)".center(60))
    print("=" * 60)
    print("Type your question. Use /reset to clear history, /exit to quit.\n")

    user_id = ""
    while not user_id:
        user_id = input("Enter Candidate/Employee ID (e.g. CAND-101 or ONB-201): ").strip()

    llm_with_tools = build_llm()

    def fresh_messages() -> list:
        return [SystemMessage(content=SYSTEM_PROMPT.format(user_id=user_id))]

    messages = fresh_messages()

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not query:
            continue

        lowered = query.lower()
        if lowered in {"/exit", "exit", "quit", "/quit"}:
            print("Exiting.")
            return
        if lowered == "/reset":
            messages = fresh_messages()
            print("(conversation history cleared)")
            continue

        messages.append(HumanMessage(content=query))
        answer = run_turn(llm_with_tools, messages)
        print(f"\nAI: {answer}")


if __name__ == "__main__":
    main()
