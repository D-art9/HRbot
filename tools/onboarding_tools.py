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

ONBOARDING_FILE = "data/mock_data/onboarding.json"


def get_onboarding_status_tool(employee_id):
    """
    Fetch onboarding status for a new joiner.
    """
    try:
        with open(ONBOARDING_FILE, "r") as file:
            onboarding_data = json.load(file)

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


def get_pending_documents_tool(employee_id):
    """
    Fetch pending onboarding documents/tasks.
    """
    try:
        with open(ONBOARDING_FILE, "r") as file:
            onboarding_data = json.load(file)

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


@tool
def get_onboarding_status(employee_id: str) -> str:
    """
    Retrieves the onboarding progress, background verification status, 
    joining date, laptop allocation status, and payroll status for a new joiner using their Employee ID (e.g., 'ONB-201').
    """
    data = get_onboarding_status_tool(employee_id)
    if not data["success"]:
        return f"Error fetching onboarding status: {data['message']}"
    return json.dumps(data)


@tool
def get_pending_documents(employee_id: str) -> str:
    """
    Retrieves the list of outstanding identity, educational, or corporate 
    documents/tasks a new employee needs to submit using their Employee ID (e.g., 'ONB-201').
    """
    data = get_pending_documents_tool(employee_id)
    if not data["success"]:
        return f"Error fetching pending list: {data['message']}"
    return json.dumps(data)


def generate_onboarding_status_response(employee_id):
    """
    Generates professional onboarding update.
    """
    result = get_onboarding_status_tool(employee_id)
    if not result["success"]:
        return result["message"]

    prompt = f"""
You are a professional HR onboarding assistant for SVYIA Cybersecurity Solutions Pvt. Ltd.

Generate a professional and helpful onboarding status update for the new employee.

Onboarding Details:
- Employee Name: {result["employee_name"]}
- Orientation Status: {result["onboarding_status"]}
- Joining Date: {result["joining_date"]}
- Background Verification: {result["verification_status"]}
- Laptop Allocation: {result["laptop_allocation"]}
- Pending Documents: {result["pending_documents"]}
- Payroll Setup: {result["payroll_setup"]}

Maintain a welcoming and corporate tone.
"""
    response = llm.invoke(prompt)
    return response.content


def generate_pending_documents_response(employee_id):
    """
    Generates professional pending documents response.
    """
    result = get_pending_documents_tool(employee_id)
    if not result["success"]:
        return result["message"]

    prompt = f"""
You are a professional HR onboarding assistant for SVYIA Cybersecurity Solutions Pvt. Ltd.

Kindly inform the employee about their pending onboarding tasks and documents.

Details:
- Employee Name: {result["employee_name"]}
- Pending Tasks/Documents: {", ".join(result["pending_list"])}

Keep the tone encouraging and clear.
"""
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    # Test blocks
    print("--- Onboarding Status ---")
    print(generate_onboarding_status_response("ONB-203"))
    print("\n--- Pending Documents ---")
    print(generate_pending_documents_response("ONB-203"))
