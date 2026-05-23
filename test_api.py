import requests
import json

base_url = "http://127.0.0.1:8000"

def test_health():
    print("\n--- Testing Health Check ---")
    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(response.json())

def test_chat_flow():
    print("\n--- Testing Multi-Turn Chat Flow ---")
    
    # Turn 1
    payload1 = {
        "query": "I still haven't received my offer letter yet",
        "user_id": "CAND-103",
        "session_id": "session123"
    }
    print("Turn 1 Query:", payload1["query"])
    response1 = requests.post(f"{base_url}/api/chat", json=payload1)
    print(f"Status Code: {response1.status_code}")
    res_json1 = response1.json()
    print("AI Response:", res_json1.get("response")[:200] + "...")
    
    # Turn 2 - Context aware follow-up
    payload2 = {
        "query": "Who is my recruiter?",
        "session_id": "session123"
    }
    print("\nTurn 2 Query (Context-aware):", payload2["query"])
    response2 = requests.post(f"{base_url}/api/chat", json=payload2)
    print(f"Status Code: {response2.status_code}")
    res_json2 = response2.json()
    print("AI Response:", res_json2.get("response")[:300] + "...")

def test_routes():
    print("\n--- Testing Recruitment Endpoint ---")
    response = requests.get(f"{base_url}/api/recruitment/status/CAND-103")
    print(f"Status Code: {response.status_code}")
    print(response.json().get("response")[:150] + "...")
    
    print("\n--- Testing Onboarding Endpoint ---")
    response = requests.get(f"{base_url}/api/onboarding/documents/CAND-103")
    print(f"Status Code: {response.status_code}")
    print(response.json().get("response")[:150] + "...")

    print("\n--- Testing Policy RAG Endpoint ---")
    response = requests.get(f"{base_url}/api/policy/query", params={"query": "What is the policy on probation?"})
    print(f"Status Code: {response.status_code}")
    print(response.json().get("response")[:200] + "...")

if __name__ == "__main__":
    test_health()
    test_chat_flow()
    test_routes()
