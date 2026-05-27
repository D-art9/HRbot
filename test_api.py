import requests
import json

base_url = "http://127.0.0.1:8000"

def test_health():
    print("\n--- Testing Health Check ---")
    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(response.json())

def test_chat_flow():
    print("\n--- Testing SSE Chat Flow ---")
    payload1 = {
        "query": "I still haven't received my offer letter yet",
        "user_id": "CAND-103",
        "session_id": "session123",
        "stream": True,
    }
    print("Turn 1 Query:", payload1["query"])
    response1 = requests.post(f"{base_url}/api/chat", json=payload1, stream=True, timeout=120)
    print(f"Status Code: {response1.status_code}")
    text1 = _read_sse_response(response1)
    print("AI Response:", text1[:200] + "...")

    print("\n--- Testing JSON Chat Flow (frontend-style messages) ---")
    payload2 = {
        "messages": [
            {"role": "user", "content": "Who is my recruiter?"},
        ],
        "session_id": "session123",
        "user_id": "CAND-103",
    }
    response2 = requests.post(f"{base_url}/chat", json=payload2, timeout=120)
    print(f"Status Code: {response2.status_code}")
    res_json2 = response2.json()
    reply = res_json2.get("data") or res_json2.get("response", "")
    print("AI Response:", str(reply)[:300] + "...")


def _read_sse_response(response):
    import json
    tokens = []
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            payload = json.loads(line[6:])
            if payload.get("type") == "token":
                tokens.append(payload.get("content", ""))
    return "".join(tokens)

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
