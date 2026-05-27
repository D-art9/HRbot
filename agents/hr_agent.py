import os
import sys
import asyncio
import json

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.onboarding_tools import get_onboarding_status, get_pending_documents
from tools.recruitment_tools import (
    get_candidate_status,
    get_interview_schedule,
    get_offer_status,
    get_recruiter_contact,
)
from tools.policy_tools import policy_rag_tool

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
    from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.llm_provider import llm
from core.logger import logger

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="input",
    output_key="output"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are SVYIA, a professional HR Assistant AI agent. You help candidates and employees. "
               "You have access to tools for checking recruitment status, interview schedules, offer status, recruiter contact details, onboarding progress, and pending onboarding documents. You also have a policy retrieval tool for answering company policy queries. "
               "For candidate/employee queries, you MUST use the provided candidate_id/employee_id parameter or ask them for it if not available. "
               "Current Candidate/Employee ID in context: {user_id}."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [
    get_candidate_status,
    get_interview_schedule,
    get_offer_status,
    get_recruiter_contact,
    get_onboarding_status,
    get_pending_documents,
    policy_rag_tool,
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define agent
agent = create_tool_calling_agent(llm_with_tools, tools, prompt)

def get_agent_executor(session_id=None):
    global memory
    current_memory = memory
    if session_id:
        from services.memory_service import memory_service
        current_memory = memory_service.get_memory(session_id)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=current_memory,
        verbose=True,
        return_intermediate_steps=True
    )

def route_hr_query(query, user_id=None, session_id=None):
    """
    Routes HR queries to appropriate workflow tools using LangChain Tool-calling Agent.
    """
    executor = get_agent_executor(session_id)
    res = executor.invoke({
        "input": query,
        "user_id": user_id or "Not Provided"
    })
    return res["output"]


async def stream_hr_query(query, user_id=None, session_id=None):
    """
    Async generator that yields SSE structured payloads for the React frontend.
    """
    try:
        logger.info(f"STREAM: Starting agent execution for query: {query[:100]}...")
        
        # Yield initial metadata stages
        yield f"data: {json.dumps({'type': 'metadata', 'workflow_stage': 'intent_classification'})}\n\n"
        await asyncio.sleep(0.01)
        
        yield f"data: {json.dumps({'type': 'metadata', 'workflow_stage': 'routing'})}\n\n"
        await asyncio.sleep(0.01)
        
        executor = get_agent_executor(session_id)
        
        # Run execution asynchronously
        res = await asyncio.wait_for(
            asyncio.to_thread(
                executor.invoke,
                {"input": query, "user_id": user_id or "Not Provided"}
            ),
            timeout=60.0
        )
        
        response = res["output"]
        
        # Find active tool from intermediate steps
        active_tool = "policy_rag_tool"
        rag_retrieval_state = "success"
        if "intermediate_steps" in res and res["intermediate_steps"]:
            active_tool = res["intermediate_steps"][0][0].tool
            rag_retrieval_state = "none" if active_tool != "policy_rag_tool" else "success"
            
        yield f"data: {json.dumps({
            'type': 'metadata',
            'active_tool': active_tool,
            'rag_state': rag_retrieval_state,
            'confidence': 0.95,
            'workflow_stage': 'generation'
        })}\n\n"
        await asyncio.sleep(0.01)
        
        words = response.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            await asyncio.sleep(0.01)
            
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        logger.error(f"STREAM Error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"





if __name__ == "__main__":

    print("\n===== SVYIA HR AI AGENT =====\n")

    user_id = input("Enter Employee/Candidate ID: ")

    while True:

        query = input("\nYou: ")

        if query.lower() in ["exit", "quit"]:

            print("\nEnding HR session...\n")

            break

        response = route_hr_query(
            query,
            user_id
        )

        print(f"\nAI: {response}")
