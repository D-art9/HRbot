import os
import sys
import asyncio
import json
from typing import Annotated, TypedDict

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

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from core.llm_provider import llm
from core.logger import logger

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


# Define state schema for LangGraph agent
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


# Define the agent model invoking node
def call_model(state: AgentState):
    messages = state["messages"]
    user_id = state.get("user_id", "Not Provided")
    
    sys_msg = SystemMessage(content=(
        "You are SVYIA, a professional HR Assistant AI agent. You help candidates and employees. "
        "You have access to tools for checking recruitment status, interview schedules, offer status, recruiter contact details, onboarding progress, and pending onboarding documents. You also have a policy retrieval tool for answering company policy queries. "
        "For candidate/employee queries, you MUST use the provided candidate_id/employee_id parameter or ask them for it if not available. "
        f"Current Candidate/Employee ID in context: {user_id}."
    ))
    
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": [response]}


# Build the state graph
workflow = StateGraph(AgentState)

# Add node definitions
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Establish flow edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Compile with checkpoint memory saver
memory_saver = MemorySaver()
app_graph = workflow.compile(checkpointer=memory_saver)


def route_hr_query(query, user_id=None, session_id=None):
    """
    Routes HR queries to appropriate workflow tools using LangGraph Tool-calling Agent.
    """
    config = {"configurable": {"thread_id": session_id or "default_session"}}
    state_input = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id or "Not Provided"
    }
    res = app_graph.invoke(state_input, config=config)
    return res["messages"][-1].content


async def stream_hr_query(query, user_id=None, session_id=None):
    """
    Async generator that yields SSE structured payloads for the React frontend.
    """
    try:
        logger.info(f"STREAM: Starting LangGraph execution for query: {query[:100]}...")
        
        # Yield initial metadata stages
        yield f"data: {json.dumps({'type': 'metadata', 'workflow_stage': 'intent_classification'})}\n\n"
        await asyncio.sleep(0.01)
        
        yield f"data: {json.dumps({'type': 'metadata', 'workflow_stage': 'routing'})}\n\n"
        await asyncio.sleep(0.01)
        
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        state_input = {
            "messages": [HumanMessage(content=query)],
            "user_id": user_id or "Not Provided"
        }
        
        active_tool = "policy_rag_tool"
        rag_retrieval_state = "success"
        response_content = ""
        
        # Stream model events
        async for chunk in app_graph.astream(
            state_input,
            config=config,
            stream_mode="updates"
        ):
            if "tools" in chunk:
                tool_messages = chunk["tools"]["messages"]
                if tool_messages:
                    active_tool = tool_messages[0].name
                    rag_retrieval_state = "none" if active_tool != "policy_rag_tool" else "success"
            
            if "agent" in chunk:
                agent_message = chunk["agent"]["messages"][-1]
                response_content = agent_message.content

        yield f"data: {json.dumps({
            'type': 'metadata',
            'active_tool': active_tool,
            'rag_state': rag_retrieval_state,
            'confidence': 0.95,
            'workflow_stage': 'generation'
        })}\n\n"
        await asyncio.sleep(0.01)
        
        words = response_content.split(" ")
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
