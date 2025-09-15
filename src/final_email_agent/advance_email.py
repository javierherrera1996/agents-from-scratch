from typing import Literal, List, TypedDict, Dict, Any, Optional
import uuid, requests

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import AGENT_TOOLS_PROMPT
from email_assistant.prompts import (
    triage_system_prompt, triage_user_prompt, agent_system_prompt,
    default_background, default_triage_instructions,
    default_response_preferences, default_cal_preferences
)
from email_assistant.schemas import State, RouterSchema  # quitamos StateInput
from email_assistant.utils import parse_email, format_email_markdown

load_dotenv(".env")

# --- Tools & LLMs
tools = get_tools()
tools_by_name = get_tools_by_name(tools)

llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema)
llm_with_tools = llm.bind_tools(tools, tool_choice="any")

# --- Nodos del agente "response_agent"
def llm_call(state: State):
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {
                        "role": "system",
                        "content": agent_system_prompt.format(
                            tools_prompt=AGENT_TOOLS_PROMPT,
                            background=default_background,
                            response_preferences=default_response_preferences,
                            cal_preferences=default_cal_preferences,
                        ),
                    },
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: State):
    out_msgs = []
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        for tc in last.tool_calls:
            tool = tools_by_name[tc["name"]]
            observation = tool.invoke(tc["args"])
            out_msgs.append({"role": "tool", "content": observation, "tool_call_id": tc["id"]})
    return {"messages": out_msgs}

def should_continue(state: State) -> Literal["Action", "__end__"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        for tc in last.tool_calls:
            if tc["name"] == "Done":
                return "__end__"
        return "Action"
    return "__end__"

# --- response_agent subgrafo
response_builder = StateGraph(State)
response_builder.add_node("llm_call", llm_call)
response_builder.add_node("environment", tool_node)
response_builder.add_edge(START, "llm_call")
response_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"Action": "environment", "__end__": END},
)
response_builder.add_edge("environment", "llm_call")
response_agent = response_builder.compile()

# --- A2A RAG externo (usa state["question"])
A2A_URL = "https://potential-palm-tree-4r7vvrv49w6fjgvr-8000.app.github.dev"

from pydantic import BaseModel, Field
from langchain.agents import StructuredTool

class A2AInput(BaseModel):
    question: str = Field(..., description="Pregunta para el agente RAG externo")

def call_external_agent(question: str) -> str:
    payload = {
        "invocationId": str(uuid.uuid4()),
        "metadata": {"caller": "langgraph-tool"},
        "input": {"question": question},
    }
    r = requests.post(f"{A2A_URL}/a2a/run", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["outputs"][0]["content"] if data.get("outputs") else ""

external_rag_tool = StructuredTool.from_function(
    func=call_external_agent,
    name="external_rag_tool",
    description="Llama a un agente RAG externo vía A2A",
    args_schema=A2AInput,
)

def node_use_tool(state: State):
    ans = external_rag_tool.invoke({"question": state["question"]})
    return {"answer": ans}

# --- TRIAGE: solo actualiza estado; NO usa Command
def triage_router(state: State):
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions,
    )
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    classification = result.classification  # "respond" | "ignore" | "notify"

    update: Dict[str, Any] = {"classification_decision": classification}

    if classification == "respond":
        # Prepara el estado para downstream:
        update["messages"] = [
            {"role": "user", "content": f"Respond to the email: {email_markdown}"}
        ]
        # clave: setear question para el RAG externo
        update["question"] = f"Draft a concise, polite reply for this email:\n{email_markdown}"
    # (notify/ignore no necesitan más)
    return update

# --- Router post-triage (para edges condicionales)
def route_after_triage(state: State) -> Literal["rag_agent", "__end__"]:
    return "rag_agent" if state.get("classification_decision") == "respond" else "__end__"

# --- Grafo global
overall = StateGraph(State)
overall.add_node("triage_router", triage_router)
overall.add_node("rag_agent", node_use_tool)
overall.add_node("response_agent", response_agent)

overall.add_edge(START, "triage_router")
overall.add_conditional_edges(
    "triage_router",
    route_after_triage,
    {"rag_agent": "rag_agent", "__end__": END},
)
overall.add_edge("rag_agent", "response_agent")
overall.add_edge("response_agent", END)

email_assistant_advance = overall.compile()
