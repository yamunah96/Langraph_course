import operator
import uuid
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2,
    max_tokens=1024,
)

# ─── System Prompts ────────────────────────────────────────────────────────────
research_agent_system_prompt = """You are a research agent. Gather numeric data for chart generation.

RULES:
- Use TavilySearch for external data.
- Always extract numeric values.
- Never refuse. Use estimates if exact values unavailable.
- Do NOT generate charts or code.

OUTPUT FORMAT (mandatory, repeat per data point):
Label: <name>
Value: <number>
Unit: <optional>"""

chart_generator_system_prompt = """You are a chart-generation agent.

Given structured Label/Value data, write one block of complete matplotlib Python code and call python_repl_tool with it.

The code must:
- Build a bar/line/pie chart from the data
- Label axes and add a title
- Save with plt.savefig('chart_output.png')
- print('Saved: chart_output.png')

Call python_repl_tool EXACTLY ONCE. After it returns, say only: FINAL ANSWER"""

# ─── Tools ─────────────────────────────────────────────────────────────────────
tavily_tool = TavilySearch(max_results=2)

@tool
def python_repl_tool(code: str) -> str:
    """Execute Python code to generate and save charts as PNG files."""
    stdout_buf = io.StringIO()
    try:
        with redirect_stdout(stdout_buf):
            exec(code, {})  # noqa: S102
        out = stdout_buf.getvalue()
        return out if out else "Executed successfully (no output)."
    except Exception as e:
        return f"Execution error: {repr(e)}"

# ─── Research agent (ReAct subgraph — stable, used for web search) ─────────────
def _make_research_agent():
    bound = llm.bind_tools([tavily_tool])

    def call_model(state: MessagesState) -> dict:
        msgs = [SystemMessage(content=research_agent_system_prompt)] + list(state["messages"])
        return {"messages": [bound.invoke(msgs)]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode([tavily_tool]))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    return g.compile()

research_agent = _make_research_agent()

# ─── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# ─── Nodes ─────────────────────────────────────────────────────────────────────
def research_node(state: AgentState, config: RunnableConfig) -> dict:
    msgs = list(state["messages"])
    result = research_agent.invoke({"messages": msgs}, {"recursion_limit": 15})
    return {"messages": result["messages"][len(msgs):], "sender": "research_agent"}


def chart_node(state: AgentState, config: RunnableConfig) -> dict:
    """Direct chart node: ask model for code → run tool once → done. No inner loop."""
    # Give chart agent only the human query + last research summary (no tool noise)
    human_msgs = [m for m in state["messages"] if getattr(m, "type", "") == "human"]
    research_summary = [
        m for m in state["messages"]
        if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", [])
    ]
    context = human_msgs + research_summary[-1:]

    chart_llm = llm.bind_tools([python_repl_tool])
    system = SystemMessage(content=chart_generator_system_prompt)

    # Step 1: ask model — it should produce a tool_call
    ai_resp: AIMessage = chart_llm.invoke([system] + context)
    new_msgs: list[BaseMessage] = [ai_resp]

    # Step 2: if model made a tool call, execute it exactly once
    if getattr(ai_resp, "tool_calls", []):
        for tc in ai_resp.tool_calls:
            tool_result = python_repl_tool.invoke(tc["args"])
            new_msgs.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )
        new_msgs.append(AIMessage(content="FINAL ANSWER"))
    else:
        # Model responded with text (no tool call) — treat as done
        new_msgs.append(AIMessage(content="FINAL ANSWER"))

    return {"messages": new_msgs, "sender": "chart_agent"}

# ─── Router ────────────────────────────────────────────────────────────────────
def router(state: AgentState) -> str:
    last = state["messages"][-1]
    if "FINAL ANSWER" in str(getattr(last, "content", "")):
        return END
    return "chart_node"

# ─── Memory ────────────────────────────────────────────────────────────────────
memory = MemorySaver()

# ─── Workflow ──────────────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("research_node", research_node)
workflow.add_node("chart_node", chart_node)

workflow.add_edge(START, "research_node")
workflow.add_edge("research_node", "chart_node")
workflow.add_conditional_edges("chart_node", router, {"chart_node": "chart_node", END: END})

graph = workflow.compile(checkpointer=memory)

# ─── Main / Final Output Check ─────────────────────────────────────────────────
if __name__ == "__main__":
    config = RunnableConfig(
        configurable={"thread_id": str(uuid.uuid4())},
        recursion_limit=50,
    )

    user_query = "Show me the top 5 programming languages by popularity in 2024 as a bar chart."

    print(f"\nQuery: {user_query}")
    print("=" * 60)

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_query)], "sender": "user"},
        config,
    )

    print("\n--- Conversation Trace ---")
    for msg in result["messages"]:
        role = getattr(msg, "type", "unknown").upper()
        content = str(getattr(msg, "content", ""))
        if content:
            print(f"\n[{role}]:\n{content[:600]}{'...' if len(content) > 600 else ''}")

    print("\n--- Final Output Check ---")
    last_content = str(getattr(result["messages"][-1], "content", ""))
    if "FINAL ANSWER" in last_content:
        print("\nSUCCESS: Chart generated. Check the saved PNG path printed above.")
    else:
        print("\nWARNING: Workflow ended without FINAL ANSWER.")
    print(f"Total messages in thread: {len(result['messages'])}")


#  lets convert this into streamlit application, we can choose the graph line,bar,advance (pie,stack), 
# additional agents
# u will get some datasets # dashboard agent