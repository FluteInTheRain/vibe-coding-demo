from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from llm import OpenAIClient
from agent import ReActAgent

mcp = FastMCP(name="mini-research-agent")


def ask_agent(question: str) -> str:
    """Run a ReAct agent that can answer multi-step questions using
    calculator, current time, and web search tools. Returns final
    answer as a string.

    The function initializes an OpenAIClient and a ReActAgent from the
    repository, runs the agent on the provided question, and returns the
    final answer text. Errors are caught and returned as a friendly
    string starting with "Error: ".
    """
    try:
        client = OpenAIClient()
        agent = ReActAgent(client)
        return agent.run(question)
    except Exception as e:  # pragma: no cover - runtime error handling
        return f"Error: {e}"


# Register the tool with FastMCP using whichever registration API is available.
# We avoid assuming a single decorator shape to keep module import safe.
try:
    if hasattr(mcp, "tool") and callable(getattr(mcp, "tool")):
        # mcp.tool() likely returns a decorator
        mcp.tool()(ask_agent)  # type: ignore[arg-type]
    elif hasattr(mcp, "register_tool") and callable(getattr(mcp, "register_tool")):
        mcp.register_tool("ask_agent", ask_agent)  # type: ignore[arg-type]
    elif hasattr(mcp, "add_tool") and callable(getattr(mcp, "add_tool")):
        mcp.add_tool("ask_agent", ask_agent)  # type: ignore[arg-type]
    else:
        # Best-effort: attach to a tools dict if present, or set an attribute
        if hasattr(mcp, "tools") and isinstance(getattr(mcp, "tools"), dict):
            mcp.tools["ask_agent"] = ask_agent  # type: ignore[index]
        else:
            setattr(mcp, "ask_agent", ask_agent)
except Exception:
    # Registration failure should not crash module import; leave ask_agent callable.
    pass


if __name__ == "__main__":
    # Run the FastMCP stdio server (default transport)
    mcp.serve()
