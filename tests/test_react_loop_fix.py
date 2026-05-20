from typing import Any, Dict, List

import pytest

from agent import ReActAgent


class DummyClient:
    def __init__(self):
        self.step = 0

    def chat(self, messages: List[Dict[str, Any]], tools=None) -> Dict[str, Any]:
        # On first call, model issues a function_call 'calculator' with an id
        self.step += 1
        if self.step == 1:
            return {
                "choices": [
                    {
                        "finish_reason": None,
                        "id": "c1",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "function_call": {"name": "calculator", "arguments": '{"expression": "2026 * 3"}', "id": "fc-1"},
                        },
                    }
                ]
            }
        # On second call, after execute_tool was called, the client should receive the tool
        # observation embedded in messages; we simulate the model seeing it and returning stop
        else:
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "The answer is 6078"}}]}

    def execute_tool(self, messages: List[Dict[str, Any]], tools, tool_call_id: str, tool_name: str, observation_text: str) -> Dict[str, Any]:
        # ensure the messages include a tool role message we added
        has_tool_msg = any(m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id for m in messages)
        assert has_tool_msg, "execute_tool should be passed messages that include the tool observation"
        # simulate the model's followup after seeing the tool observation
        return {"choices": [{"finish_reason": None, "message": {"role": "assistant", "content": "", }}]}


def test_react_loop_stops_after_tool_observation():
    client = DummyClient()
    agent = ReActAgent(client)
    res = agent.run("Compute 2026 * 3 for me")
    assert "6078" in res
