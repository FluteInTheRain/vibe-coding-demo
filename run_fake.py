"""Run the ReAct agent with a local FakeClient for testing (no network/API key required).

Usage:
    python run_fake.py "1 + 1 = ?"
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from agent import ReActAgent
from tools import build_tools


class FakeClient:
    def chat(self, messages: List[Dict[str, Any]], tools=None):
        # Very small rule-based fake: if user asks math, request calculator; if asks year, request get_current_year; if asks capital, request web_search
        user = next((m['content'] for m in messages if m.get('role') == 'user'), '')
        tool_count = sum(1 for m in messages if m.get('role') == 'tool')

        def resp_function(name: str, args: Dict[str, Any], id_: str = None):
            # return a choice with function_call; arguments as JSON string
            return {
                'choices': [{
                    'id': id_ or 'ch-1',
                    'message': {'role': 'assistant', 'function_call': {'id': id_ or 'ch-1-fc', 'name': name, 'arguments': json.dumps(args)}},
                    'finish_reason': None
                }]
            }

        def resp_content(content: str, id_: str = None):
            return {'choices': [{'id': id_ or 'ch-final', 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}]}

        if any(tok in user for tok in ['1 + 1', '1+1', '1 + 1 =']):
            if tool_count == 0:
                return resp_function('calculator', {'expression': '1 + 1'}, id_='calc-1')
            else:
                # after tool observation, return final
                return resp_content('2', id_='final-1')

        if 'Năm nay' in user or 'năm nay' in user:
            if tool_count == 0:
                return resp_function('get_current_year', {}, id_='year-1')
            elif tool_count == 1:
                # after year, request calc
                last_tool = [m for m in messages if m.get('role') == 'tool'][-1]
                obs = json.loads(last_tool['content'])
                year = obs.get('result')
                return resp_function('calculator', {'expression': f"{year} * 3"}, id_='calc-2')
            else:
                return resp_content('Done', id_='final-2')

        if 'Thủ đô' in user or 'thủ đô' in user:
            if tool_count == 0:
                return resp_function('web_search', {'query': user}, id_='web-1')
            else:
                return resp_content('Hà Nội', id_='final-3')

        return resp_content("I don't know", id_='final-0')


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python run_fake.py "your question"')
        sys.exit(2)

    question = sys.argv[1]
    fake = FakeClient()

    agent = ReActAgent.__new__(ReActAgent)
    agent.client = fake
    agent.tools_map, agent.tool_schemas = build_tools()

    result = agent.run(question)
    print('Final Result:', result)


if __name__ == '__main__':
    main()
