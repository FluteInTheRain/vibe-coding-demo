from __future__ import annotations

import sys

from llm import OpenAIClient
from agent import ReActAgent


def main() -> None:
    """CLI entrypoint: read question from argv and run the ReAct agent.

    Raises:
        ValueError: if no question argument is provided.
    """
    if len(sys.argv) < 2:
        raise ValueError("Missing question argument; usage: python main.py \"your question\"")

    question = sys.argv[1]

    client = OpenAIClient()
    agent = ReActAgent(client)

    result = agent.run(question)
    print("Final Result:", result)


if __name__ == "__main__":
    main()
