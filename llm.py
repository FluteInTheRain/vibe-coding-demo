from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import openai


class OpenAIClient:
    """Thin wrapper around OpenAI's chat API for the lab.

    This client is intentionally minimal: it loads the OPENAI_API_KEY from the
    environment (via dotenv) during initialization and exposes a simple
    `chat` method that forwards messages and optional function metadata to the
    OpenAI API. No retries or advanced timeout handling are implemented for
    Lab 1 simplicity.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialize the client and load the API key from environment.

        Args:
            model: The OpenAI model id to use for chat requests.
        Raises:
            ValueError: if OPENAI_API_KEY is not set in the environment.
        """
        load_dotenv()
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        openai.api_key = api_key

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to OpenAI and return the raw response object.

        Args:
            messages: The conversation messages formatted for OpenAI (list of {"role":..., "content":...} or function call messages).
            tools: Optional list of function metadata (OpenAI function-calling schema).

        Returns:
            The raw response dict returned by the OpenAI Python client.
        """
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if tools is not None:
            kwargs["functions"] = tools

        resp = openai.ChatCompletion.create(**kwargs)
        # The openai library returns a model-specific object; convert to dict
        # for easier handling by the agent code and tests.
        return resp if isinstance(resp, dict) else resp.__dict__
