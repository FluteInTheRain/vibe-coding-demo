from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import openai


class OpenAIClient:
    """Thin wrapper around OpenAI's chat API for the lab.

    This client is intentionally minimal: it loads the OPENAI_API_KEY from the
    environment (via dotenv) during initialization and exposes a simple
    `chat` method that forwards messages and optional function metadata to the
    OpenAI API. The client also normalizes the response shape so callers can
    rely on function_call ids and string-typed arguments.
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

    def _ensure_function_call_shape(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize choices so each function_call has an id and string arguments.

        This protects the agent from variations in provider SDKs that omit
        function_call.id or return arguments as an object. We prefer the
        provider-supplied id when present; otherwise we derive a deterministic
        id from the choice index (not a random UUID).
        """
        choices = resp.get("choices") or []
        for idx, choice in enumerate(choices):
            msg = choice.get("message") or {}
            fc = msg.get("function_call")
            if fc is None:
                continue
            # Ensure the function_call has an id; prefer any provider id, else use deterministic one
            if not fc.get("id"):
                # Try choice-level id first
                if choice.get("id"):
                    fc["id"] = choice["id"]
                else:
                    fc["id"] = f"choice-{idx}-fc"
            # Ensure arguments are a JSON string
            args = fc.get("arguments")
            if args is None:
                fc["arguments"] = "{}"
            elif not isinstance(args, str):
                try:
                    fc["arguments"] = json.dumps(args)
                except Exception:
                    fc["arguments"] = "{}"
            # write back normalized function_call
            msg["function_call"] = fc
            choice["message"] = msg
            choices[idx] = choice
        resp["choices"] = choices
        return resp

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to OpenAI and return the normalized response object.

        Args:
            messages: The conversation messages formatted for OpenAI (list of {"role":..., "content":...} or function call messages).
            tools: Optional list of function metadata (OpenAI function-calling schema).

        Returns:
            A normalized response dict where choices[*].message.function_call has
            an 'id' and 'arguments' is a JSON string.
        """
        # Prepare messages to send to the new OpenAI API.
        # - Keep 'tool' role messages as-is (the new API accepts them when 'tools' is used).
        # - Strip any function_call/tool_calls metadata from assistant messages so the
        #   server does not expect tool responses within the same request.
        send_messages = []
        for m in messages:
            role = m.get('role')
            if role == 'assistant':
                # copy only role and content, drop function_call/tool_calls to avoid API validation errors
                # ensure content is always a string (None -> "")
                send_messages.append({'role': 'assistant', 'content': (m.get('content') or '')})
            elif role == 'tool':
                # Skip tool-role messages in plain chat(); tool responses must be
                # supplied via execute_tool's tool_responses in the same request.
                continue
            else:
                send_messages.append(m)

        kwargs: Dict[str, Any] = {"model": self.model, "messages": send_messages}

        # Normalize tools/functions payload for the new OpenAI client.
        # The new API expects 'tools' entries with a 'type' and a 'function' object.
        if tools is not None:
            has_tool_role = any((m.get('role') == 'tool') for m in messages)
            if has_tool_role:
                sanitized_tools = []
                for t in tools:
                    name = t.get('name')
                    params = t.get('parameters', {})
                    sanitized = {
                        'type': 'function',
                        'description': t.get('description', ''),
                        'function': {
                            'name': name,
                            'parameters': params,
                        }
                    }
                    sanitized_tools.append(sanitized)
                kwargs['tools'] = sanitized_tools
            else:
                kwargs['functions'] = tools

        # Use the new OpenAI client API (openai>=1.0). Do not call legacy ChatCompletion.
        OpenAI = getattr(openai, 'OpenAI', None)
        if OpenAI is None:
            raise RuntimeError('openai.OpenAI client not available; install openai>=1.0')

        client = OpenAI()
        try:
            resp_obj = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f'OpenAI chat call failed: {e}')

        # Convert SDK-specific response objects to a plain dict
        resp_dict = None
        if hasattr(resp_obj, 'model_dump'):
            try:
                resp_dict = resp_obj.model_dump()
            except Exception:
                resp_dict = None
        if resp_dict is None and hasattr(resp_obj, 'to_dict'):
            try:
                resp_dict = resp_obj.to_dict()
            except Exception:
                resp_dict = None
        if resp_dict is None:
            try:
                resp_dict = json.loads(json.dumps(resp_obj, default=lambda o: getattr(o, '__dict__', str(o))))
            except Exception:
                resp_dict = resp_obj if isinstance(resp_obj, dict) else getattr(resp_obj, '__dict__', {})

        return self._ensure_function_call_shape(resp_dict)

    def execute_tool(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], tool_call_id: str, tool_name: str, observation_text: str) -> Dict[str, Any]:
        """Send a follow-up request that includes the tool observation tied to a tool_call_id.

        The openai v2 API requires that when an assistant issues a tool call, the
        client include the corresponding tool response messages in the next
        request using tool_calls/ tool_responses fields. This helper constructs
        that request and returns the normalized model response.
        """
        OpenAI = getattr(openai, 'OpenAI', None)
        if OpenAI is None:
            raise RuntimeError('openai.OpenAI client not available; install openai>=1.0')

        client = OpenAI()
        # Build tool_responses structure expected by the new API
        tool_responses = [{
            'tool_call_id': tool_call_id,
            'content': observation_text,
        }]

        # Construct messages for the execute_tool call:
        # - Remove any existing tool-role messages (they will be supplied via tool_responses)
        # - Append an assistant message that declares the tool_call via 'tool_calls'
        messages_for_exec: List[Dict[str, Any]] = []
        for m in messages:
            if m.get('role') == 'tool':
                continue
            # copy assistant/user/system messages but ensure content is a string
            if m.get('role') == 'assistant':
                messages_for_exec.append({'role': 'assistant', 'content': (m.get('content') or '')})
            else:
                messages_for_exec.append(m)

        # Append the assistant message that contains the tool_calls list referencing the call id
        messages_for_exec.append({'role': 'assistant', 'tool_calls': [{'tool_call_id': tool_call_id}]})

        # Immediately follow with the tool message responding to that call id (required by API)
        messages_for_exec.append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': observation_text})

        # The new client expects the tool response as a message following the assistant's tool_calls
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages_for_exec, "tools": tools}

        try:
            resp_obj = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f'OpenAI execute_tool call failed: {e}')

        # normalize
        if hasattr(resp_obj, 'model_dump'):
            return self._ensure_function_call_shape(resp_obj.model_dump())
        if hasattr(resp_obj, 'to_dict'):
            return self._ensure_function_call_shape(resp_obj.to_dict())
        try:
            resp_dict = json.loads(json.dumps(resp_obj, default=lambda o: getattr(o, '__dict__', str(o))))
        except Exception:
            resp_dict = resp_obj if isinstance(resp_obj, dict) else getattr(resp_obj, '__dict__', {})
        return self._ensure_function_call_shape(resp_dict)
