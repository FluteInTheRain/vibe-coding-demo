from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _format_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _eval_ast(node: ast.AST) -> float:
    """Evaluate a restricted AST node supporting only safe numeric operations.

    Allowed node types: Expression, BinOp, UnaryOp, Constant.
    Allowed operators: Add, Sub, Mult, Div, Pow, USub, FloorDiv, Mod.

    Raises:
        ValueError: if an unsafe node or operator is encountered or evaluation fails.
    """

    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant):
        if _is_number(node.value):
            return node.value  # type: ignore[return-value]
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return operator.add(left, right)
        if isinstance(op, ast.Sub):
            return operator.sub(left, right)
        if isinstance(op, ast.Mult):
            return operator.mul(left, right)
        if isinstance(op, ast.Div):
            return operator.truediv(left, right)
        if isinstance(op, ast.FloorDiv):
            return operator.floordiv(left, right)
        if isinstance(op, ast.Mod):
            return operator.mod(left, right)
        if isinstance(op, ast.Pow):
            return operator.pow(left, right)
        raise ValueError(f"Unsupported binary operator: {type(op).__name__}")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_ast(node.operand)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    # Any other node types are explicitly disallowed
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculator(expression: str) -> Dict[str, Optional[str]]:
    """Evaluate a numeric expression using a safe AST whitelist.

    The expression is parsed with ast.parse(..., mode='eval') and only the
    following nodes/operators are permitted: Expression, BinOp, UnaryOp,
    Constant, Add, Sub, Mult, Div, Pow, USub, FloorDiv, Mod. Any use of Name,
    Call, Attribute, or other nodes will result in an error.

    Returns an envelope: {"result": str, "error": Optional[str]}.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except Exception as e:
        return {"result": "", "error": f"Parse error: {e!s}"}

    # Walk the parsed AST to ensure no disallowed nodes are present.
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.FloorDiv, ast.Mod)):
            continue
        # Allowed nodes are Expression, BinOp, UnaryOp, Constant and operator classes
        # Reject all others explicitly
        return {"result": "", "error": f"Disallowed node in expression: {type(node).__name__}"}

    try:
        value = _eval_ast(parsed)
    except Exception as e:
        return {"result": "", "error": f"Evaluation error: {e!s}"}

    if not _is_number(value):
        return {"result": "", "error": "Result is not a number"}

    return {"result": _format_number(value), "error": None}


def get_current_year() -> Dict[str, Optional[str]]:
    """Return the fixed current year for the lab environment.

    Returns an envelope: {"result": str, "error": Optional[str]}.
    """
    return {"result": "2026", "error": None}


def web_search(query: str) -> Dict[str, Optional[str]]:
    """Mock web search used for lab exercises.

    This returns a deterministic fake result so tests and examples are stable.
    Returns an envelope: {"result": str, "error": Optional[str]}.
    """
    return {"result": f"MOCK_SEARCH_RESULT for {query}", "error": None}


def final_answer(answer: str) -> Dict[str, Optional[str]]:
    """Wrap the final answer in the standard envelope.

    This function represents the tool the agent will call when it wants to
    finish. Returns an envelope: {"result": str, "error": Optional[str]}.
    """
    return {"result": answer, "error": None}


def build_tools() -> Tuple[Dict[str, Callable[..., Dict[str, Optional[str]]]], List[Dict[str, Any]]]:
    """Return a tuple (tools_map, functions_metadata) for function-calling.

    tools_map: mapping from function name to callable
    functions_metadata: list of OpenAI-style function descriptions (name, description, parameters)
    """
    tools_map: Dict[str, Callable[..., Dict[str, Optional[str]]]] = {
        "calculator": calculator,
        "get_current_year": get_current_year,
        "web_search": web_search,
        "final_answer": final_answer,
    }

    functions_metadata: List[Dict[str, Any]] = [
        {
            "name": "calculator",
            "description": "Evaluate a numeric expression using only safe operators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
        {
            "name": "get_current_year",
            "description": "Return the current year for the lab environment.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "web_search",
            "description": "Mock web search returning a deterministic string.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
        {
            "name": "final_answer",
            "description": "Return the agent's final answer (signals completion).",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "Final answer text"}},
                "required": ["answer"],
            },
        },
    ]

    return tools_map, functions_metadata


__all__ = ["calculator", "get_current_year", "web_search", "final_answer", "build_tools"]
