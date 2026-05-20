# ReAct Agent convention dự án

## Stack
- Python 3.12, OpenAI SDK (>=1.0), python-dotenv, pytest
- Không dùng async; tất cả I/O đồng bộ cho dễ đọc.

## Coding style
- Type hints trên mọi public function.
- Docstring (Google style) cho function/class public.
- Mỗi tool function trả về envelope: {"result": str, "error": str | None}.
- Không print() trong logic; dùng logging hoặc trả về string.

## Architecture decisions
- 4 module: main.py, agent.py, tools.py, llm.py.
- ReAct loop có MAX_ITERATIONS = 10 (giới hạn cứng).
- LLM client (llm.py) bao bọc OpenAI để dễ thay backend sau (Anthropic, Mistral...).

## Test
- pytest tests/ cho mọi module có logic.
- Mock OpenAI/Tavily ở tests/, không gọi network thật.

## Plan-first policy
- Mọi tính năng >1 file -> BẮT BUỘC Plan Mode trước.
