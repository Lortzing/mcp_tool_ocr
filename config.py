"""Configuration helpers for the MCP document parser."""

import os


# The Qwen multimodal models are exposed through an OpenAI-compatible API. The
# default values mirror the DashScope endpoint so the tool works out of the box
# for users that export the standard `DASHSCOPE_API_KEY` variable. All values
# can be overridden via environment variables in case a self-hosted gateway is
# used.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
QWEN_VLM_MODEL = os.getenv("QWEN_VLM_MODEL", "qwen3-vl-plus")
