# MCP Document Parser + OCR

This project exposes a document understanding workflow as a [Model Context Protocol](https://github.com/modelcontextprotocol) (MCP) tool.  
It can download remote files, run OCR/table extraction, optionally invoke a Vision-Language Model (VLM), and return structured JSON for PDFs, images, DOCX, XLSX, and PPTX files.

## Key Features

- **Multi-format parsing** via PyMuPDF, pdfplumber, Camelot, python-docx, pandas/openpyxl, and python-pptx.
- **OCR fallback** using Tesseract with configurable languages.
- **Qwen3-VL-Plus integration** through the official `openai` Python SDK (OpenAI-compatible endpoint).
- **Configurable PDF parsing modes** that let you choose between OCR-centric extraction or VLM-driven Markdown conversion.
- **MCP tool** `parse_document_url` that accepts a single `options` dictionary while retaining the original flexibility.
- **Health endpoint** for simple runtime checks when hosted through FastMCP.

## Configuring Qwen3-VL-Plus via the OpenAI SDK

The VLM client uses the official `openai` package to talk to Aliyun's OpenAI-compatible endpoint.

1. Export the required environment variables (defaults shown below). The API key can be provided through either `OPENAI_API_KEY` or the legacy `DASHSCOPE_API_KEY` variable.

   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
   export QWEN_VLM_MODEL="qwen3-vl-plus"
   ```

2. Call the MCP tool with the desired parsing mode:

   ```json
   {
     "tool": "parse_document_url",
     "arguments": {
       "file_url": "https://example.com/sample.pdf",
       "options": {
         "pdf_parse_mode": "vlm",   // or "ocr"
         "use_ocr": true,
         "ocr_lang": "eng+chi_sim"
       }
     }
   }
   ```

   - `pdf_parse_mode="vlm"` converts every PDF page to an image, sends the batch to Qwen3-VL-Plus with a Markdown-oriented prompt, and returns the model's Markdown response alongside the rendered page images.
   - `pdf_parse_mode="ocr"` keeps the hybrid extractor, optionally running OCR on empty pages and using the VLM to transcribe individual pages when needed.
   - Image parsing always leverages the VLM with prompts tuned to describe scenes, tables, and formulas (rendered as LaTeX).

> **Note:** Outbound network access remains disabled in the evaluation environment; automated tests exercise the OpenAI code paths through stubs.

## Development Setup

```bash
uv sync  # or pip install -r requirements if preferred
```

Run unit tests with:

```bash
pytest -q
```

## Repository Structure

- `main.py` – core parsers, OCR helpers, VLM client, and MCP wiring.
- `tests/test_mcp.py` – dependency stubs plus unit tests covering the dispatch logic and parser behaviours.
- `example_files/` – sample assets for manual experiments.

## Extending / Debugging Tips

- The `VLMClient` wraps the OpenAI Responses API and returns both the Markdown text and the raw payload for auditing or caching.
- To enable verbose logging, wrap the MCP server (`FastMCP`) invocation with your preferred logger before deploying.
- When running OCR-heavy workloads, adjust the `options["timeout"]` value to accommodate slow downloads.

