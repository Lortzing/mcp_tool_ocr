# MCP Document Parser + OCR

This project exposes a document understanding workflow as a [Model Context Protocol](https://github.com/modelcontextprotocol) (MCP) tool.  
It can download remote files, run OCR/table extraction, optionally invoke a Vision-Language Model (VLM), and return structured JSON for PDFs, images, DOCX, XLSX, and PPTX files.

## Key Features

- **Multi-format parsing** via PyMuPDF, pdfplumber, Camelot, python-docx, pandas/openpyxl, and python-pptx.
- **OCR fallback** using Tesseract with configurable languages.
- **DashScope Qwen2.5-VL integration** through a dedicated VLM client helper.
- **MCP tool** `parse_document_url` that accepts a single `options` dictionary while retaining the original flexibility.
- **Health endpoint** for simple runtime checks when hosted through FastMCP.

## Using the DashScope Qwen2.5-VL backend

1. Obtain a DashScope API key (the repository defaults to `qwen2.5-vl`).
2. When constructing the MCP request, pass the VLM settings inside `options`:

   ```json
   {
     "tool": "parse_document_url",
     "arguments": {
       "file_url": "https://example.com/sample.pdf",
       "options": {
         "use_ocr": true,
         "run_vlm": true,
         "vlm_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
         "vlm_token": "sk-...",
         "vlm_model": "qwen2.5-vl"
       }
     }
   }
   ```

3. The tool automatically normalises DashScope URLs and prepares the OpenAI-compatible payload (text instructions + base64 images) before calling the API.

> **Note:** Network egress is blocked in this execution environment.  A manual attempt to call the Qwen endpoint therefore fails with `Tunnel connection failed: 403` (see the testing section).  The code path is exercised through automated tests with HTTP stubs.

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

- The `VLMClient` automatically falls back to a generic JSON POST for custom providers when the endpoint is not DashScope.
- To enable verbose logging, wrap the MCP server (`FastMCP`) invocation with your preferred logger before deploying.
- When running OCR-heavy workloads, adjust the `options["timeout"]` value to accommodate slow downloads.

