# MCP Document Parser + OCR

This project exposes a document understanding workflow as a [Model Context Protocol](https://github.com/modelcontextprotocol) (MCP) tool.  
It can download remote files, run OCR/table extraction, optionally invoke a Vision-Language Model (VLM), and return structured JSON for PDFs, images, DOCX, XLSX, and PPTX files.

## Key Features

- **Multi-format parsing** via PyMuPDF, pdfplumber, Camelot, python-docx, pandas/openpyxl, and python-pptx.
- **OCR fallback** using Tesseract with configurable languages.
- **Qwen3-VL-Plus integration** through the official `openai` Python SDK (OpenAI-compatible endpoint).
- **Asynchronous VLM scheduling** with an internal task manager that caps concurrent multimodal calls at five to maximise throughput without overwhelming upstream services.
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

   The asynchronous VLM task manager automatically batches requests and respects a hard concurrency ceiling of five active calls, so no additional tuning is required.

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

## Output Layout Fidelity

To help downstream consumers reconstruct documents faithfully, every parser now exposes an explicit ordering of textual, tabular, and visual content:

- **PDF pages** include an `elements` array listing recognised text, OCR results, tables, embedded images, and VLM Markdown in reading order.
- **DOCX files** add a `content.flow` array that interleaves paragraphs, tables, and extracted images with their corresponding indices and payloads.
- **Images** expose a `content_flow` list that tracks the raw image, optional OCR output, and VLM transcription.
- **PPTX slides** return an `elements` array per slide, preserving the sequence of text boxes, pictures, and speaker notes.
- **PDF VLM mode** surfaces `document_elements` that pair the page image set with the aggregated Markdown response.

These structures ensure each image, table, or text snippet can be reinserted at its original position when rendering the parsed output.

## Development Setup

```bash
uv sync  # or pip install -r requirements if preferred
```

Run unit tests with:

```bash
pytest -q
```

## Manual Testing with the Sample Report PDF

The repository includes a small client script (`test.py`) that exercises the `parse_document_url` MCP tool end-to-end. To reproduce the flow with the hosted sample report:

1. Start the MCP server in one terminal:

   ```bash
   uv run python main.py
   ```

2. In a second terminal, call the tool through the client:

   ```bash
   uv run python test.py
   ```

   The client is preconfigured to fetch `https://sample-files.com/downloads/documents/pdf/sample-report.pdf` and will print the structured response payload returned by the tool.

Example output (truncated for brevity):

```
Type: pdf
Keys: ['type', 'filename', 'elapsed_seconds', 'summary', 'markdown']
Elapsed seconds: 1.74
Summary snippet: # Parse Result for sample-report.pdf

- **Type:** PDF
- **Elapsed Seconds:** 1.75

### Warnings
- camelot failed: module 'camelot' has no attribute 'read_pdf'
```

The server log will emit warnings like `Cannot set gray stroke color because /'P6' is an invalid float value`; these originate from PyMuPDF while rendering embedded images and can be safely ignored during local testing.

## Repository Structure

- `main.py` – core parsers, OCR helpers, VLM client, and MCP wiring.
- `tests/test_mcp.py` – dependency stubs plus unit tests covering the dispatch logic and parser behaviours.
- `example_files/` – sample assets for manual experiments.

## Extending / Debugging Tips

- The `VLMClient` wraps the OpenAI Responses API and returns both the Markdown text and the raw payload for auditing or caching.
- To enable verbose logging, wrap the MCP server (`FastMCP`) invocation with your preferred logger before deploying.
- When running OCR-heavy workloads, adjust the `options["timeout"]` value to accommodate slow downloads.

