# document_parser_refactored.py
from __future__ import annotations

import io
import os
import time
import json
import base64
import traceback
import urllib.parse
from typing import Optional, Dict, Any, List

import fitz
import pdfplumber
import camelot
import pytesseract
from pytesseract import Output as TESSERACT_OUTPUT
from PIL import Image
import docx
import pandas as pd
import openpyxl
from pptx import Presentation
import httpx
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QWEN_VLM_MODEL,
)

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

def _b64(bytestr: bytes) -> str:
    return base64.b64encode(bytestr).decode("utf-8")


class VLMClient:
    """Minimal wrapper around the OpenAI client for multimodal interactions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or ""
        self.base_url = base_url or ""
        self.model = model or "qwen3-vl-plus"
        self.timeout = timeout
        self._client: OpenAI | None = None
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

    def _ensure_client(self) -> bool:
        if self._client is None:
            return False
        return True

    def _safe_dump(self, response: Any) -> Any:
        if response is None:
            return None
        dump = getattr(response, "model_dump", None)
        if callable(dump):
            try:
                return dump()
            except Exception:
                pass
        json_fn = getattr(response, "json", None)
        if callable(json_fn):
            try:
                return json.loads(json_fn())
            except Exception:
                pass
        return getattr(response, "__dict__", str(response))

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ""
        output = getattr(response, "output", None)
        if isinstance(output, list):
            collected: List[str] = []
            for item in output:
                contents = getattr(item, "content", None)
                if isinstance(contents, list):
                    for content in contents:
                        text = getattr(content, "text", None) or getattr(content, "value", None)
                        if isinstance(text, str):
                            collected.append(text)
            if collected:
                return "".join(collected)

        choices = getattr(response, "choices", None)
        if isinstance(choices, list):
            texts = []
            for choice in choices:
                message = getattr(choice, "message", None)
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        texts.append(content)
                elif hasattr(message, "content"):
                    content = getattr(message, "content")
                    if isinstance(content, str):
                        texts.append(content)
            if texts:
                return "".join(texts)
        return ""

    def generate(self, prompt: str, images_b64: List[str]) -> Dict[str, Any]:
        if not self._ensure_client():
            return {"error": "OpenAI client not configured"}

        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for img in images_b64:
            content.append({"type": "input_image", "image": {"base64": img}})

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
            )
            return {
                "markdown": self._extract_text(response),
                "raw": self._safe_dump(response),
            }
        except Exception as exc:
            return {"error": f"OpenAI call failed: {exc}"}


class OCRHelper:
    @staticmethod
    def _normalize_lang(lang: str | List[str]) -> str:
        """
        规范化语言参数：
        - 如果传入 list，例如 ["eng", "chi_sim"]，则转成 "eng+chi_sim"
        - 如果传入 str，原样返回
        """
        if isinstance(lang, (list, tuple)):
            return "+".join(lang)
        return lang

    @staticmethod
    def image_to_data(image_bytes: bytes, lang: str | List[str] = "eng") -> Dict[str, Any]:
        """
        OCR 识别图像，支持多语言。
        e.g. lang="eng+chi_sim" 或 ["eng", "chi_sim"]
        """
        lang_str = OCRHelper._normalize_lang(lang)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            data: dict = pytesseract.image_to_data(img, lang=lang_str)
        except pytesseract.TesseractError as e:
            return {"error": f"OCR failed with lang={lang_str}: {str(e)}"}

        out = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = str(data.get("text", [])[i] or "").strip()
            if not txt:
                continue
            conf_raw = data.get("conf", [])[i]
            try:
                conf = float(conf_raw) if conf_raw != "-1" else None
            except Exception:
                conf = None
            bbox = {
                "left": int(data.get("left", [])[i]),
                "top": int(data.get("top", [])[i]),
                "width": int(data.get("width", [])[i]),
                "height": int(data.get("height", [])[i]),
            }
            out.append({"text": txt, "conf": conf, "bbox": bbox})
        return {"blocks": out, "lang": lang_str}



# -------------------------
# Parsers (clean interface: parse(file_bytes: bytes))
# -------------------------

class BaseParser:
    def __init__(
        self,
        use_ocr: bool = False,
        vlm_client: Optional[VLMClient] = None,
        ocr_lang: str | List[str] = "eng",
    ):
        # OCR is disabled by default. We keep the flag for backwards compatibility
        # but avoid invoking OCR logic in new flows unless explicitly requested.
        self.use_ocr = use_ocr
        self.vlm_client = vlm_client
        self.ocr_lang = ocr_lang

    def parse(self, data: bytes) -> Dict[str, Any]:
        raise NotImplementedError


class PDFParser(BaseParser):
    def __init__(
        self,
        *args,
        camelot_enable: bool = True,
        pdfplumber_enable: bool = True,
        parse_mode: str = "ocr",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.camelot_enable = camelot_enable
        self.pdfplumber_enable = pdfplumber_enable
        self.parse_mode = parse_mode.lower()
        if self.parse_mode not in {"ocr", "vlm"}:
            self.parse_mode = "ocr"

    def _extract_selectable_text(self, doc) -> List[Dict[str, Any]]:
        pages = []
        for i, page in enumerate(doc):
            try:
                text = page.get_text("text", sort=True)
            except Exception:
                text = ""
            pages.append({"page_number": i + 1, "text": text})
        return pages

    def _extract_tables_pdfplumber(self, pdf_bytes: bytes) -> Dict[str, Any]:
        out = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                parsed = [t for t in tables]
                out.append({"page_number": i + 1, "tables": parsed})
        return {"pages": out}

    def _extract_tables_camelot(self, pdf_bytes: bytes) -> Dict[str, Any]:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(pdf_bytes)
            tmpname = tf.name
        try:
            tables = camelot.read_pdf(tmpname, pages="all")
            extracted = []
            for t in tables:
                try:
                    df = t.df
                    extracted.append({"shape": df.shape, "csv": df.to_csv(index=False)})
                except Exception:
                    try:
                        extracted.append({"csv": t.to_csv()})
                    except Exception:
                        extracted.append({"error": "failed to convert camelot table"})
            return {"tables_count": len(extracted), "tables": extracted}
        finally:
            try:
                os.remove(tmpname)
            except Exception:
                pass

    def _render_page_to_png(self, page, zoom: float = 2.0) -> bytes:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")

    def _extract_embedded_images(self, doc) -> List[Dict[str, Any]]:
        images_out = []
        for pno in range(len(doc)):
            page = doc[pno]
            try:
                imgs = page.get_images(full=True)
            except Exception:
                imgs = []
            for img_ref in imgs:
                try:
                    xref = img_ref[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n < 5:
                        imgbytes = pix.tobytes("png")
                    else:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        imgbytes = pix.tobytes("png")
                    images_out.append({"page": pno + 1, "bytes_b64": _b64(imgbytes)})
                except Exception:
                    continue
        return images_out

    def parse(self, data: bytes) -> Dict[str, Any]:
        start = time.time()
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            return {"error": "failed to open pdf with pymupdf", "details": str(e)}

        if self.parse_mode == "vlm":
            return self._parse_with_vlm(doc, start)
        return self._parse_with_ocr(data, doc, start)

    def _parse_with_ocr(self, pdf_bytes: bytes, doc, start: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "code": 0,
            "type": "pdf",
            "mode": "ocr",
            "pages": [],
            "tables": {},
            "images": [],
            "warnings": [],
        }

        try:
            result["pages"] = self._extract_selectable_text(doc)
        except Exception as e:
            result["warnings"].append(f"pymupdf text extraction failed: {str(e)}")

        if self.pdfplumber_enable:
            try:
                result["tables_pdfplumber"] = self._extract_tables_pdfplumber(pdf_bytes)
            except Exception as e:
                result["warnings"].append(f"pdfplumber failed: {str(e)}")

        if self.camelot_enable:
            try:
                result["tables_camelot"] = self._extract_tables_camelot(pdf_bytes)
            except Exception as e:
                result["warnings"].append(f"camelot failed: {str(e)}")

        need_ocr_pages: List[int] = []
        try:
            pages = result.get("pages", [])
            if not pages:
                need_ocr_pages = list(range(len(doc)))
            else:
                need_ocr_pages = [i for i, page in enumerate(pages) if not page.get("text", "").strip()]
        except Exception:
            need_ocr_pages = list(range(len(doc)))

        vlm_page_results: List[Dict[str, Any]] = []
        for pnum in need_ocr_pages:
            try:
                page = doc[pnum]
                img_bytes = self._render_page_to_png(page, zoom=2.0)
                if self.use_ocr:
                    try:
                        ocr_result = OCRHelper.image_to_data(img_bytes, lang=self.ocr_lang)
                    except Exception as ocr_err:
                        ocr_result = {"error": str(ocr_err)}
                    if len(result["pages"]) > pnum:
                        result["pages"][pnum]["ocr"] = ocr_result

                if self.vlm_client:
                    prompt = self._build_pdf_page_prompt(pnum + 1)
                    vlm_response = self.vlm_client.generate(prompt, [_b64(img_bytes)])
                else:
                    vlm_response = {"error": "OpenAI client not configured"}
                vlm_page_results.append({"page_number": pnum + 1, "vlm": vlm_response})
            except Exception as e:
                result["warnings"].append(f"analysis failed on page {pnum + 1}: {str(e)}")

        try:
            result["images"] = self._extract_embedded_images(doc)
        except Exception as e:
            result["warnings"].append(f"extract embedded images failed: {str(e)}")

        if vlm_page_results:
            result["vlm_pages"] = vlm_page_results

        result["elapsed_seconds"] = time.time() - start
        return result

    def _build_pdf_page_prompt(self, page_number: int) -> str:
        return (
            "You are converting scanned PDF pages into structured Markdown. "
            f"Transcribe all readable content on page {page_number} from the provided image. "
            "Use Markdown headings, lists and tables whenever appropriate, and render mathematics "
            "using LaTeX syntax between $...$ or $$...$$. Describe figures or charts in one or two "
            "sentences placed where they appear in the page. Respond with Markdown only."
        )

    def _render_document_to_images(self, doc) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        for idx in range(len(doc)):
            try:
                png_bytes = self._render_page_to_png(doc[idx], zoom=2.0)
                images.append({"page_number": idx + 1, "image_b64": _b64(png_bytes)})
            except Exception:
                images.append({"page_number": idx + 1, "error": "render_failed"})
        return images

    def _build_pdf_document_prompt(self, page_images: List[Dict[str, Any]]) -> str:
        total_pages = len(page_images)
        return (
            "You are an expert assistant that converts PDF documents into high quality Markdown. "
            f"The user will provide {total_pages} page image(s). Process them in order, producing "
            "Markdown that mirrors the layout and reading order. Follow these rules:\n"
            "1. Preserve all textual content faithfully.\n"
            "2. Use Markdown tables for tabular data.\n"
            "3. Replace charts, figures or photos with concise descriptive captions at the same position.\n"
            "4. Render mathematics using LaTeX syntax inside $...$ or $$...$$.\n"
            "5. Start each page with a heading like '## Page X' to maintain pagination.\n"
            "Return only Markdown without additional commentary."
        )

    def _parse_with_vlm(self, doc, start: float) -> Dict[str, Any]:
        page_images = self._render_document_to_images(doc)
        images_payload = [item.get("image_b64") for item in page_images if "image_b64" in item]

        if not images_payload:
            return {
                "error": "pdf to image conversion failed",
                "details": page_images,
            }

        if not self.vlm_client:
            vlm_response = {"error": "OpenAI client not configured"}
        else:
            prompt = self._build_pdf_document_prompt(page_images)
            vlm_response = self.vlm_client.generate(prompt, images_payload)

        return {
            "code": 0,
            "type": "pdf",
            "mode": "vlm",
            "pages": [{"page_number": item["page_number"]} for item in page_images],
            "page_images": page_images,
            "vlm": vlm_response,
            "elapsed_seconds": time.time() - start,
        }


class ImageParser(BaseParser):
    def parse(self, data: bytes) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {"code": 0, "type": "image", "vlm": None, "elapsed_seconds": None}
        img_b64 = _b64(data)

        if self.use_ocr:
            try:
                result["ocr"] = OCRHelper.image_to_data(data, lang=self.ocr_lang)
            except Exception as e:
                result["ocr"] = {"error": str(e)}

        if self.vlm_client:
            try:
                prompt = self._build_image_prompt()
                vlm_res = self.vlm_client.generate(prompt, [img_b64])
                result["vlm"] = vlm_res
            except Exception as e:
                result["vlm"] = {"error": str(e)}
        else:
            result["vlm"] = {"error": "OpenAI client not configured"}
        result["elapsed_seconds"] = time.time() - start
        return result

    @staticmethod
    def _build_image_prompt() -> str:
        return (
            "You are analysing a single image. Extract every readable text segment, "
            "describe the visual elements and layout, and reproduce any tables using "
            "Markdown table syntax. When mathematical expressions appear, rewrite them "
            "using LaTeX math enclosed in $...$ or $$...$$. Respond with concise Markdown "
            "combining the transcription and the scene description."
        )


class DocxParser(BaseParser):
    def parse(self, data: bytes) -> Dict[str, Any]:
        start = time.time()
        try:
            doc = docx.Document(io.BytesIO(data))
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            tables = []
            for table in doc.tables:
                rows = []
                for r in table.rows:
                    cells = [c.text for c in r.cells]
                    rows.append(cells)
                tables.append(rows)
            images_vlm: List[Dict[str, Any]] = []
            try:
                rels = doc.part.rels.values()
                for rel in rels:
                    if "image" in rel.reltype:
                        blob = rel.target_part.blob
                        if self.vlm_client:
                            prompt = ImageParser._build_image_prompt()
                            vlm_res = self.vlm_client.generate(prompt, [_b64(blob)])
                        else:
                            vlm_res = {"error": "OpenAI client not configured"}
                        images_vlm.append({"vlm": vlm_res})
            except Exception as e:
                images_vlm.append({"error": f"failed to analyse images: {str(e)}"})

            return {
                "code": 0,
                "type": "docx",
                "content": {"paragraphs": paras, "tables": tables},
                "images_vlm": images_vlm,
                "elapsed_seconds": time.time() - start,
            }
        except Exception as e:
            return {"error": str(e)}


class XlsxParser(BaseParser):
    def parse(self, data: bytes) -> Dict[str, Any]:
        start = time.time()
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
            out = {}
            for name, df in sheets.items():
                out[name] = {"shape": df.shape, "csv": df.to_csv(index=False)}
            return {"code": 0, "type": "xlsx", "content": {"sheets": out}, "elapsed_seconds": time.time() - start}
        except Exception as e:
            return {"error": str(e)}


class PptxParser(BaseParser):
    """Parse .pptx files using python-pptx"""
    def parse(self, data: bytes) -> Dict[str, Any]:
        start = time.time()
        try:
            prs = Presentation(io.BytesIO(data))
        except Exception as e:
            return {"error": f"failed to open pptx: {str(e)}"}

        slides_out = []
        images_out = []
        for i, slide in enumerate(prs.slides):
            slide_text_chunks = []
            notes_text = None
            try:
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        text = shape.text_frame.text.strip()
                        if text:
                            slide_text_chunks.append(text)

                    # 提取图片
                    try:
                        image = getattr(shape, "image", None)
                        if image is not None:
                            blob = image.blob
                            images_out.append({"slide": i + 1, "bytes_b64": _b64(blob)})
                    except Exception:
                        pass

                try:
                    if getattr(slide, "notes_slide", None) is not None and getattr(slide.notes_slide, "notes_text_frame", None) is not None:
                        notes_text = slide.notes_slide.notes_text_frame.text
                except Exception:
                    notes_text = None
            except Exception:
                pass
            slides_out.append({"slide_number": i + 1, "texts": slide_text_chunks, "notes": notes_text})

        images_vlm_results: List[Dict[str, Any]] = []
        for img in images_out:
            if self.vlm_client:
                try:
                    prompt = ImageParser._build_image_prompt()
                    vlm_res = self.vlm_client.generate(prompt, [img["bytes_b64"]])
                except Exception as e:
                    vlm_res = {"error": str(e)}
            else:
                vlm_res = {"error": "OpenAI client not configured"}
            images_vlm_results.append({"slide": img.get("slide"), "vlm": vlm_res})

        return {
            "code": 0,
            "type": "pptx",
            "slides": slides_out,
            "images": images_out,
            "images_vlm": images_vlm_results,
            "elapsed_seconds": time.time() - start,
        }


# -------------------------
# DocumentProcessor: picks parser by extension and dispatches
# -------------------------

class DocumentProcessor:
    def __init__(
        self,
        use_ocr: bool = False,
        ocr_lang: str = "eng",
        camelot_enable: bool = True,
        pdfplumber_enable: bool = True,
        pdf_parse_mode: str = "ocr",
    ):
        resolved_base_url = OPENAI_BASE_URL
        resolved_token = OPENAI_API_KEY
        resolved_model = QWEN_VLM_MODEL

        self.vlm_client = VLMClient(
            api_key=resolved_token,
            base_url=resolved_base_url,
            model=resolved_model,
        )

        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.camelot_enable = camelot_enable
        self.pdfplumber_enable = pdfplumber_enable
        self.pdf_parse_mode = pdf_parse_mode.lower() if isinstance(pdf_parse_mode, str) else "ocr"
        if self.pdf_parse_mode not in {"ocr", "vlm"}:
            self.pdf_parse_mode = "ocr"

    def _select_parser(self, filename: str) -> BaseParser:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return PDFParser(
                use_ocr=self.use_ocr,
                vlm_client=self.vlm_client,
                ocr_lang=self.ocr_lang,
                camelot_enable=self.camelot_enable,
                pdfplumber_enable=self.pdfplumber_enable,
                parse_mode=self.pdf_parse_mode,
            )
        if any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]):
            return ImageParser(use_ocr=self.use_ocr, vlm_client=self.vlm_client, ocr_lang=self.ocr_lang)
        if lower.endswith(".docx") or lower.endswith(".doc"):
            return DocxParser(use_ocr=self.use_ocr, vlm_client=self.vlm_client, ocr_lang=self.ocr_lang)
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            return XlsxParser(use_ocr=self.use_ocr, vlm_client=self.vlm_client, ocr_lang=self.ocr_lang)
        if lower.endswith(".pptx") or lower.endswith(".ppt"):
            return PptxParser(use_ocr=self.use_ocr, vlm_client=self.vlm_client, ocr_lang=self.ocr_lang)
        raise ValueError("Unsupported file extension")

    def parse(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        parser = self._select_parser(filename)
        return parser.parse(file_bytes)


# -------------------------
# MCP wrapper
# -------------------------

mcp = FastMCP() if FastMCP else None

if mcp:
    @mcp.tool
    def parse_document_url(
        file_url: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Main MCP entrypoint."""

        opts = options or {}
        try:
            with httpx.Client(timeout=opts.get("timeout", 60.0)) as client:
                resp = client.get(file_url)
                resp.raise_for_status()
                file_bytes = resp.content
        except Exception as e:
            return {"error": f"Failed to fetch file: {str(e)}"}

        filename = os.path.basename(urllib.parse.urlparse(file_url).path) or "downloaded_file"
        
        dp = DocumentProcessor(
            use_ocr=opts.get("use_ocr", False),
            ocr_lang=opts.get("ocr_lang", "eng"),
            camelot_enable=opts.get("camelot_enable", True),
            pdfplumber_enable=opts.get("pdfplumber_enable", True),
            pdf_parse_mode=opts.get("pdf_parse_mode", "ocr"),
        )

        try:
            parsed = dp.parse(file_bytes, filename)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except ValueError as e:
            return {"error": str(e)}
        except Exception:
            return {"error": "Unhandled exception during parse", "trace": traceback.format_exc()}

    @mcp.custom_route("/api/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "document-parser-mcp-refactored"})

if __name__ == "__main__":
    if mcp:
        host = os.getenv("MCP_HOST", "localhost")
        port = int(os.getenv("MCP_PORT", "50002"))
        mcp.run(transport="http", host=host, port=port, path="/mcp", log_level="info")
    else:
        print("FastMCP not available, cannot run service.")
