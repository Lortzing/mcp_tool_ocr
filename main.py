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

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_ENDPOINT,
    DASHSCOPE_MODEL,
    DASHSCOPE_PROVIDER,
)

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

def _b64(bytestr: bytes) -> str:
    return base64.b64encode(bytestr).decode("utf-8")


class VLMClient:
    """VLM client (sync). assumes httpx is available and endpoint configured if used."""

    def __init__(
        self,
        endpoint: str = "",
        token: Optional[str] = None,
        timeout: float = 120.0,
        provider: str | None = None,
        model: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.token = token
        self.timeout = timeout
        self.provider = (provider or "generic").lower()
        self.model = model

    def call_sync(
        self,
        images_b64: List[str],
        ocr_text: str,
        task: str = "document_understanding",
    ) -> Dict[str, Any]:
        """Dispatch to the correct backend based on provider."""

        if not self.endpoint:
            return {"error": "VLM endpoint not configured"}

        if "dashscope.aliyuncs.com" in self.endpoint or self.provider == "dashscope":
            return self._call_dashscope(images_b64, ocr_text, task)

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        payload = {"images_b64": images_b64, "ocr_text": ocr_text, "task": task}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                return {"vlm_result": resp.json()}
        except Exception as e:
            return {"error": f"VLM call failed: {str(e)}"}

    def _call_dashscope(self, images_b64: List[str], ocr_text: str, task: str) -> Dict[str, Any]:
        """Call Qwen2.5-VL through DashScope's OpenAI-compatible endpoint."""

        if not self.token:
            return {"error": "DashScope VLM requires an access token"}

        model = self.model or "qwen2.5-vl"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        messages_content: List[Dict[str, Any]] = []
        if ocr_text:
            messages_content.append(
                {
                    "type": "input_text",
                    "text": f"Task: {task}\nOCR Context:\n{ocr_text}",
                }
            )
        else:
            messages_content.append({"type": "input_text", "text": f"Task: {task}"})

        for img in images_b64:
            messages_content.append({"type": "input_image", "image": img})

        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": messages_content,
                }
            ],
        }

        endpoint = self.endpoint.rstrip("/")
        # DashScope uses /responses for the OpenAI compatible format.
        if not endpoint.endswith("/responses"):
            endpoint = f"{endpoint}/responses"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                return {"vlm_result": resp.json()}
        except Exception as e:
            return {"error": f"DashScope VLM call failed: {str(e)}"}


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
    def __init__(self, use_ocr: bool = False, vlm_client: Optional[VLMClient] = None, ocr_lang: str | List[str] = "eng"):
        # OCR is disabled by default. We keep the flag for backwards compatibility
        # but avoid invoking OCR logic in new flows unless explicitly requested.
        self.use_ocr = use_ocr
        self.vlm_client = vlm_client
        self.ocr_lang = ocr_lang

    def parse(self, data: bytes) -> Dict[str, Any]:
        raise NotImplementedError


class PDFParser(BaseParser):
    def __init__(self, *args, camelot_enable: bool = True, pdfplumber_enable: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.camelot_enable = camelot_enable
        self.pdfplumber_enable = pdfplumber_enable

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
        result: Dict[str, Any] = {"code": 0, "type": "pdf", "pages": [], "tables": {}, "images": [], "warnings": []}
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            return {"error": "failed to open pdf with pymupdf", "details": str(e)}

        # selectable text
        try:
            result["pages"] = self._extract_selectable_text(doc)
        except Exception as e:
            result["warnings"].append(f"pymupdf text extraction failed: {str(e)}")

        # tables
        if self.pdfplumber_enable:
            try:
                result["tables_pdfplumber"] = self._extract_tables_pdfplumber(data)
            except Exception as e:
                result["warnings"].append(f"pdfplumber failed: {str(e)}")

        if self.camelot_enable:
            try:
                result["tables_camelot"] = self._extract_tables_camelot(data)
            except Exception as e:
                result["warnings"].append(f"camelot failed: {str(e)}")

        # pages that need OCR
        need_ocr_pages = []
        try:
            if not result["pages"] or any((not p.get("text", "").strip()) for p in result["pages"]):
                if result.get("pages"):
                    need_ocr_pages = [i for i, p in enumerate(result.get("pages", [])) if not p.get("text", "").strip()]
                else:
                    need_ocr_pages = list(range(len(doc)))
        except Exception:
            need_ocr_pages = list(range(len(doc)))

        vlm_page_results: List[Dict[str, Any]] = []
        for pnum in need_ocr_pages:
            try:
                page = doc[pnum]
                img_bytes = self._render_page_to_png(page, zoom=2.0)
                if self.vlm_client and self.vlm_client.endpoint:
                    vlm_response = self.vlm_client.call_sync([_b64(img_bytes)], "")
                else:
                    vlm_response = {"error": "vlm endpoint not configured"}
                vlm_page_results.append({"page_number": pnum + 1, "vlm": vlm_response})
            except Exception as e:
                result["warnings"].append(f"VLM analysis failed on page {pnum+1}: {str(e)}")

        # embedded images
        try:
            result["images"] = self._extract_embedded_images(doc)
        except Exception as e:
            result["warnings"].append(f"extract embedded images failed: {str(e)}")

        if vlm_page_results:
            result["vlm_pages"] = vlm_page_results

        result["elapsed_seconds"] = time.time() - start
        return result


class ImageParser(BaseParser):
    def parse(self, data: bytes) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {"code": 0, "type": "image", "vlm": None, "elapsed_seconds": None}
        if self.vlm_client and self.vlm_client.endpoint:
            try:
                img_b64 = _b64(data)
                vlm_res = self.vlm_client.call_sync([img_b64], "")
                result["vlm"] = vlm_res
            except Exception as e:
                result["vlm"] = {"error": str(e)}
        else:
            result["vlm"] = {"error": "vlm endpoint not configured"}
        result["elapsed_seconds"] = time.time() - start
        return result


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
                        if self.vlm_client and self.vlm_client.endpoint:
                            vlm_res = self.vlm_client.call_sync([_b64(blob)], "")
                        else:
                            vlm_res = {"error": "vlm endpoint not configured"}
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
            if self.vlm_client and self.vlm_client.endpoint:
                try:
                    vlm_res = self.vlm_client.call_sync([img["bytes_b64"]], "")
                except Exception as e:
                    vlm_res = {"error": str(e)}
            else:
                vlm_res = {"error": "vlm endpoint not configured"}
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
        vlm_endpoint: Optional[str] = None,
        vlm_token: Optional[str] = None,
        ocr_lang: str = "eng",
        camelot_enable: bool = True,
        pdfplumber_enable: bool = True,
        vlm_provider: Optional[str] = None,
        vlm_model: Optional[str] = None,
        enable_vlm: bool = False,
    ):
        if enable_vlm:
            resolved_endpoint = vlm_endpoint or DASHSCOPE_ENDPOINT
            resolved_token = vlm_token or DASHSCOPE_API_KEY
            resolved_provider = (vlm_provider or DASHSCOPE_PROVIDER) if DASHSCOPE_PROVIDER else vlm_provider
            resolved_model = vlm_model or DASHSCOPE_MODEL

            self.vlm_client = VLMClient(
                endpoint=resolved_endpoint,
                token=resolved_token,
                provider=resolved_provider,
                model=resolved_model,
            )
        else:
            self.vlm_client = None
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.camelot_enable = camelot_enable
        self.pdfplumber_enable = pdfplumber_enable

    def _select_parser(self, filename: str) -> BaseParser:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return PDFParser(use_ocr=self.use_ocr, vlm_client=self.vlm_client, ocr_lang=self.ocr_lang, camelot_enable=self.camelot_enable, pdfplumber_enable=self.pdfplumber_enable)
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

        run_vlm = opts.get("run_vlm", False)
        vlm_api_url = opts.get("vlm_api_url")
        vlm_token = opts.get("vlm_token")
        vlm_model = opts.get("vlm_model")
        vlm_provider = opts.get("vlm_provider")

        if not vlm_provider and isinstance(vlm_api_url, str) and "dashscope.aliyuncs.com" in vlm_api_url:
            vlm_provider = "dashscope"

        dp = DocumentProcessor(
            use_ocr=opts.get("use_ocr", False),
            vlm_endpoint=vlm_api_url if run_vlm and vlm_api_url else None,
            vlm_token=vlm_token,
            ocr_lang=opts.get("ocr_lang", "eng"),
            camelot_enable=opts.get("camelot_enable", True),
            pdfplumber_enable=opts.get("pdfplumber_enable", True),
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            enable_vlm=run_vlm,
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
