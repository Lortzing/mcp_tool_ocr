# document_parser_refactored.py
from __future__ import annotations

import io
import os
import time
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

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

def _b64(bytestr: bytes) -> str:
    return base64.b64encode(bytestr).decode("utf-8")


class VLMClient:
    """VLM client (sync). assumes httpx is available and endpoint configured if used."""
    def __init__(self, endpoint: str = "", token: Optional[str] = None, timeout: float = 120.0):
        self.endpoint = endpoint
        self.token = token
        self.timeout = timeout

    def call_sync(self, images_b64: List[str], ocr_text: str, task: str = "document_understanding") -> Dict[str, Any]:
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
        # OCR is disabled by default per latest requirements. The parameter is kept
        # for backwards compatibility but ignored by the new logic.
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

        vlm_page_requests: List[Dict[str, Any]] = []
        for pnum in need_ocr_pages:
            try:
                page = doc[pnum]
                img_bytes = self._render_page_to_png(page, zoom=2.0)
                vlm_page_requests.append({"page": pnum + 1, "image_b64": _b64(img_bytes)})
            except Exception as e:
                result["warnings"].append(f"render page {pnum+1} for vlm failed: {str(e)}")

        # embedded images
        try:
            result["images"] = self._extract_embedded_images(doc)
        except Exception as e:
            result["warnings"].append(f"extract embedded images failed: {str(e)}")

        page_vlm_results = []
        images_vlm_results = []
        if self.vlm_client and self.vlm_client.endpoint:
            for payload in vlm_page_requests:
                try:
                    vlm_result = self.vlm_client.call_sync([payload["image_b64"]], "")
                except Exception as e:
                    vlm_result = {"error": f"VLM call failed: {str(e)}"}
                page_vlm_results.append({"page": payload.get("page"), "vlm": vlm_result})
                page_index = (payload.get("page") or 1) - 1
                if 0 <= page_index < len(result.get("pages", [])):
                    result["pages"][page_index]["vlm"] = vlm_result

            for img in result.get("images", []):
                img_b64 = img.get("bytes_b64")
                if not img_b64:
                    continue
                try:
                    vlm_result = self.vlm_client.call_sync([img_b64], "")
                except Exception as e:
                    vlm_result = {"error": f"VLM call failed: {str(e)}"}
                images_vlm_results.append({"page": img.get("page"), "vlm": vlm_result})
        else:
            if vlm_page_requests:
                result["warnings"].append("Pages without selectable text detected but VLM endpoint not configured.")
            if result.get("images"):
                result.setdefault("warnings", []).append("Embedded images present but VLM endpoint not configured.")

        if page_vlm_results:
            result["vlm_on_pages"] = page_vlm_results
        if images_vlm_results:
            result["vlm_on_images"] = images_vlm_results

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
            result["warnings"] = ["VLM endpoint not configured; image content not analysed."]
        result["elapsed_seconds"] = time.time() - start
        return result


class DocxParser(BaseParser):
    def _extract_images(self, document: docx.Document) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        rels = getattr(document.part, "rels", {})
        for rel in rels.values():
            if getattr(rel, "is_external", False):
                continue
            target_part = getattr(rel, "target_part", None)
            if not target_part:
                continue
            if getattr(target_part, "content_type", "").startswith("image/"):
                try:
                    blob = target_part.blob
                    images.append({"bytes_b64": _b64(blob), "content_type": target_part.content_type})
                except Exception:
                    continue
        return images

    def parse(self, data: bytes) -> Dict[str, Any]:
        start = time.time()
        try:
            doc = docx.Document(io.BytesIO(data))
        except Exception as e:
            return {"error": str(e)}

        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        tables: List[List[List[str]]] = []
        for table in doc.tables:
            rows = []
            for r in table.rows:
                cells = [c.text for c in r.cells]
                rows.append(cells)
            tables.append(rows)

        images = self._extract_images(doc)
        vlm_results = []
        if images and self.vlm_client and self.vlm_client.endpoint:
            for img in images:
                try:
                    vlm_res = self.vlm_client.call_sync([img["bytes_b64"]], "")
                except Exception as e:
                    vlm_res = {"error": str(e)}
                vlm_results.append({"content_type": img.get("content_type"), "vlm": vlm_res})
        elif images:
            vlm_results = [{"content_type": img.get("content_type"), "vlm": {"warning": "VLM endpoint not configured"}} for img in images]

        return {
            "code": 0,
            "type": "docx",
            "content": {"paragraphs": paras, "tables": tables},
            "images": images,
            "vlm_on_images": vlm_results,
            "elapsed_seconds": time.time() - start,
        }


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

                    try:
                        image = getattr(shape, "image", None)
                        if image is not None:
                            blob = image.blob
                            images_out.append({"slide": i + 1, "bytes_b64": _b64(blob)})
                    except Exception:
                        pass

                try:
                    if (
                        getattr(slide, "notes_slide", None) is not None
                        and getattr(slide.notes_slide, "notes_text_frame", None) is not None
                    ):
                        notes_text = slide.notes_slide.notes_text_frame.text
                except Exception:
                    notes_text = None
            except Exception:
                pass
            slides_out.append({"slide_number": i + 1, "texts": slide_text_chunks, "notes": notes_text})

        vlm_responses = []
        if self.vlm_client and self.vlm_client.endpoint and images_out:
            for image in images_out:
                try:
                    vlm_output = self.vlm_client.call_sync([image["bytes_b64"]], "")
                except Exception as e:
                    vlm_output = {"error": str(e)}
                vlm_responses.append({"slide": image.get("slide"), "vlm": vlm_output})
        elif images_out:
            vlm_responses = [
                {"slide": image.get("slide"), "vlm": {"warning": "VLM endpoint not configured"}}
                for image in images_out
            ]

        return {
            "code": 0,
            "type": "pptx",
            "slides": slides_out,
            "images": images_out,
            "vlm_on_images": vlm_responses,
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
    ):
        self.vlm_client = VLMClient(endpoint=vlm_endpoint, token=vlm_token) if vlm_endpoint else None
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
        run_vlm: bool = False,
        vlm_api_url: Optional[str] = None,
        vlm_token: Optional[str] = None,
        use_ocr: bool = False,
        ocr_lang: str = "eng",
        camelot_enable: bool = True,
        pdfplumber_enable: bool = True,
    ) -> str:
        """Main MCP entrypoint.
        - file_url: http(s) URL pointing to the file
        - run_vlm + vlm_api_url: if True and provided, VLM will be used
        """
        import json

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(file_url)
                resp.raise_for_status()
                file_bytes = resp.content
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch file: {str(e)}"}, ensure_ascii=False)

        filename = os.path.basename(urllib.parse.urlparse(file_url).path) or "downloaded_file"

        vlm_endpoint = vlm_api_url if run_vlm and vlm_api_url else None
        dp = DocumentProcessor(
            use_ocr=use_ocr,
            vlm_endpoint=vlm_endpoint,
            vlm_token=vlm_token,
            ocr_lang=ocr_lang,
            camelot_enable=camelot_enable,
            pdfplumber_enable=pdfplumber_enable,
        )

        try:
            result = dp.parse(file_bytes, filename)
        except ValueError as e:
            result = {"error": str(e)}
        except Exception:
            result = {"error": "Unhandled exception during parse", "trace": traceback.format_exc()}

        return json.dumps(result, ensure_ascii=False)

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
