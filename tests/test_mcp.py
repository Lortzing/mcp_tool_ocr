# tests/test_mcp.py
import sys
import types
import base64
import io

# ------------------------------------------------------------
# 1) 在导入被测模块前，先往 sys.modules 注入必要的 stub 模块，
#    以避免测试环境必须装齐所有第三方依赖。
# ------------------------------------------------------------
def _make_module(name: str, attrs: dict = None):
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    return m

# minimal stub for pytesseract so "from pytesseract import Output as TESSERACT_OUTPUT" works
_make_module("pytesseract", {"Output": object, "image_to_data": lambda *a, **k: {}})

# minimal stub for PIL.Image
class _DummyPILImage:
    @staticmethod
    def open(fp):
        class Img:
            def convert(self, mode):
                return Img()
        return Img()
_make_module("PIL", {"Image": _DummyPILImage})

# stub other modules imported at top of document_parser_refactored.py
_make_module(
    "fitz",
    {
        "Matrix": lambda x, y: None,
        "Pixmap": type(
            "_Pixmap",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "tobytes": lambda self, fmt: b"",
            },
        ),
        "csRGB": object(),
        "open": lambda *a, **k: None,
    },
)
_make_module("pdfplumber")
_make_module("camelot", {"read_pdf": lambda *a, **k: []})
_make_module("docx", {"Document": lambda f: None})
_make_module("pandas", {"read_excel": lambda *a, **k: {}})
_make_module("openpyxl")
_make_module("pptx", {"Presentation": lambda f: None})

# httpx stub (not used in most tests but must exist)
class _SimpleResponse:
    def __init__(self, content=b"", json_obj=None):
        self.content = content
        self._json = json_obj or {}
    def raise_for_status(self):
        return None
    def json(self):
        return self._json

class _HttpxClient:
    def __init__(self, timeout=None):
        self.timeout = timeout
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def get(self, url):
        return _SimpleResponse(content=b"", json_obj={})
    def post(self, url, json=None, headers=None):
        return _SimpleResponse(content=b"", json_obj={"ok": True})

_make_module("httpx", {"Client": _HttpxClient})

# fastmcp stub: set FastMCP to None so parse_document_url isn't registered (tests don't need MCP runtime)
_make_module("fastmcp", {"FastMCP": None})

# starlette stubs
_make_module("starlette")
_make_module("starlette.requests", {"Request": object})
_make_module("starlette.responses", {"JSONResponse": lambda j: j})

# ------------------------------------------------------------
# 2) 现在导入被测模块（此刻不会因为缺少依赖而导入失败）
# ------------------------------------------------------------
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main as dpr

# convenience: base64 helper from module (if needed)
_b64 = dpr._b64

# ------------------------------------------------------------
# 3) 测试开始
# ------------------------------------------------------------

def test_select_parser_mapping_basic():
    dp = dpr.DocumentProcessor(use_ocr=False)
    assert isinstance(dp._select_parser("file.pdf"), dpr.PDFParser)
    assert isinstance(dp._select_parser("file.PNG"), dpr.ImageParser)
    assert isinstance(dp._select_parser("file.jpeg"), dpr.ImageParser)
    assert isinstance(dp._select_parser("file.docx"), dpr.DocxParser)
    assert isinstance(dp._select_parser("file.xlsX"), dpr.XlsxParser)
    assert isinstance(dp._select_parser("file.pptx"), dpr.PptxParser)

def test_documentprocessor_dispatch_calls_parser(monkeypatch):
    dp = dpr.DocumentProcessor(use_ocr=False)
    class DummyParser:
        def parse(self, file_bytes):
            return {"parsed_by": "dummy"}
    # replace _select_parser so parse() returns our dummy
    monkeypatch.setattr(dpr.DocumentProcessor, "_select_parser", lambda self, filename: DummyParser())
    res = dp.parse(b"irrelevant", "some.pdf")
    assert res == {"parsed_by": "dummy"}

def test_pdfparser_triggers_ocr_and_vlm(monkeypatch):
    """
    验证在“可选文本为空 -> 需要 OCR -> OCR 调用并且 VLM 被触发”的主要控制流。
    使用 fake doc + 替换 PDFParser 的局部方法，避免真实依赖 fitz/pdfplumber/camelot。
    """
    # 1) 准备 Fake fitz.open 返回的 doc：支持迭代/索引/len（两页）
    class FakeDoc:
        def __init__(self, n):
            self._n = n
        def __len__(self): return self._n
        def __iter__(self):
            # iteration only used by our patched _extract_selectable_text (we patch it)
            return iter([None]*self._n)
        def __getitem__(self, idx):
            return object()

    monkeypatch.setattr(dpr.fitz, "open", lambda *a, **k: FakeDoc(2))

    # 2) 强制 selectable text 为空 => 触发 OCR（模拟返回两个页面，第一页无文本）
    def fake_extract_selectable_text(self, doc):
        return [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "already"}]
    monkeypatch.setattr(dpr.PDFParser, "_extract_selectable_text", fake_extract_selectable_text)

    # 3) 把 _render_page_to_png 返回固定 png bytes（不使用真实渲染）
    monkeypatch.setattr(dpr.PDFParser, "_render_page_to_png", lambda self, page, zoom=2.0: b"PNG_BYTES")

    # 4) 模拟 OCR 输出
    monkeypatch.setattr(dpr.OCRHelper, "image_to_data", staticmethod(lambda b, lang="eng": {"blocks":[{"text":"recognized"}]}))

    # 5) 模拟 embedded images 提取（用于 VLM 的输入）
    monkeypatch.setattr(dpr.PDFParser, "_extract_embedded_images", lambda self, doc: [{"page":1, "bytes_b64": _b64(b"IMG1") }])

    # 6) 准备 VLMClient 实例并替换 call_sync 为可观测的 fake
    vlm = dpr.VLMClient(endpoint="http://vlm")
    called = {}
    def fake_vlm_call(images_b64, ocr_text, task="document_understanding"):
        called['imgs'] = images_b64
        called['ocr_text'] = ocr_text
        return {"vlm_result": "ok"}
    monkeypatch.setattr(vlm, "call_sync", fake_vlm_call)

    # 实例并调用
    pdfp = dpr.PDFParser(use_ocr=True, vlm_client=vlm, ocr_lang="eng")
    res = pdfp.parse(b"%PDF-dummy%")
    # 断言 OCR 已被放到页面结果中
    assert "pages" in res
    assert res["pages"][0].get("ocr") == {"blocks":[{"text":"recognized"}]}
    # 断言 VLM 被调用，且传入的 images_b64 经过 base64 编码（我们在 fake_extract_embedded_images 提供了图片）
    assert res.get("vlm") == {"vlm_result": "ok"}
    assert isinstance(called.get("imgs"), list)
    assert "recognized" in called.get("ocr_text", "")

def test_imageparser_ocr_and_vlm(monkeypatch):
    # OCR fake
    monkeypatch.setattr(dpr.OCRHelper, "image_to_data", staticmethod(lambda b, lang="eng": {"blocks":[{"text":"imgtext"}]}))
    # VLM fake client
    vlm = dpr.VLMClient(endpoint="http://vlm")
    monkeypatch.setattr(vlm, "call_sync", lambda imgs, ocr_text: {"vlm_result":"img_ok"})

    ip = dpr.ImageParser(use_ocr=True, vlm_client=vlm, ocr_lang="eng")
    res = ip.parse(b"\x89PNG...fake")
    assert res["ocr"] == {"blocks":[{"text":"imgtext"}]}
    assert res["vlm"] == {"vlm_result":"img_ok"}

def test_docxparser_parses_paragraphs_and_tables(monkeypatch):
    # fake Document object returning paragraphs and tables
    class FakePara:
        def __init__(self, txt): self.text = txt
    class FakeCell:
        def __init__(self, txt): self.text = txt
    class FakeRow:
        def __init__(self, cells): self.cells = [FakeCell(c) for c in cells]
    class FakeTable:
        def __init__(self, rows): self.rows = [FakeRow(r) for r in rows]
    class FakeDoc:
        def __init__(self, f):
            self.paragraphs = [FakePara("p1"), FakePara("  "), FakePara("p2")]
            self.tables = [FakeTable([["r1c1","r1c2"], ["r2c1","r2c2"]])]
    # patch docx.Document
    monkeypatch.setattr(dpr.docx, "Document", lambda f: FakeDoc(f))

    dp = dpr.DocxParser(use_ocr=False)
    res = dp.parse(b"dummy docx")
    assert res["type"] == "docx"
    assert res["content"]["paragraphs"] == ["p1", "p2"]
    assert isinstance(res["content"]["tables"], list)
    assert res["content"]["tables"][0][0][0] == "r1c1"

def test_xlsxparser_reads_sheets(monkeypatch):
    # fake DataFrame-like with shape and to_csv
    class FakeDF:
        def __init__(self):
            self.shape = (2, 1)
        def to_csv(self, index=False):
            return "col\n1\n2\n"
    monkeypatch.setattr(dpr.pd, "read_excel", lambda b, sheet_name=None: {"Sheet1": FakeDF()})

    xp = dpr.XlsxParser()
    res = xp.parse(b"fake xlsx")
    assert res["type"] == "xlsx"
    assert "Sheet1" in res["content"]["sheets"]
    assert res["content"]["sheets"]["Sheet1"]["shape"] == (2, 1)

# ------------------------------------------------------------
# run tests with: pytest -q
# ------------------------------------------------------------
