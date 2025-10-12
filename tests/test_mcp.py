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


class _DummyOpenAIResponse:
    def __init__(self):
        self.output = []

    def model_dump(self):
        return {"ok": True}


class _DummyOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.responses = types.SimpleNamespace(create=lambda *a, **k: _DummyOpenAIResponse())


_make_module("openai", {"OpenAI": _DummyOpenAIClient})

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
    class FakeDoc:
        def __init__(self, n):
            self._n = n

        def __len__(self):
            return self._n

        def __iter__(self):
            return iter([None] * self._n)

        def __getitem__(self, idx):
            return object()

    monkeypatch.setattr(dpr.fitz, "open", lambda *a, **k: FakeDoc(2))
    monkeypatch.setattr(
        dpr.PDFParser,
        "_extract_selectable_text",
        lambda self, doc: [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "already"}],
    )
    monkeypatch.setattr(dpr.PDFParser, "_render_page_to_png", lambda self, page, zoom=2.0: b"PNG_BYTES")
    monkeypatch.setattr(
        dpr.OCRHelper,
        "image_to_data",
        staticmethod(lambda b, lang="eng": {"blocks": [{"text": "recognized"}]}),
    )

    vlm = dpr.VLMClient(api_key="")
    called: dict = {}

    def fake_generate(prompt, images):
        called["prompt"] = prompt
        called["images"] = images
        return {"markdown": "page markdown"}

    monkeypatch.setattr(vlm, "generate", fake_generate)

    pdfp = dpr.PDFParser(use_ocr=True, vlm_client=vlm, ocr_lang="eng", parse_mode="ocr")
    res = pdfp.parse(b"%PDF-dummy%")
    assert res["mode"] == "ocr"
    assert res["pages"][0]["ocr"] == {"blocks": [{"text": "recognized"}]}
    assert res["vlm_pages"][0]["vlm"] == {"markdown": "page markdown"}
    assert called["images"] == [dpr._b64(b"PNG_BYTES")]
    assert "markdown" in called["prompt"].lower()
    element_types = [el["type"] for el in res["pages"][0]["elements"]]
    assert "ocr_text" in element_types
    assert any(el.get("vlm", {}).get("markdown") == "page markdown" for el in res["pages"][0]["elements"] if el["type"] == "vlm_markdown")
    assert all("image_b64" not in el for el in res["pages"][0]["elements"])


def test_pdfparser_vlm_mode_generates_markdown(monkeypatch):
    class FakeDoc:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return object()

    monkeypatch.setattr(dpr.fitz, "open", lambda *a, **k: FakeDoc())
    monkeypatch.setattr(
        dpr.PDFParser,
        "_render_document_to_images",
        lambda self, doc: [{"page_number": 1, "image_b64": "IMG_B64"}],
    )

    vlm = dpr.VLMClient(api_key="")
    captured: dict = {}

    def fake_generate(prompt, images):
        captured["prompt"] = prompt
        captured["images"] = images
        return {"markdown": "## Page 1\nConverted"}

    monkeypatch.setattr(vlm, "generate", fake_generate)

    pdfp = dpr.PDFParser(use_ocr=False, vlm_client=vlm, parse_mode="vlm")
    res = pdfp.parse(b"%PDF%")
    assert res["mode"] == "vlm"
    assert res["vlm"] == {"markdown": "## Page 1\nConverted"}
    assert captured["images"] == ["IMG_B64"]
    assert "markdown" in captured["prompt"].lower()
    assert res["pages"][0]["elements"][0]["type"] == "image"
    assert res["document_elements"][0]["vlm"] == {"markdown": "## Page 1\nConverted"}
    assert "description" in res["page_images"][0]


def test_imageparser_ocr_and_vlm(monkeypatch):
    monkeypatch.setattr(
        dpr.OCRHelper,
        "image_to_data",
        staticmethod(lambda b, lang="eng": {"blocks": [{"text": "imgtext"}]}),
    )

    vlm = dpr.VLMClient(api_key="")
    called: dict = {}

    def fake_generate(prompt, images):
        called["prompt"] = prompt
        called["images"] = images
        return {"markdown": "image markdown"}

    monkeypatch.setattr(vlm, "generate", fake_generate)

    ip = dpr.ImageParser(use_ocr=True, vlm_client=vlm, ocr_lang="eng")
    image_bytes = b"\x89PNG...fake"
    res = ip.parse(image_bytes)
    assert res["ocr"] == {"blocks": [{"text": "imgtext"}]}
    assert res["vlm"] == {"markdown": "image markdown"}
    assert called["images"] == [dpr._b64(image_bytes)]
    assert "tables" in called["prompt"].lower()
    flow_types = [item["type"] for item in res["content_flow"]]
    assert flow_types[0] == "image"
    assert res["content_flow"][-1]["vlm"] == {"markdown": "image markdown"}
    assert res["image"]["description"].startswith("[binary content omitted")
    assert "image_b64" not in str(res)

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
    flow_types = [item["type"] for item in res["content"]["flow"]]
    assert flow_types.count("paragraph") >= 1
    assert "table" in flow_types
    assert res["images"] == [] or "description" in res["images"][0]

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


def test_build_markdown_report_produces_plain_text():
    parsed = {
        "type": "pdf",
        "mode": "ocr",
        "elapsed_seconds": 0.5,
        "pages": [
            {
                "page_number": 1,
                "text": "Hello world",
                "elements": [
                    {"type": "image", "description": "[binary content omitted: embedded image]"},
                    {"type": "vlm_markdown", "vlm": {"markdown": "**Summary**"}},
                ],
            }
        ],
        "images": [{"page": 1, "description": "[binary content omitted: embedded image]"}],
    }
    markdown = dpr.build_markdown_report(parsed, filename="sample.pdf")
    assert markdown.startswith("# Parse Result for sample.pdf")
    assert "embedded image" in markdown
    assert "IMG_B64" not in markdown
    assert "**Summary**" in markdown

# ------------------------------------------------------------
# run tests with: pytest -q
# ------------------------------------------------------------
