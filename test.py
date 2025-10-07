import asyncio
from fastmcp import Client

async def main():
    # 连接到你的本地 MCP 服务
    async with Client("http://localhost:50002/mcp") as client:
        # 列出可用工具（确认连接成功）
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        # 调用 parse_document_url 工具
        print("\n--- Running parse_document_url ---")
        result = await client.call_tool(
            "parse_document_url",
            {
                "file_url": "http://localhost:5050/example_files/test_docx.docx",
                "run_vlm": False,
                "use_ocr": False,
                "ocr_lang": "eng",
                "camelot_enable": False,
                "pdfplumber_enable": False,
            },
        )

        print("\n--- Result ---")
        if isinstance(result, dict):
            # 打印主要字段
            print("Type:", result.get("type"))
            print("Keys:", list(result.keys())[:10])
            print("Elapsed seconds:", result.get("elapsed_seconds"))
        else:
            print(result.data)

if __name__ == "__main__":
    asyncio.run(main())
