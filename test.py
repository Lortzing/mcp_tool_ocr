import asyncio
import json
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
                "file_url": "https://sample-files.com/downloads/documents/pdf/sample-report.pdf",
            },
        )

        print("\n--- Result ---")
        payload = None
        if isinstance(result, dict):
            payload = result
        elif hasattr(result, "content"):
            for item in getattr(result, "content", []):
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    try:
                        payload = json.loads(text_value)
                    except json.JSONDecodeError:
                        payload = text_value
                    break

        if isinstance(payload, dict):
            print("Type:", payload.get("type"))
            print("Keys:", list(payload.keys())[:10])
            print("Elapsed seconds:", payload.get("elapsed_seconds"))
            summary = payload.get("summary")
            if isinstance(summary, str):
                print("Summary snippet:", summary[:200])
        elif isinstance(payload, str):
            print(payload[:500])
        else:
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
