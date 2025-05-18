# server.py
from typing import List, Dict

from mcp.server.fastmcp import FastMCP

from src.mcp import tavily_search, naver_blog_search

# Create an MCP server
mcp = FastMCP("Demo")

@mcp.tool()
async def web_search(query: str) -> List[Dict]:
    """
    웹 검색 API에 검색 요청을 보냅니다.
    """
    return tavily_search.search_web(query)

@mcp.tool()
async def blog_search(query: str) -> List[Dict]:
    """
    네이버 블로그 API에 검색 요청을 보냅니다.
    """
    return naver_blog_search.search_blog(query)


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="sse")