from typing import List, Dict

import dotenv
from langchain_community.tools import TavilySearchResults

dotenv.load_dotenv()

# 도구 생성
tavily_tool = TavilySearchResults(
    search_depth="advanced",
    include_answer=False,
    include_raw_content=True,
    time_range='week',
    max_results=3
)

def search_web(query: str) -> List[Dict]:
    results = tavily_tool.invoke(query, k=2)
    news = []
    for item in results:
        news.append({"title": item["title"], "url": item["url"], "content": item["content"]})
    return news


if __name__ == "__main__":
    print(search_web("IT 뉴스 알려줘"))
