import os
import requests
from typing import List, Dict

import dotenv

dotenv.load_dotenv()

def search_blog(query:  str) -> List[Dict]:
    """네이버 블로그 API에 검색 요청을 보냅니다."""
    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"]
    }
    params = {"query": query, "display": 3, "start": 1}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        results = []
        for item in response.json()['items']:
            results.append({"title": item["title"], "url": item["link"], "content": item["description"]})
        return results
    else:
        return []


if __name__ == "__main__":
    print(search_blog("IT 뉴스 알려줘"))
