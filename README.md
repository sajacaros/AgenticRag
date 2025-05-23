# AgenticRag
## poetry init
- apply pyproject.toml
## Tavily 키 발급
- https://app.tavily.com/home
## 네이버 키 발급
- https://developers.naver.com/apps
## copy .env
- `.env_template -> .env`
## uv
```
$ uv add package
$ uv add package --extra starndard # packate[starndard]
$ uv export -o requirements.txt   
$ uv python install 3.12
$ uv sync
```
## .env 작성

## MCP 서버 실행(src/mcp/server.py)
## 서버 스펙 작성(src/mcp/server_transport.json)
```json
{
  "mcpServers": {
    "default-server": {
      "type": "sse",
      "url": "http://localhost:8123/sse",
      "note": "For SSE connections, add this URL directly in Client"
    }
  }
}
```
## inspector 실행
- MCP에서 제공하는 개발용 디버깅 웹페이지
- node만 설치되어 있으면 실행 가능
```shell
npx @modelcontextprotocol/inspector --config .\src\mcp\server_transport.json --server default-server
```
## inspector 접속 및 확인
## MCP Client 실행(src/mcp/MCP_Client.ipynb)