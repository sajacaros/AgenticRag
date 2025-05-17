from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_openai import ChatOpenAI

# 오늘 날짜 설정
today = datetime.today().strftime("%Y-%m-%d")

# 프롬프트 템플릿
system = """You are an AI assistant providing restaurant menu information and general food-related knowledge.
For information about the restaurant's menu, use the search_menu tool.
For other general information, use the wiki_summary tool.
For wine recommendations or pairing information, use the search_wine tool.
If additional web searches are needed or for the most up-to-date information, use the search_web tool.
"""

examples = [
    HumanMessage("트러플 리조또의 가격과 특징, 그리고 어울리는 와인에 대해 알려주세요.", name="example_user"),
    AIMessage("메뉴 정보를 검색하고, 위키피디아에서 추가 정보를 찾은 후, 어울리는 와인을 검색해보겠습니다.", name="example_assistant"),
    AIMessage("", name="example_assistant", tool_calls=[{"name": "search_menu", "args": {"query": "트러플 리조또"}, "id": "1"}]),
    ToolMessage("트러플 리조또: 가격 ₩28,000, 이탈리아 카나롤리 쌀 사용, 블랙 트러플 향과 파르메산 치즈를 듬뿍 넣어 조리", tool_call_id="1"),
    AIMessage("트러플 리조또의 가격은 ₩28,000이며, 이탈리아 카나롤리 쌀을 사용하고 블랙 트러플 향과 파르메산 치즈를 듬뿍 넣어 조리합니다. 이제 추가 정보를 위키피디아에서 찾아보겠습니다.", name="example_assistant"),
    AIMessage("", name="example_assistant", tool_calls=[{"name": "wiki_summary", "args": {"query": "트러플 리조또", "k": 1}, "id": "2"}]),
    ToolMessage("트러플 리조또는 이탈리아 요리의 대표적인 리조또 요리 중 하나로, 고급 식재료인 트러플을 사용하여 만든 크리미한 쌀 요리입니다. 주로 아르보리오나 카나롤리 등의 쌀을 사용하며, 트러플 오일이나 생 트러플을 넣어 조리합니다. 리조또 특유의 크리미한 질감과 트러플의 강렬하고 독특한 향이 조화를 이루는 것이 특징입니다.", tool_call_id="2"),
    AIMessage("트러플 리조또의 특징에 대해 알아보았습니다. 이제 어울리는 와인을 검색해보겠습니다.", name="example_assistant"),
    AIMessage("", name="example_assistant", tool_calls=[{"name": "search_wine", "args": {"query": "트러플 리조또에 어울리는 와인"}, "id": "3"}]),
    ToolMessage("트러플 리조또와 잘 어울리는 와인으로는 주로 중간 바디의 화이트 와인이 추천됩니다. 1. 샤르도네: 버터와 오크향이 트러플의 풍미를 보완합니다. 2. 피노 그리지오: 산뜻한 산미가 리조또의 크리미함과 균형을 이룹니다. 3. 베르나차: 이탈리아 토스카나 지방의 화이트 와인으로, 미네랄리티가 트러플과 잘 어울립니다.", tool_call_id="3"),
    AIMessage("트러플 리조또(₩28,000)는 이탈리아의 대표적인 리조또 요리 중 하나로, 이탈리아 카나롤리 쌀을 사용하고 블랙 트러플 향과 파르메산 치즈를 듬뿍 넣어 조리합니다. 주요 특징으로는 크리미한 질감과 트러플의 강렬하고 독특한 향이 조화를 이루는 점입니다. 고급 식재료인 트러플을 사용해 풍부한 맛과 향을 내며, 주로 아르보리오나 카나롤리 등의 쌀을 사용합니다. 트러플 리조또와 잘 어울리는 와인으로는 중간 바디의 화이트 와인이 추천됩니다. 특히 버터와 오크향이 트러플의 풍미를 보완하는 샤르도네, 산뜻한 산미로 리조또의 크리미함과 균형을 이루는 피노 그리지오, 그리고 미네랄리티가 트러플과 잘 어울리는 이탈리아 토스카나 지방의 베르나차 등이 좋은 선택이 될 수 있습니다.", name="example_assistant"),
]

few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", system + f"Today's date is {today}."),
    *examples,
    ("human", "{user_input}"),
    ("placeholder", "{messages}"),
])

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# 검색 도구를 직접 LLM에 바인딩 가능
llm_with_tools = llm.bind_tools(tools=tools)

# Few-shot 프롬프트를 사용한 체인 구성
fewshot_search_chain = few_shot_prompt | llm_with_tools

# 도구 실행 체인 정의
@chain
def restaurant_menu_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)

    tool_msgs = []
    for tool_call in ai_msg.tool_calls:
        print(f"{tool_call['name']}: \n{tool_call}")
        print("-"*100)

        if tool_call["name"] == "tavily_search_results_json":
            tool_message = tavily_search.invoke(tool_call, config=config)
            tool_msgs.append(tool_message)

        elif tool_call["name"] == "wiki_summary":
            tool_message = wiki_summary_search.invoke(tool_call, config=config)
            tool_msgs.append(tool_message)

        elif tool_call["name"] == "wine_search":
            tool_message = wine_search.invoke(tool_call, config=config)
            tool_msgs.append(tool_message)

        elif tool_call["name"] == "menu_search":
            tool_message = menu_search.invoke(tool_call, config=config)
            tool_msgs.append(tool_message)

    print("tool_msgs: \n", tool_msgs)
    print("-"*100)
    return fewshot_search_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)