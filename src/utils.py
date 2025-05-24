from IPython.display import HTML
import uuid

def graph_to_mermaid(graph) -> str:
    """
    주어진 graph 객체를 Mermaid 형식 문자열로 변환합니다.
    LangGraph 0.4.5 버전에 맞춰 작성되었습니다.
    Parameters:
        graph: 노드와 엣지를 포함한 그래프 객체.
               graph.nodes와 graph.edges를 가져올 수 있어야 하며,
               각 edge는 source, target 속성을 가져야 합니다.
    Returns:
        Mermaid 형식 문자열
    """
    nodes = graph.nodes
    edges = graph.edges
    mermaid = ["graph TD"]

    # 조건부 엣지 그룹핑을 위한 딕셔너리
    conditional_groups = {}
    normal_edges = []

    # 엣지를 분류
    for edge in edges:
        source = edge.source
        target = edge.target

        # LangGraph 0.4.5에서 조건부 엣지 확인
        has_condition = False
        condition_name = None

        # 조건부 엣지는 보통 같은 source에서 여러 target으로 가는 패턴
        # 또는 edge.data에 조건 정보가 포함됨
        if hasattr(edge, 'data') and edge.data:
            if isinstance(edge.data, dict):
                # 조건 함수나 조건 정보 확인
                if 'condition' in edge.data and callable(edge.data['condition']):
                    has_condition = True
                    condition_name = getattr(edge.data['condition'], '__name__', 'condition')
                elif 'branch' in edge.data:
                    # branch 정보에서 조건 함수 이름 추출
                    branch_info = edge.data['branch']
                    if callable(branch_info):
                        has_condition = True
                        condition_name = getattr(branch_info, '__name__', 'condition')
                    elif isinstance(branch_info, str):
                        has_condition = True
                        condition_name = branch_info
                # add_additional_edge도 조건부 엣지로 처리
                elif edge.data.get('langgraph_node') == '__interrupt':
                    has_condition = True
                    condition_name = 'additional_condition'

        # edge 자체에 조건 정보가 있는지 확인
        if not has_condition:
            # 같은 source에서 나가는 여러 엣지가 있는지 확인하여 조건부 엣지 추정
            source_edges = [e for e in edges if e.source == source]
            if len(source_edges) > 1:
                # 여러 엣지가 있고 특별한 패턴(예: __end__나 특정 노드로 가는)이 있으면 조건부로 간주
                targets = [e.target for e in source_edges]
                if '__end__' in targets or len(set(targets)) > 1:
                    has_condition = True
                    # 기본 조건 이름을 source 기반으로 추정 (예: call_model -> should_continue)
                    if source == 'call_model':
                        condition_name = 'should_continue'
                    else:
                        condition_name = f"{source}_condition"

        # 조건부 엣지 그룹핑
        if has_condition:
            group_key = f"{source}_{condition_name}"
            if group_key not in conditional_groups:
                conditional_groups[group_key] = {
                    'source': source,
                    'condition_name': condition_name,
                    'targets': []
                }

            # 라벨 정보 추출 (조건 결과 키 값)
            label = None
            if hasattr(edge, 'data') and edge.data and isinstance(edge.data, dict):
                # 조건 매핑에서 키 값을 찾아서 라벨로 사용
                for key, value in edge.data.items():
                    if value == target or (isinstance(value, str) and value == target):
                        label = key
                        break
                # 또는 직접 label 속성이 있는 경우
                if not label:
                    label = edge.data.get('label', None)

            # 특별한 경우 처리 (END 노드)
            if target in ['__end__', 'END'] and not label:
                label = "END"
            elif not label:
                # 기본적으로 타겟 노드 이름을 라벨로 사용
                label = target.replace('__', '').upper() if target.startswith('__') else target

            conditional_groups[group_key]['targets'].append({
                'target': target,
                'label': label
            })
        else:
            normal_edges.append(edge)
    # 헬퍼 함수: 노드 이름 변환
    def get_node_name(node):
        if node == '__start__':
            return 'START'
        elif node == '__end__':
            return 'END'
        return node
    # 일반 노드 추가
    for node in nodes:
        if node not in ['__start__', '__end__']:  # 특수 노드 제외
            mermaid.append(f'{node}["{node}"]')
        else:
            # START, END 노드는 특별한 모양으로
            if node == '__start__':
                mermaid.append(f'START(("START"))')
            elif node == '__end__':
                mermaid.append(f'END(("END"))')

    # 조건부 엣지 노드들 추가
    for group_key, group_info in conditional_groups.items():
        condition_id = f"cond_{group_info['source']}_{group_info['condition_name']}"
        # 조건 함수 이름을 다이아몬드 노드에 표시
        mermaid.append(f'{condition_id}([{group_info["condition_name"]}])')
        mermaid.append(f'classDef conditionalNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px')
        mermaid.append(f'class {condition_id} conditionalNode')

        # 소스 노드에서 조건 노드로
        source_node = get_node_name(group_info['source'])
        mermaid.append(f"{source_node} --> {condition_id}")

        # 조건 노드에서 각 타겟으로 분기 (점선으로 표시)
        for target_info in group_info['targets']:
            target_node = get_node_name(target_info['target'])
            edge_label = target_info['label'] if target_info['label'] else ("END" if target_node == 'END' else target_node)
            mermaid.append(f"{condition_id} -.->|{edge_label}| {target_node}")

    # 일반 엣지 추가
    for edge in normal_edges:
        source = edge.source
        target = edge.target

        # 특수 노드 변환
        source_node = source if source not in ['__start__', '__end__'] else ('START' if source == '__start__' else 'END')
        target_node = target if target not in ['__start__', '__end__'] else ('START' if target == '__start__' else 'END')

        # 라벨 확인
        label = None
        if hasattr(edge, 'data') and edge.data and isinstance(edge.data, dict):
            label = edge.data.get('label', None)

        if label and label != target:
            mermaid.append(f"{source_node} -->|{label}| {target_node}")
        else:
            mermaid.append(f"{source_node} --> {target_node}")

    return "\n".join(mermaid)

def simple_mermaid_render(mermaid_code):
    """외부 패키지 없이 Mermaid 렌더링"""
    chart_id = f"chart-{uuid.uuid4().hex[:8]}"
    html = f"""
    <div id="{chart_id}" class="mermaid">
    {mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true
            }}
        }});
        mermaid.init(undefined, document.getElementById('{chart_id}'));
    </script>
    """
    return HTML(html)

def render_graph(graph):
    """
    그래프 객체를 받아 Mermaid로 렌더링합니다.
    """
    mermaid_code = graph_to_mermaid(graph)
    return simple_mermaid_render(mermaid_code)
