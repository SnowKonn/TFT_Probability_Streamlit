import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 네트워크 그래프를 생성하는 함수
def create_network_graph_from_df(dataframe):
    G = nx.Graph()
    for _, row in dataframe.iterrows():
        G.add_node(row['name'], type='champion', cost=row['cost'])
        for trait in ['Trait1', 'Trait2', 'Trait3']:
            if pd.notna(row[trait]) and row[trait]:
                G.add_node(row[trait], type='trait')
                G.add_edge(row['name'], row[trait])
    return G

# 챔피언 코스트에 따른 색상을 지정하는 함수
def generate_node_colors_by_cost(G, cost_colors, trait_color):
    node_colors = []
    for node, data in G.nodes(data=True):
        if data['type'] == 'champion':
            node_colors.append(cost_colors[data['cost']])
        else:
            node_colors.append(trait_color)
    return node_colors

# 엑셀 파일로부터 데이터를 읽어들이는 부분
excel_file_path = 'Data/Champ_info_season10.xlsx'
df_champ_info = pd.read_excel(excel_file_path, sheet_name='champ_info')
df_champ_info = df_champ_info.drop(columns=['id'])

# 챔피언 코스트에 따른 색상 지정
cost_colors = {
    1: 'gray',
    2: 'green',
    3: 'blue',
    4: 'purple',
    5: 'yellow',
}

# 특성 색상 지정
trait_color = 'orange'

# 네트워크 그래프 생성 및 색상 지정
G = create_network_graph_from_df(df_champ_info)
node_colors = generate_node_colors_by_cost(G, cost_colors, trait_color)

# 그룹 지정
champion_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'champion']
trait_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'trait']


# 네트워크 그래프 시각화 설정
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.3, iterations=100)
nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight='normal', node_size=500, font_size=12)
plt.title('Improved TFT Champion and Trait Synergy Network with Cost Colors', size=20)
plt.show()

# 특정 챔피언 또는 특성 이름
source_node = 'Amumu'
k = 3  # 최단 거리의 최대 값 (원하는 값으로 변경)

# 그룹 지정
edgelord_neighbors_distance_3 = list(nx.single_source_shortest_path_length(G, source_node, cutoff=k).keys())

# 이 노드들만 포함하는 서브그래프를 생성합니다.
edgelord_subgraph_distance_3 = G.subgraph(edgelord_neighbors_distance_3)

# 서브그래프의 노드 색상을 지정합니다. (최단 거리가 3 이하인 노드만 포함)
subgraph_node_colors_distance_3 = generate_node_colors_by_cost(edgelord_subgraph_distance_3, cost_colors, trait_color)

# 서브그래프를 시각화합니다.
plt.figure(figsize=(12, 8))
subgraph_pos_distance_3 = nx.spring_layout(edgelord_subgraph_distance_3, k=0.4, iterations=50)
nx.draw(edgelord_subgraph_distance_3, subgraph_pos_distance_3, with_labels=True, node_color=subgraph_node_colors_distance_3, font_weight='normal', node_size=1000, font_size=11)
plt.title(f'Subgraph of Units within Shortest Path Distance {k} from "{source_node}"', size=15)
plt.show()
