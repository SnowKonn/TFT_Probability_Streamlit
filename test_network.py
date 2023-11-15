import pandas as pd
import networkx as nx
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
excel_file_path = '/mnt/data/Champ_info_season10.xlsx'
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

# 네트워크 그래프 시각화 설정
plt.figure(figsize=(20, 15))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight='bold', node_size=1000, font_size=14)
plt.title('Improved TFT Champion and Trait Synergy Network with Cost Colors', size=20)
plt.show()
