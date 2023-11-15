import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



target_champ = 'Karthus'
target_trait = 'Pentakill'
cost_constraint = 5
players_lv = 9
champ_included = ['Thresh', 'Yorick', target_champ]

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

# 챔피언 정보를 포함한 데이터프레임을 로드합니다.
df_champ_info = pd.read_excel('Data/Champ_info_season10.xlsx', sheet_name='champ_info')
df_champ_info = df_champ_info.drop(columns=['id'])

# 데이터프레임으로부터 그래프를 생성합니다.
G = create_network_graph_from_df(df_champ_info)


# 데이터를 로드합니다.
trait_activation_df = pd.read_excel('Data/Champ_info_season10.xlsx', sheet_name='trait_info')
trait_activation_df = trait_activation_df.drop(columns=['id'])  # 'id' 열 제외

# 각 시너지의 점수를 나타내는 데이터프레임을 생성합니다.
trait_score_df = trait_activation_df.copy()
for level in ['activate_level1', 'activate_level2', 'activate_level3', 'activate_level4']:
    trait_score_df[level] = pd.to_numeric(trait_score_df[level], errors='coerce') * 2

trait_score_df.iloc[:, 1:] = trait_score_df.iloc[:, 1:].cumsum(axis=1).fillna(0)

# 원래의 갯수 데이터를 유지하는 trait_activation_df에서는 숫자만 남기고 나머지를 제거합니다.
columns_to_keep = ['Trait name'] + [col for col in trait_activation_df.columns if 'activate_level' in col]
trait_activation_df = trait_activation_df[columns_to_keep]

# 시너지 점수 계산에 사용할 조합의 수를 줄이기 위해 Yone과 연결된 챔피언 중 5코스트 미만인 챔피언들만을 대상으로 합니다.
target_champ_combinations = [node for node, distance in nx.single_source_shortest_path_length(G, target_champ, cutoff=4).items()
                     if node not in champ_included and G.nodes[node]['type'] == 'champion' and G.nodes[node]['cost'] <= cost_constraint]

# 조합을 생성할 때 일부 챔피언들을을 포함합니다.
all_combinations = [tuple(champ_included) + combo for combo in combinations(target_champ_combinations, players_lv-len(champ_included))]

# 시너지 점수 계산 함수를 정의합니다.
def calculate_synergy_score(combination, traits_activation, traits_score, target_champ, target_trait):
    # 조합의 시너지를 카운트합니다.
    trait_count = {trait: 0 for trait in traits_score.keys()}
    # 챔피언 조합의 코스트 합계를 계산합니다.
    total_cost = 0
    for champ in combination:
        # G[champ]를 통해 해당 챔피언에 연결된 특성들을 얻습니다.
        connected_traits = G[champ]
        # 챔피언의 코스트를 합산합니다.
        total_cost += G.nodes[champ]['cost']
        for trait in connected_traits:
            # 특성 타입이 'trait'인 노드만 카운트합니다.
            if G.nodes[trait]['type'] == 'trait':
                # 대상 챔피언의 대상 특성이면 2를 추가합니다.
                if champ == target_champ and trait == target_trait:
                    trait_count[trait] += 2
                else:
                    trait_count[trait] += 1

    # 활성화된 시너지의 점수를 계산합니다.
    total_score = 0
    for trait, count in trait_count.items():
        # 가장 높은 활성화된 시너지 레벨부터 확인합니다.
        for level in sorted(traits_activation[trait], reverse=True):
            if count >= traits_activation[trait][level]:
                # 해당 시너지 레벨의 점수를 추가합니다.
                total_score += traits_score[trait][level]
                break  # 가장 높은 활성화된 시너지 레벨의 점수만 추가합니다.
    # 챔피언 조합의 코스트 합계를 시너지 점수에 추가합니다.
    total_score += total_cost
    return total_score

# 시너지 점수 계산에 사용할 딕셔너리를 만듭니다.
traits_activation = trait_activation_df.set_index('Trait name').to_dict('index')
traits_score = trait_score_df.set_index('Trait name').to_dict('index')

# 각 조합의 시너지 점수를 계산하고 상위 5개를 반환합니다.
combination_scores = [
    (combo, calculate_synergy_score(combo, traits_activation, traits_score, target_champ, target_trait))
    for combo in all_combinations
]
top_5_combinations = sorted(combination_scores, key=lambda x: x[1], reverse=True)[:5]

comb_to_graph = top_5_combinations[4][0]
nodes_to_include = set(comb_to_graph)
for champ in comb_to_graph:
    # 각 챔피언에 연결된 특성을 추가합니다.
    traits = [n for n in G[champ] if G.nodes[n]['type'] == 'trait']
    nodes_to_include.update(traits)


# 챔피언 코스트에 따른 색상을 지정하는 함수
def generate_node_colors_by_cost(G, cost_colors, trait_color):
    node_colors = []
    for node, data in G.nodes(data=True):
        if data['type'] == 'champion':
            node_colors.append(cost_colors[data['cost']])
        else:
            node_colors.append(trait_color)
    return node_colors


# 챔피언 코스트에 따른 색상 지정
cost_colors = {
    1: 'gray',
    2: 'green',
    3: 'lightblue',
    4: 'purple',
    5: 'yellow',
}


# 특성 색상 지정
trait_color = 'orange'

best_comp_graph = G.subgraph(nodes_to_include)

subgraph_node_colors_distance_3 = generate_node_colors_by_cost(best_comp_graph, cost_colors, trait_color)


# 서브그래프를 시각화합니다.
plt.figure(figsize=(12, 8))
pos_best_comp = nx.spring_layout(best_comp_graph, k=0.4, iterations=50)
nx.draw(best_comp_graph, pos_best_comp, with_labels=True, node_color=subgraph_node_colors_distance_3, font_weight='normal', node_size=1000, font_size=11)
plt.title(f'Graph of Best Comp Units', size=15)
plt.show()
