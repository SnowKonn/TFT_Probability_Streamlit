import pandas as pd
import networkx as nx
from itertools import combinations


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
    trait_score_df[level] = pd.to_numeric(trait_score_df[level], errors='coerce').fillna(0) * 3

# 원래의 갯수 데이터를 유지하는 trait_activation_df에서는 숫자만 남기고 나머지를 제거합니다.
columns_to_keep = ['Trait name'] + [col for col in trait_activation_df.columns if 'activate_level' in col]
trait_activation_df = trait_activation_df[columns_to_keep]

# 시너지 점수 계산에 사용할 조합의 수를 줄이기 위해 Yone과 연결된 챔피언 중 5코스트 미만인 챔피언들만을 대상으로 합니다.
missfortune_combinations = [node for node, distance in nx.single_source_shortest_path_length(G, 'Miss Fortune', cutoff=4).items() 
                     if node != 'Miss Fortune' and G.nodes[node]['type'] == 'champion' and G.nodes[node]['cost'] < 6]

# 조합을 생성할 때 Yone을 포함합니다.
all_combinations = [('Miss Fortune',) + combo for combo in combinations(missfortune_combinations, 6)]

# 시너지 점수 계산 함수를 정의합니다.
def calculate_synergy_score(combination, traits_activation, traits_score):
    # 조합의 시너지를 카운트합니다.
    trait_count = {trait: 0 for trait in traits_score.keys()}
    for champ in combination:
        # G[champ]를 통해 해당 챔피언에 연결된 특성들을 얻습니다.
        connected_traits = G[champ]
        for trait in connected_traits:
            # 특성 타입이 'trait'인 노드만 카운트합니다.
            if G.nodes[trait]['type'] == 'trait':
                trait_count[trait] += 1

    # 활성화된 시너지의 점수를 계산합니다.
    total_score = 0
    for trait, count in trait_count.items():
        if trait in traits_activation:
            trait_info = traits_activation[trait]
            for level, required in trait_info.items():
                if count >= required:
                    total_score += traits_score[trait][level]
                    break  # Only the highest active level counts.
    return total_score

# 시너지 점수 계산에 사용할 딕셔너리를 만듭니다.
traits_activation = trait_activation_df.set_index('Trait name').to_dict('index')
traits_score = trait_score_df.set_index('Trait name').to_dict('index')

# 각 조합의 시너지 점수를 계산하고 상위 5개를 반환합니다.
combination_scores = [(combo, calculate_synergy_score(combo, traits_activation, traits_score)) for combo in all_combinations]
top_5_combinations = sorted(combination_scores, key=lambda x: x[1], reverse=True)[:10]
print(top_5_combinations)
