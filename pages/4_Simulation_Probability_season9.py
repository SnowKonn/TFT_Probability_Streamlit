import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import string
import random
import time
import streamlit as st


# Immutable Global Variable
MAX_COST = 5

# parameter setting
competitive_parameter = 0
lv_set = 5
others_lv_set = 4
find_info_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
competitors_n = 0
dead_pl_n = 7
no_head_rate = 0.85


def set_find_info_dict(cost: int, n: int, init=False):
    if init:
        for i in range(1, 6):
            find_info_dict[i] = 0
        find_info_dict[cost] = n
        temp_find_info_dict = find_info_dict.copy()
    else:
        find_info_dict[cost] += n
        temp_find_info_dict = find_info_dict.copy()
    return temp_find_info_dict


def get_find_info_list(target_info_dict: dict):
    result_list = []
    for i in range(1, 6):
        n = target_dict[i]
        for j in range(n):
            result_list.append(str(i) + string.ascii_lowercase[j])
    return result_list


def initialize_peices_bag():
    pieces_bag = {}
    key_cost_list = [1, 2, 3, 4, 5]
    name_list = [string.ascii_lowercase[i] for i in range(13)]

    for k in key_cost_list:
        pieces_bag[k] = \
            list(map(lambda x: str(k) + x, name_list[:max_pieces_chart.iloc[1, k - 1]])) * \
            max_pieces_chart.iloc[0, k - 1]

    return pieces_bag


def check_terminal_condition(player_list, target_list, target_class_n, target_pieces_n):
    if len(target_list) < target_class_n:
        print("Wrong Target class")
        return []
    total_satisfaction_n = 0
    if type(target_pieces_n) == int:
        for i in range(len(target_list)):
            i_result = int(sum(map(lambda x: x in target_list[i], player_list)) >= target_pieces_n)
            total_satisfaction_n += i_result
        if total_satisfaction_n < target_class_n:
            return False
        else:
            return True
    elif type(target_pieces_n) == list:
        for i in range(len(target_list)):
            i_result = int(sum(map(lambda x: x in target_list[i], player_list)) >= target_pieces_n[i])
            total_satisfaction_n += i_result
        if total_satisfaction_n < target_class_n:
            return False
        else:
            return True


def check_indiv_condition(player_list, target_element, target_pieces_n):

    i_result = int(sum(map(lambda x: x in [target_element], player_list)) >= target_pieces_n)

    if i_result:
        return True
    else:
        return False


if __name__ == "__main__":


    st.set_page_config(
        page_title="Roll Expectation",
        page_icon="👋",
    )

    st.write("# How many times do I re-roll? :revolving_hearts:")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        How many rolls do you need to acquire the specific pieces? \n
        Use the information below to find out the expected number of re-rolls. \n
        Good luck, everyone!:star: """
    )
    st.divider()

    max_pieces_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Max N', index_col=0)
    probability_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Probability Cost', index_col=0)
    probability_chart.index.name = 'Level'
    probability_chart.columns.name = 'Cost'

    np.dot(max_pieces_chart.iloc[0, :], max_pieces_chart.iloc[1, :])
    player_list = [i for i in range(8)]

    st.write('- Please set lobby info')

    col1, col2 = st.columns([1, 1])

    with col1:
        already_sold_ratio_option = st.selectbox(
            "N people same comp",
            options=(0, 1, 2, 3),
        )
        dead_option = st.selectbox(
            "N of the dead players",
            (0, 1, 2, 3, 4, 5, 6),
            key='dead_players',
            index=0
        )

    with col2:
        lv_option = st.selectbox(
            "Select Your Lv",
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
            key='Level',
            index=6
        )
        lv_option_others = st.selectbox(
            "Other player's average Lv",
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
            key='Level_others',
            index=6
        )
        av_field_cost_option_others = st.selectbox(
            "Average field Cost on the stage",
            (10, 20, 30, 40, 50, 60, 70),
            key='Field_cost',
            index=2
        )

    st.write('- Insert the number of kinds of champions you need')

    finding_n_option = st.selectbox(
        "N Types of Champions",
        (1, 2, 3),
        key='finding_n',
        index=0
    )
    st.write('- Please specify the cost and quantity of each required champion')
    col2_1, col2_2 = st.columns([1, 1])
    a_cost_list = []
    a_n_pieces_list = []

    with col2_1:
        cost_option1 = st.selectbox(
            "Cost of the piece",
            (1, 2, 3, 4, 5),
            key='cost1',
            index=3
        )

    with col2_2:
        finding_n1 = st.selectbox(
            "How many do you want?",
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
            key='finding_n1',
            index=0
        )
        a_cost_list.append(cost_option1)
        a_n_pieces_list.append(finding_n1)

    if finding_n_option > 1:
        col3_1, col3_2 = st.columns([1, 1])

        with col3_1:
            cost_option2 = st.selectbox(
                "Cost of the piece",
                (1, 2, 3, 4, 5),
                key='cost2',
                index=3
            )

        with col3_2:
            finding_n2 = st.selectbox(
                "How many do you want?",
                (1, 2, 3, 4, 5, 6, 7, 8, 9),
                key='finding_n2',
                index=0
            )
            a_cost_list.append(cost_option2)
            a_n_pieces_list.append(finding_n2)

    if finding_n_option > 2:
        col4_1, col4_2 = st.columns([1, 1])

        with col4_1:
            cost_option3 = st.selectbox(
                "Cost of the piece",
                (1, 2, 3, 4, 5),
                key='cost3',
                index=3
            )

        with col4_2:
            finding_n3 = st.selectbox(
                "How many do you want?",
                (1, 2, 3, 4, 5, 6, 7, 8, 9),
                key='finding_n3',
                index=0
            )
            a_cost_list.append(cost_option3)
            a_n_pieces_list.append(finding_n3)

    st.write("- How many of the conditions right above(champion specific option) should be met?")
    fin_condition_option = st.selectbox(
        "N of terminal condition",
        (k for k in range(1, finding_n_option+1)),
        key='fin_condition',
        index=0
    )

    # max_pieces_chart

    lv_set = lv_option
    others_lv_set = lv_option_others

    avg_field_room_cost = av_field_cost_option_others
    competitors_n = already_sold_ratio_option
    dead_pl_n = dead_option
    no_head_rate = 0.9


    if st.button('Run'):
            

        simul_n = 300
        target_info_df = pd.DataFrame([a_cost_list, a_n_pieces_list], index=['Cost', 'N']).T.sort_values(by='Cost')
        a_cost_list = target_info_df['Cost'].tolist()
        a_n_pieces_list = target_info_df['N'].tolist()
        for q in range(len(a_cost_list)):
            if q == 0:
                target_dict = set_find_info_dict(cost=a_cost_list[q], n=1, init=True)
            else:
                target_dict = set_find_info_dict(cost=a_cost_list[q], n=1, init=False)
        target_list = get_find_info_list(target_dict)
        champ_specific_condition = dict(zip(target_list, a_n_pieces_list))
        # st.write(target_list)

        result_cost_list_total = []
        # Set the other player's condition
        for iq in range(simul_n):

            lv_bag = {}
            pl_bag = {}
            for k in probability_chart.index:
                k = int(k)
                setted_lv_sample = []
                for j in range(MAX_COST):
                    setted_lv_sample = setted_lv_sample + ([j + 1] * int(probability_chart.iloc[k - 1, j] * 100))

                lv_bag[k] = setted_lv_sample.copy()

            exhast_all_cost = 5
            setted_lv_sample = list(filter(lambda x: x != exhast_all_cost, setted_lv_sample))
            pieces_bag = initialize_peices_bag()

            for i in range(1, 8):
                pl_bag[i] = []
                i_th_field_room_cost = 0
                if i <= competitors_n:
                    while i_th_field_room_cost < avg_field_room_cost:
                        temp_peice_cost = random.sample(lv_bag[others_lv_set], 1)[0]
                        selected_peices = random.sample(pieces_bag[temp_peice_cost], 1)[0]
                        if np.random.binomial(1, no_head_rate):
                            if selected_peices not in target_list:
                                continue
                            else:
                                pl_bag[i].append(selected_peices)
                                pieces_bag[temp_peice_cost].remove(selected_peices)
                                i_th_field_room_cost += temp_peice_cost
                        else:
                            pl_bag[i].append(selected_peices)
                            pieces_bag[temp_peice_cost].remove(selected_peices)
                            i_th_field_room_cost += temp_peice_cost
                else:
                    if i < 8 - dead_pl_n:
                        while i_th_field_room_cost < avg_field_room_cost:
                            temp_peice_cost = random.sample(lv_bag[others_lv_set], 1)[0]
                            selected_peices = random.sample(pieces_bag[temp_peice_cost], 1)[0]

                            pl_bag[i].append(selected_peices)
                            pieces_bag[temp_peice_cost].remove(selected_peices)
                            i_th_field_room_cost += temp_peice_cost

            total_cost = 0
            i_ = 0
            pl_bag[i_] = []
            terminal_condition = False

            while (total_cost < 200) and (not terminal_condition):

                lv_list_j = random.sample(lv_bag[lv_set], 5)

                for p in lv_list_j:
                    if len(pieces_bag[p]) == 0:
                        exhast_all_cost = p
                        lv_bag[lv_set] = list(filter(lambda x: x != exhast_all_cost, lv_bag[lv_set]))
                    p = random.choice(lv_bag[lv_set])
                    # print(pieces_bag)
                    # print(p)
                    # print(pl_bag)
                    p_th_pieces = random.choice(pieces_bag[p])

                    if p_th_pieces in target_list:
                    
                        if check_indiv_condition(pl_bag[i_], p_th_pieces, champ_specific_condition[p_th_pieces]):
                            pass
                        else:
                            pl_bag[i_].append(p_th_pieces)
                            pieces_bag[p].remove(p_th_pieces)
                            # total_cost += p
                total_cost += 1

                terminal_condition = check_terminal_condition(pl_bag[i_], target_list, target_class_n=fin_condition_option,
                                                            target_pieces_n=a_n_pieces_list)


            result_cost_list_total.append(total_cost)

        result_array = np.array(result_cost_list_total)
        st.write("Avg:", round(np.mean(result_array), 2))
        st.write("Std:", round(np.std(result_array), 2))


        fig, ax = plt.subplots()
        result_array.sort()
        result_df = pd.DataFrame(np.unique(result_array, return_counts=True), ['re_roll', 'count']).T
        result_df['density_prob'] = result_df['count']/ sum(result_df['count'])
        result_df['cum_prob'] = result_df['density_prob'].cumsum()

        diff_table = abs(result_df['re_roll'] - np.mean(result_array))
        min_abs_diff_value = min(diff_table)
        min_abs_diff_value_index = diff_table.loc[diff_table == min_abs_diff_value].index[0]
        nearest_values = result_df.loc[min_abs_diff_value_index, 're_roll']

        diff_values = np.mean(result_array) - nearest_values
        diff_sign = int(np.sign(diff_values))

        mean_info_table = result_df.loc[[min_abs_diff_value_index+diff_sign, min_abs_diff_value_index]][['re_roll', 'cum_prob']].sort_index()

        # 주어진 두 포인트 p1과 p2
        p1 = mean_info_table.iloc[0].values
        p2 = mean_info_table.iloc[1].values

        # p3의 X 값
        x_p3 = np.mean(result_array)

        # p1과 p2를 이용하여 기울기 계산
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])

        # p3의 Y 값을 예측
        y_p3 = p1[1] + (x_p3 - p1[0]) * slope

        mean_coordinate = (x_p3, y_p3)


        ax.plot(result_df['re_roll'], result_df['cum_prob'], label='Cumulative Probability', linewidth=1, color='b',
                )
        ax.plot(*mean_coordinate, 'bo', label='Average')
        ax.grid(linestyle='--')
        ax2 = ax.twinx()
        ax2.hist(result_array, density=True, label='Probability Mass(Right)', range=[min(result_df['re_roll']), max(result_df['re_roll'])],
                bins=15, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title('Simulation Result')


        # ax.spines[['top']].set_visible(False)
        # ax2.spines[['top']].set_visible(False)

        ax.tick_params(axis='x', direction='in', which='both')
        ax.tick_params(axis='y', direction='in', which='both')
        ax2.tick_params(axis='x', direction='in', which='both')
        ax2.tick_params(axis='y', direction='in', which='both')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.set_xlabel('N re-roll')
        ax.set_ylabel('Cumulative Probabilty')
        ax2.set_ylabel('Probability')
        ax.legend(lines + lines2, labels + labels2, loc=0)
        # st.write(probability_chart.style.pipe(make_pretty), use_container_width=True)

        st.pyplot(fig)
        # st.write(probability_chart.style.pipe(make_pretty), use_container_width=True)

