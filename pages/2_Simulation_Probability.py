import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import string
import random
import time
import streamlit as st

heading_properties = [('font-size', '16px')]

cell_properties = [('font-size', '16px')]

font_size_setting = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

table_align = {  # for row hover use <tr> instead of <td>
    'selector': 'table',
    'props': [('align-items', 'center')]
}
cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#43ACA7'), ('width', '10%')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal; '
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #F1F4FF; color: black; ; font-size:16px; width: 20%;'
}


def make_pretty(styler):
    styler.format(formatter=lambda x: '{:.0f}%'.format(x * 100))
    styler.background_gradient(cmap='Blues', axis=1)
    styler.set_table_styles([table_align, cell_hover, index_names, headers] + font_size_setting)

    return styler


def df_stat_style(styler):
    styler.format(formatter=lambda x: '{:.1f}%'.format(x * 100) if x < 1 else '{:.1f}'.format(x))
    styler.set_table_styles([table_align, cell_hover, index_names, headers] + font_size_setting)

    return styler

def cal_for_float(p_t, j):
    p_temp = ((1-p_t) ** j )*p_t
    return p_temp

def prob_cal(p_t, i):
    p_temp = 0
    for j in range(i):
        p_temp = p_temp + ((1-p_t) ** j)*p_t
    return p_temp


def cal_prob_target(lv, piece_cost, finding_n, already_sold_ratio, n_slot=5):
    n_class = max_pieces_chart.loc['Piece Class N', piece_cost]

    prob_unit = probability_chart.loc[lv, piece_cost]
    finding_ratio = finding_n / n_class
    the_other_cost_probability = sum(probability_chart.loc[lv, probability_chart.columns.difference([piece_cost])])

    probability_target = 1 - (prob_unit * (
                1 - finding_ratio) + finding_ratio * already_sold_ratio * prob_unit + the_other_cost_probability) ** n_slot

    p_t = probability_target
    probability_list = []
    p_temp = 0
    for i in range(40):
        #     print(f'{np.round(p_temp*100,2)}%')
        probability_list.append(p_temp)
        p_temp = p_temp + (1 - p_t) ** (i) * p_t

    fig, ax = plt.subplots()
    # plt.title(f'LV: {lv}, {piece_cost} Cost')
    color_plot = 'k'
    if piece_cost == 1: color_plot = 'k'
    if piece_cost == 2: color_plot = 'g'
    if piece_cost == 3: color_plot = 'b'
    if piece_cost == 4: color_plot = 'purple'
    if piece_cost == 5: color_plot = 'y'

    ax.plot(probability_list, '-.', color=color_plot, label='Probability')

    ax.tick_params(axis='x', direction='in', which='both')
    ax.tick_params(axis='y', direction='in', which='both')
    ax.spines[['top']].set_visible(False)
    ax.spines[['right']].set_visible(False)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 40])
    ax.set_xlabel('N of Reroll')
    ax.set_ylabel('Probability')
    # plt.title('Probability')
    ax.grid(linestyle='--')

    geo_median = -1 / np.log2(1 - p_t)
    geo_mean = 1 / p_t
    geo_std = np.sqrt((1 - p_t) / p_t ** 2)
    if geo_mean < 40:
        mean_coordinate = (int(np.round(geo_mean, 0)), prob_cal(p_t, int(np.round(geo_mean, 0))))
        ax.plot( *mean_coordinate, 'go', label='Mean')
    ax.legend()

    # print(f'probability {np.round(p_t * 100, 2)}%')
    # print(f'median {np.round(geo_median, 1)}')
    # print(f'mean {np.round(geo_mean, 1)}')
    # print(f'std {np.round(geo_std, 1)}')
    stats = {'P': probability_target, 'Median': geo_median, 'Mean': geo_mean, 'Std': geo_std}

    return probability_target, fig, ax, stats


# Show All Costs

@st.cache_resource
def get_plot_target_all_costs(lv, finding_n, already_sold_ratio, n_slot=5):
    fig, ax = plt.subplots()
    for piece_cost in range(1, 6):
        n_class = max_pieces_chart.loc['Piece Class N', piece_cost]

        prob_unit = probability_chart.loc[lv, piece_cost]
        finding_ratio = finding_n / n_class
        the_other_cost_probability = sum(probability_chart.loc[lv, probability_chart.columns.difference([piece_cost])])

        probability_target = 1 - (prob_unit * (
                    1 - finding_ratio) + finding_ratio * already_sold_ratio * prob_unit + the_other_cost_probability) ** n_slot

        p_t = probability_target
        probability_list = []
        p_temp = 0
        for i in range(40):
            #     print(f'{np.round(p_temp*100,2)}%')
            probability_list.append(p_temp)
            p_temp = p_temp + (1 - p_t) ** (i) * p_t

        color_plot = 'k'
        if piece_cost == 1: color_plot = 'gray'
        if piece_cost == 2: color_plot = 'g'
        if piece_cost == 3: color_plot = 'b'
        if piece_cost == 4: color_plot = 'purple'
        if piece_cost == 5: color_plot = 'y'

        ax.plot(probability_list, '-.', color=color_plot)

        ax.tick_params(axis='x', direction='in', which='both')
        ax.tick_params(axis='y', direction='in', which='both')
        ax.spines[['top']].set_visible(False)
        ax.spines[['right']].set_visible(False)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 40])
        ax.grid(linestyle='--')

    return fig, ax


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
        find_info_dict[cost] = n
        temp_find_info_dict = find_info_dict.copy()
    return temp_find_info_dict


def get_find_info_list(target_info_dict: dict):
    result_list = []
    for i in range(1, 6):
        n = target_dict[i]
        for j in range(n):
            result_list.append(str(i) + string.ascii_lowercase[j])
    return result_list


minimum_satisfied_thr = 2


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

    max_pieces_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Max N', index_col=0)
    probability_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Probability Cost', index_col=0)
    probability_chart.index.name = 'Level'
    probability_chart.columns.name = 'Cost'

    np.dot(max_pieces_chart.iloc[0, :], max_pieces_chart.iloc[1, :])
    player_list = [i for i in range(8)]

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

    lv_set = 4
    others_lv_set = 5
    cost = 1
    target_pool_n = 4
    target_pieces_n = 9
    target_class_n = 3

    avg_field_room_cost = 15
    competitors_n = 0
    dead_pl_n = 0
    no_head_rate = 0.9

    simul_n = 100

    target_dict = set_find_info_dict(cost=cost, n=target_pool_n, init=True)
    target_list = get_find_info_list(target_dict)

    result_cost_list_total = []

    for iq in range(simul_n):
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
        i = 0
        pl_bag[i] = []
        terminal_condition = False

        while (total_cost < 300) and (not terminal_condition):

            lv_list_j = random.sample(lv_bag[lv_set], 5)

            for p in lv_list_j:
                if len(pieces_bag[p]) == 0:
                    exhast_all_cost = p
                    lv_bag[lv_set] = list(filter(lambda x: x != exhast_all_cost, lv_bag[lv_set]))
                p = random.choice(lv_bag[lv_set])
                p_th_pieces = random.choice(pieces_bag[p])
                if p_th_pieces in target_list:
                    if check_indiv_condition(pl_bag[i], p_th_pieces, target_pieces_n):
                        pass
                    else:
                        pl_bag[i].append(p_th_pieces)
                        pieces_bag[p].remove(p_th_pieces)
                        # total_cost += p
            total_cost += 1

            terminal_condition = check_terminal_condition(pl_bag[i], target_list, target_class_n=target_class_n,
                                                          target_pieces_n=target_pieces_n)

        result_cost_list_total.append(total_cost)

    result_array = np.array(result_cost_list_total)
    print("Avg:", round(np.mean(result_array), 2))
    print("Std:", round(np.std(result_array), 2))


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

    fig, ax = plt.subplots()
    ax.hist(result_array)
    st.pyplot(fig)
    # st.write(probability_chart.style.pipe(make_pretty), use_container_width=True)

