import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
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
        ax.plot( *mean_coordinate, 'bo', label='Average')
    ax.legend()

    # print(f'probability {np.round(p_t * 100, 2)}%')
    # print(f'median {np.round(geo_median, 1)}')
    # print(f'mean {np.round(geo_mean, 1)}')
    # print(f'std {np.round(geo_std, 1)}')
    stats = {'P': probability_target, 'Median': geo_median, 'Mean': geo_mean, 'Std': geo_std}

    return probability_target, fig, ax, stats


if __name__ == "__main__":

    st.set_page_config(
        page_title="Roll Expectation",
        page_icon="👋",
    )

    st.write("# How many times do I re-roll? :revolving_hearts:")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        How many rolls do you need to acquire the specific champions? \n
        Use the information below to find out the expected number of re-rolls. \n
        Good luck, everyone!:star: """
    )
    st.divider()
    max_pieces_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Max N', index_col=0)
    probability_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Probability Cost', index_col=0)
    probability_chart.index.name = 'Level'
    probability_chart.columns.name = 'Cost'
    # st.write(probability_chart.style.pipe(make_pretty), use_container_width=True)

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False


    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        already_sold_ratio_option = st.radio(
            "N people same comp",
            options=(0, 1, 2),
        )

    with col2:
        lv_option = st.selectbox(
            "Select Your Lv",
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
            key='Level',
            index=6
        )

    with col3:
        cost_option = st.selectbox(
            "Cost of the piece",
            (1, 2, 3, 4, 5),
            key='cost',
            index=3
        )

    with col4:
        finding_n_option = st.selectbox(
            "N Types of champions",
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
            key='finding_n',
            index=2
        )

    # max_pieces_chart
    lv = lv_option
    piece_cost = cost_option
    finding_n = finding_n_option
    already_sold_ratio = 0.15 * float(already_sold_ratio_option)
    probability_target, fig, ax, stats = cal_prob_target(lv, piece_cost, finding_n, already_sold_ratio, n_slot=5)

    st.markdown(
        f"""
        - The probability of the desired piece appearing in the shop is **{np.round(stats['P']*100, 1)}%**.
        - On average, you need to re-roll **{np.round(stats['Mean'],1)}** times. 
        - With 50% likelihood, you need to reroll **{np.round(stats['Median'],1)}** to get the champions.
        - If you've re-rolled more than **{np.round(stats['Mean'] + stats['Std'],1)}** and still haven't gotten it, that's unlucky.
        """
                )
    st.dataframe(pd.DataFrame([stats], index=['Stats']).style.pipe(df_stat_style), width=700)
    st.write('\n')
    st.pyplot(fig)

    st.divider()

    st.markdown(
        """
        ### Input variables
        
        - **N people same comp**: Indicates the degree to which you overlap with other players. \n
            - Computationally, it's assumed that 15% of the total pieces are already drawn for each person you overlap with.
        - **LV**: Your level when rerolling
        - **Cost**: The cost of the piece you want to draw
        - **N types of pieces**: The number of different types of pieces you need. \n
            - Example: If you need both Ryze or Sion, It is 2.
        \n \n
        """
    )

    st.divider()
    st.markdown("""
    The probabilities above are calculated based on the geometric distribution. 
    They are computed assuming the geometric distribution for the 'probability of obtaining at least one specific piece after N re-rolls.' 
    Please keep this in mind. If you'd like to know more details, please refer to the simulation page.
    """)


