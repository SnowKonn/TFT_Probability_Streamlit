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
    'props': 'background-color: #F1F4FF; color: black; ; font-size:16px; width: 100%;'
}


def df_stat_style(styler):
    styler.format(formatter=lambda x: x if type(x) == str \
        else '{:.1f}%'.format(x * 100) if x < 1 \
            else '{:.1f}'.format(x))
    styler.set_table_styles([table_align, cell_hover, index_names, headers] + font_size_setting)

    return styler

if __name__ == "__main__":

    augments_df = pd.DataFrame(
        [
            ['P', 'P', 'P', 0.01],
            ['P', 'P', 'G', 0.01],
            ['P', 'G', 'P', 0.01],
            ['P', 'G', 'G', 0.02],
            ['P', 'S', 'P', 0.01],
            ['P', 'S', 'G', 0.04],
            ['G', 'P', 'P', 0.01],
            ['G', 'P', 'G', 0.1],
            ['G', 'P', 'S', 0.06],
            ['G', 'G', 'P', 0.03],
            ['G', 'G', 'G', 0.22],
            ['G', 'S', 'P', 0.02],
            ['G', 'S', 'G', 0.18],
            ['S', 'P', 'P', 0.01],
            ['S', 'G', 'P', 0.05],
            ['S', 'G', 'G', 0.12],
            ['S', 'S', 'P', 0.05],
            ['S', 'S', 'G', 0.05],
        ], columns=['2-1', '3-2', '4-2', 'probability']
    )


    st.set_page_config(
        page_title="What is the color of next augments?",
        page_icon="👋",
    )

    st.write("# What is the color of the next Augments?")
    st.sidebar.success("Select a demo above.")

    st.divider()
    # st.dataframe(augments_df.style.pipe(df_stat_style), width=700)

    max_pieces_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Max N', index_col=0)
    probability_chart = pd.read_excel('Data/Probability Chart.xlsx', 'Probability Cost', index_col=0)
    probability_chart.index.name = 'Level'
    probability_chart.columns.name = 'Cost'

    np.dot(max_pieces_chart.iloc[0, :], max_pieces_chart.iloc[1, :])
    player_list = [i for i in range(8)]

    stage_option = st.selectbox(
        "Which stage for augmentation color?",
        options=('2-1', '3-2', '4-2'),
    )

    if stage_option == '2-1':
        st.dataframe(
            augments_df.groupby(by='2-1').sum().loc[:, ['probability']].style.pipe(df_stat_style),
            width=200)

    if stage_option == '3-2':
        prev_option = st.radio(
            "Select the color at 2-1",
            ['P', 'G', 'S'],
            horizontal=True
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            conditional_prob_df = augments_df.loc[augments_df['2-1'] == prev_option]
            conditional_prob_df['probability'] = conditional_prob_df['probability']/sum(conditional_prob_df['probability'])
            st.dataframe(
                conditional_prob_df.loc[augments_df['2-1']==prev_option].groupby(by='3-2').sum().loc[:, ['probability']].style.pipe(df_stat_style),
                width=300)
        with col2:
            conditional_prob_df = augments_df.loc[augments_df['2-1'] == prev_option]
            conditional_prob_df['probability'] = conditional_prob_df['probability']/sum(conditional_prob_df['probability'])
            st.dataframe(
                conditional_prob_df.groupby(by='4-2').sum().loc[:, ['probability']].style.pipe(df_stat_style),
                width=300)
        st.write('- Probability of 4-2 augmentation color when only 2-1 augmentation info is available')

    if stage_option == '4-2':
        prev_option_1 = st.radio(
            "Select the color at 2-1",
            ['P', 'G', 'S'],
            horizontal=True
        )
        prev_option_2 = st.radio(
            "Select the color at 3-2",
            ['P', 'G', 'S'],
            horizontal=True
        )

        conditional_prob_df = augments_df.loc[augments_df['2-1'] == prev_option_1]
        conditional_prob_df = conditional_prob_df.loc[augments_df['3-2'] == prev_option_2]
        conditional_prob_df['probability'] = conditional_prob_df['probability']/sum(conditional_prob_df['probability'])
        st.dataframe(
            conditional_prob_df.groupby(by='4-2').sum().loc[:, ['probability']].style.pipe(df_stat_style),
            width=300)
