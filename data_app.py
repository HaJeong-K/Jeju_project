# -*- coding:utf-8 -*-

import streamlit as st
import price_app as price
import peak_app as peak
import weather_app as weather

def run_data_app():
    st.header('분석내용')
    items = ['물가', '성수기', '날씨']
    item = st.sidebar.selectbox('분석항목', items)
    tabs = ['시각화', '통계', '데이터']

    if item == '물가':
        st.subheader('물가')
        tab1, tab2, tab3, tab4 = st.tabs(['머신러닝'] + tabs)
        with tab1:
            price.pred_app()
        with tab2:
            price.vis_app()
        with tab3:
            price.stat_app()
        with tab4:
            price.data_app()
    elif item == '성수기':
        st.subheader('성수기')
        tab1, tab2, tab3 = st.tabs(tabs)
        with tab1:
            peak.vis_app()
        with tab2:
            peak.stat_app()
        with tab3:
            peak.data_app()
    elif item == '날씨':
        st.subheader('날씨')
        tab1, tab2 = st.tabs(['시각화', '데이터'])
        with tab1:
            weather.vis_app()
        with tab2:
            weather.data_app()
    else:
        pass