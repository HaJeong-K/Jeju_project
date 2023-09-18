# -*- coding:utf-8 -*-

import streamlit as st

def run_data_app():
    st.header('분석내용')
    st.markdown('(임시)각 항목별로 시각화 및 통계자료와 사용된 전처리 데이터 노출')
    items = ['물가', '성수기', '날씨']
    item = st.sidebar.selectbox('분석항목', items)
    tabs = ['시각화', '통계', '데이터']

    if item == '물가':
        st.subheader('물가')
        tab1, tab2, tab3 = st.tabs(tabs)
        with tab1:
            st.markdown('시각화')
        with tab2:
            st.markdown('통계')
        with tab3:
            st.markdown('데이터')
    elif item == '성수기':
        st.subheader('성수기')
        tab1, tab2, tab3 = st.tabs(tabs)
        with tab1:
            st.markdown('시각화')
        with tab2:
            st.markdown('통계')
        with tab3:
            st.markdown('데이터')
    elif item == '날씨':
        st.subheader('날씨')
        tab1, tab2, tab3 = st.tabs(tabs)
        with tab1:
            st.markdown('시각화')
        with tab2:
            st.markdown('통계')
        with tab3:
            st.markdown('데이터')
    else:
        pass