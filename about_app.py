# -*- coding:utf-8 -*-

import streamlit as st

def run_about_app():
    st.header('개요')
    tab1, tab2 = st.tabs(['구성원 및 역할', '타임라인'])
    with tab1:
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image('./data/streamlit/구성원.png')
    with tab2:
        col4, col5, col6 = st.columns([1, 6, 1])
        with col5:
            st.image('./data/streamlit/일정.png')