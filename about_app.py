# -*- coding:utf-8 -*-

import streamlit as st

def run_about_app():
    st.header('개요')
    tab1, tab2, tab3 = st.tabs(['목적', '구성원 및 역할', '타임라인'])
    with tab1:
        st.subheader('목적')
    with tab2:
        st.subheader('구성원 및 역할')
    with tab3:
        st.subheader('타임라인')