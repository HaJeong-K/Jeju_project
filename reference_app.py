# -*- coding:utf-8 -*-

import streamlit as st

def run_reference_app():
    st.header('자료출처')
    st.markdown('- https://data.kma.go.kr/climate/RankState/selectRankStatisticsDivisionList.do?pgmNo=179#  \n  기상청 날씨자료 (날씨 데이터)')
    st.markdown('- https://www.visitjeju.net/tourdata/  \n 비짓제주, 방문 관광객 실태조사 DB (물가 데이터, 성수기 데이터)')
    st.markdown('- https://www.tripadvisor.co.kr/Attractions-g983296-Activities-Jeju_Island.html  \n 트립어드바이저, 제주 관광 명소 (관광지 텍스트 마이닝)')
    st.markdown('- https://www.jejudatahub.net/data/view/data/580  \n 제주데이터허브, 방문목적별 입도객 현황 (날씨 데이터)')
    st.markdown('- https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1J20112  \n Kosis, 품목별 소비자 물가 지수')