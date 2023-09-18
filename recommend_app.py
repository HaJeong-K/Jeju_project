# -*- coding:utf-8 -*-

import streamlit as st

def run_recommend_app():
    st.header('Home')
    st.markdown('(임시)텍스트마이닝으로 생성한 전반적 워드클라우드 시각화 자료 노출')
    division = st.radio(
        '다음을 기준으로 관광지를 소개해드립니다.',
        ['전체', '성별', '연령대별'],
        horizontal = True)
    
    if division == '전체':
        st.markdown('전체')
    elif division == '성별':
        gender = st.radio(
            '성별을 선택해주세요.',
            ['남성', '여성'],
            horizontal = True
        )
        if gender == '남성':
            st.markdown('남성')
        elif gender == '여성':
            st.markdown('여성')
        else:
            st.markdown('선택한 항목이 없습니다.')
    elif division == '연령대별':
        age = st.radio(
            '연령대를 선택해주세요.',
            ['20대 미만', '20대~30대', '40대~60대', '70대 이상'],
            horizontal = True
        )
        if age == '20대 미만':
            st.markdown('20대 미만')
        elif age == '20대~30대':
            st.markdown('20대~30대')
        elif age == '40대~60대':
            st.markdown('40대~60대')
        elif age == '70대 이상':
            st.markdown('70대 이상')
        else:
            st.markdown('선택한 항목이 없습니다.')
    else:
        st.markdown('선택한 항목이 없습니다.')
    
    st.markdown('(임시)각 항목별로는 추천 관광지 5곳의 사진, 리뷰, 링크와 함께  \n 해당 집단의 과거 구역별 방문빈도 지도 노출')