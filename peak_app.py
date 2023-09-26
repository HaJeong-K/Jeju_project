# -*- coding:utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
import requests
import pingouin as pg
from PIL import Image


def vis_app():
    tabs = ['방문객 수', '만족도']
    tab1, tab2 = st.tabs(tabs)
    with tab1:
        df_5years = data_preprocessing()
        st.markdown('#### boxplot으로 분포를 확인합니다.')
        x_5years = df_5years.loc[(df_5years['월'] == '07') | (df_5years['월'] == '08'), '총원'].reset_index(drop = True)

        plt.rcParams['font.family'] = 'NanumGothic'
        fig, ax = plt.subplots(ncols = 2, sharey = True)
        sns.boxplot(data = x_5years, ax = ax[0])
        sns.boxplot(data = df_5years.loc[:, '총원'], ax = ax[1])
        ax[0].set_xlabel('성수기')
        ax[1].set_xlabel('전체')
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].grid(False)
        ax[1].grid(False)
        ax[0].set_ylabel('방문객 수')
        ax[0].ticklabel_format(axis = 'y', style = 'plain')

        fig.savefig('./data/성수기/방문객수.png', bbox_inches = 'tight')
        image = Image.open('./data/성수기/방문객수.png')
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(image)

    with tab2:
        [summer_df, others_df] = satis_data_processing()
        items = ['관광정보', '관광지매력', '교통정보', '관광지편의성', '쇼핑',
            '숙박', '여행경비', '음식', '대중교통', '주민친절', '치안']
        item = st.radio('독립변수를 하나 선택해주세요.', items, horizontal = True)
        indep_var = str(item) + ' 100점평균'
        summer_df_sco = summer_df[indep_var]
        others_df_sco = others_df[indep_var]

        plt.rcParams['font.family'] = 'NanumGothic'
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.boxplot([summer_df_sco, others_df_sco])
        ax.set_xticklabels(['성수기', '비성수기'])
        ax.set_title(f'{item} 만족 100점 평균 - 박스 플롯')
        # colors = ['red', 'blue']
        # for patch, color in zip(ax['boxes'], colors):
        #     patch.set_facecolor(color)
        # for median in ax['medians']:
        #     median.set_style(color = 'black', linewidth = 2)
        ax.grid(False)
        fig.tight_layout()

        fig.savefig(f'./data/성수기/{item}.png', bbox_inches = 'tight')
        image = Image.open(f'./data/성수기/{item}.png')
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(image)
        
def stat_app():
    tabs = ['방문객 수', '만족도']
    tab1, tab2 = st.tabs(tabs)
    with tab1:
        df_5years = data_preprocessing()
        mu_5years = df_5years.loc[:, '총원'].mean()
        x_5years = df_5years.loc[(df_5years['월'] == '07') | (df_5years['월'] == '08'), '총원'].reset_index(drop = True)
        x_5years = x_5years.astype('float64')
        ttest_5years = pg.ttest(x_5years, mu_5years).round(4)
        p_val_5years = float(ttest_5years.iloc[0, 3])

        st.markdown('#### 통계검정 - 단일표본 t-test')
        st.markdown('- 귀무가설: 5년간 월별 입도객 수의 전체 평균은 성수기인 7-8월의 평균 입도객 수와 통계적으로 유의한 차이가 없다.')
        st.markdown('- 대립가설: 5년간 월별 입도객 수의 전체 평균은 성수기인 7-8월의 평균 입도객 수와 통계적으로 유의한 차이가 있다.')
        st.dataframe(ttest_5years, hide_index = True)
        alpha = 0.05
        if p_val_5years < alpha:
            st.markdown(f'p-value값은 {p_val_5years}로 {alpha}미만이기 때문에 귀무가설을 기각합니다.')
            st.markdown('즉, 5년간 월별 입도객 수의 전체 평균은 성수기인 7-8월의 평균 입도객 수와 통계적으로 유의한 차이가 있습니다.')
        else:
            st.markdown(f'p-value값은 {p_val_5years}로 {alpha}이상이기 때문에 귀무가설을 기각하지 못합니다.')
            st.markdown('즉, 5년간 월별 입도객 수의 전체 평균은 성수기인 7-8월의 평균 입도객 수와 통계적으로 유의한 차이가 없습니다.')
        st.divider()
        
        st.markdown('#### Shapiro-Wilk test')
        st.markdown('- 표본의 크기가 10으로 작기 때문에 정규성 검정')
        st.markdown('- 귀무가설: 통계적으로 표본은 정규성을 따르는 모집단 분포로부터 추출되었다.')
        st.markdown('- 대립가설: 통계적으로 표본은 정규성을 따르는 모집단 분포로부터 추출되지 않았다.')
        shapiro_5years = pg.normality(x_5years).round(4)

        shapiro_p_5years = float(shapiro_5years.iloc[0, 1])
        st.dataframe(pg.normality(x_5years).round(4))
        if shapiro_p_5years < alpha:
            st.markdown(f'p-value값은 {shapiro_p_5years}로 {alpha}미만이기 때문에 귀무가설을 기각합니다.')
            st.markdown('즉, 통계적으로 표본은 정규성을 따르는 모집단 분포로부터 추출되지 않았습니다.')
        else:
            st.markdown(f'p-value값은 {shapiro_p_5years}로 {alpha}이상이기 때문에 귀무가설을 기각하지 못합니다.')
            st.markdown('즉, 통계적으로 표본은 정규성을 따르는 모집단 분포로부터 추출되었습니다.')
        st.divider()

    with tab2:
        st.markdown('#### 통계검정 - 독립표본 t-test')
        st.markdown('- 귀무가설: 성수기와 비성수기 간의 각 항목별 만족 100점 평균은 유의미한 차이가 없다.')
        st.markdown('- 대립가설: 성수기와 비성수기 간의 각 항목별 만족 100점 평균은 유의미한 차이가 있다.')
        st.markdown('- 성수기 : 7~8월, 비성수기 : 나머지 월')
        st.markdown('- column들이 항목이 아니라 척도와 평균점수이기 때문에 요인분석이 불가하다고 판단')
        st.markdown('- 데이터 양이 많지 않아 alpha를 0.1로 두고 측정')
        st.divider()
        [summer_df, others_df] = satis_data_processing()
        items = ['관광정보', '관광지매력', '교통정보', '관광지편의성', '쇼핑',
            '숙박', '여행경비', '음식', '대중교통', '주민친절', '치안']
        item = st.radio('독립변수를 선택해주세요.', items, horizontal = True)
        indep_var = item + ' 100점평균'
        summer_df_sco = summer_df[indep_var]
        others_df_sco = others_df[indep_var]
        t_statistic, p_value = ttest_ind(summer_df_sco, others_df_sco)

        alpha = 0.10  # 유의수준 설정
        if p_value < alpha:
            st.markdown(f'귀무 가설을 기각합니다. 성수기와 비성수기 간 ' + item + ' 100점 평균에는 유의미한 차이가 있습니다.')
        else:
            st.markdown(f'귀무 가설을 기각하지 않습니다. 성수기와 비성수기 간 ' + item + ' 100점 평균에는 유의미한 차이가 없습니다.')
        st.markdown(f'관광정보 t-통계량: {t_statistic: 0.4f}')
        st.markdown(f'관광정보 p-값: {p_value: 0.4f}')
        st.divider()
        st.markdown('''
        #### 유의미한 값이 나오지 않은 이유 추정
        1. 데이터 양이 적음
        - 코로나 때문에 누락도 있지만, 성수기를 7-8월로만 잡았기 때문에 성수기와 비성수기 간의 데이터 양 차이도 존재하며, 4년치밖에 없어서 총 데이터 량도 적은편임
        2. 제주도의 만족도 점수 자체가 성수기/비성수가를 크게 타지 않을 가능성이 높아서
        - 시기보다는 평가항목마다 만족도/불만족도가 크게 갈리는 것으로 추정 > 시기를 타지 않고 항상 불만족도가 높은 항목이 존재(여행경비, 대중교통 둘 다 성수기 비성수기를 가리지 않고 평균점수가 60점대 아래임)
        - 별개로 가장 p값이 낮게 나온 건 여행경비 부분이라, 데이터가 지금보다 더 많이 누적될 경우 유의미한 차이가 나타날 수도 있다고 봄''')
        st.divider()

def data_app():
    tabs = ['방문객 수', '만족도']
    tab1, tab2 = st.tabs(tabs)
    with tab1:
        st.dataframe(data_preprocessing(), hide_index = True)
    with tab2:
        [summer_df, others_df] = satis_data_processing()
        items = ['관광정보', '관광지매력', '교통정보', '관광지편의성', '쇼핑',
            '숙박', '여행경비', '음식', '대중교통', '주민친절', '치안']
        items = [item + ' 100점평균' for item in items]
        summer_df_sco = summer_df[['연도', '세부분류'] + items].reset_index(drop = True)
        others_df_sco = others_df[['연도', '세부분류'] + items].reset_index(drop = True)
        st.markdown('#### 성수기')
        st.dataframe(summer_df_sco, hide_index = True)
        st.markdown('#### 비성수기')
        st.dataframe(others_df_sco, hide_index = True)

def data_preprocessing():
    PATH = {}
    PATH[2018] = 'https://api.odcloud.kr/api/3083546/v1/uddi:246e7b85-9ecc-4b0d-89d6-9aaff6369813'
    PATH[2019] = 'https://api.odcloud.kr/api/3083546/v1/uddi:a7bf1e97-d777-4550-8e98-b254dc64d9e6'
    PATH[2020] = 'https://api.odcloud.kr/api/3083546/v1/uddi:70b77b60-b9ee-4cd4-ad49-d24626f1af6b'
    PATH[2021] = 'https://api.odcloud.kr/api/3083546/v1/uddi:edda6259-c720-4c6e-8b11-f991b9720e34'
    PATH[2022] = 'https://api.odcloud.kr/api/3083546/v1/uddi:6d8a391a-e30e-4feb-bd74-e5efc804e631'
    serviceKey = 'YjwuOSErCt4PB%2BANN4eK26d76AIC6dbwF52v%2FNkCmmZdIi4ZSXRSITMWtc2y%2B%2F8gjL6p4%2FBfiNZgoraqpsihDg%3D%3D'

    df_dict = {}
    for i in range(2018, 2023):
        url = PATH[i] + '?page=1&perPage=12&serviceKey=' + serviceKey
        req_temp = requests.get(url)
        json_temp = req_temp.json()
        df_temp = pd.DataFrame(json_temp['data'])
        df_temp.iloc[:, -3:] = df_temp.iloc[:, -3:].astype('int64')
        df_temp['총원'] = df_temp.iloc[:, -3:].sum(axis = 1)
        df_temp = df_temp.loc[:, ['구분연월', '총원']]
        df_temp[['연', '월']] = df_temp['구분연월'].str.split('-', expand = True)
        df_dict[i] = df_temp
    df_5years = pd.concat(df_dict.values(), ignore_index = True)
    
    return df_5years

def satis_data_processing():
    DATA_PATH = './data/성수기/'
    YEARS = [2018, 2019, 2020, 2021]
    jeju_result = None
    for YEAR in YEARS:
        temp_result = pd.read_csv(DATA_PATH + f'result_{YEAR}.csv')
        temp_result = temp_result.drop(['Unnamed: 0'], axis = 1)
        temp_result['연도'] = YEAR
        jeju_result = pd.concat([jeju_result, temp_result])
    jeju_result = jeju_result.reset_index(drop = True).drop(['분류'], axis = 1)
    summer_df = jeju_result[(jeju_result['세부분류'] == '07월') | (jeju_result['세부분류'] == '08월')]
    others_df = jeju_result[(jeju_result['세부분류'] != '07월') & (jeju_result['세부분류'] != '08월')]
    return [summer_df, others_df]