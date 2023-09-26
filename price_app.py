# -*- coding:utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_samples, silhouette_score
from factor_analyzer import FactorAnalyzer
from re import X
import requests
import pingouin as pg
from PIL import Image

def pred_app():
    cost_list = data_preprocessing()
    Regression_analysis = pd.read_excel('./data/물가/회귀분석용.xlsx')
    Regression_analysis['날짜'] = pd.to_datetime(Regression_analysis['날짜'])
    Regression_analysis['월'] = Regression_analysis['날짜'].dt.month
    Regression_analysis = Regression_analysis[['월', '환율', '저압 전기요금', '유류할증료', '최저시급', '공산품 지수']]
    Regression_df = Regression_analysis.copy()
    Regression_df = Regression_analysis.iloc[:48]
    
    name_dict = {0 : '전체 지출 비용',
        1 : '숙박 비용',
        2 : '식음료 비용',
        3 : '항공, 선박 비용',
        4 : '관광, 문화 지출 비용'}
    YEARS = [2018, 2019, 2020, 2021, 2022]
    result_dict = {}
    fig2, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))
    fig4, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))

    for i in range(len(cost_list)):
        temp_dict = {}
        merged = pd.concat(cost_list[i].values(), axis = 0, ignore_index = True)
        new_index = range(len(merged), len(merged) + 12)
        new_data = []
        for j in range(12):
            new_data.append({
                '분석 카테고리': '월별',
                '세부 카테고리': f'{j+1}월',
                '연도': 2022})
        prediction = pd.DataFrame(new_data)
        merged_prediction = pd.concat([merged, prediction], ignore_index = True)
        temp_df = pd.concat([merged_prediction, Regression_analysis], axis = 1)
        temp_df = temp_df[['연도', '월', '평균', '환율', '저압 전기요금', '유류할증료', '최저시급', '공산품 지수']]
        temp_df.columns = ['연도', '월', '평균', '환율', '전기세', '유류할증료', '최저시급', '공산품 수출 지수']
        
        for month in range(1, 13):
            means_2018_to_2021 = temp_df.loc[(temp_df['연도'].isin([2018, 2019, 2020, 2021])) & (temp_df['월'] == month), '평균']
            mean_2018_to_2021 = means_2018_to_2021.mean()
            temp_df.loc[47 + month, '평균'] = mean_2018_to_2021

        X_train = temp_df.iloc[:48][['최저시급', '공산품 수출 지수']]
        y_train = temp_df.iloc[:48]['평균']

        # 선형 회귀 모델을 생성하고 학습 데이터를 사용하여 모델을 훈련하기
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 2022년 월평균 전체 지출 비용을 예측하기
        temp_2022 = temp_df.iloc[48:][['최저시급', '공산품 수출 지수']]
        temp_predictions_2022 = model.predict(temp_2022)
        
        temp_dict_2 = {}
        for YEAR in YEARS:
            temp_dict[YEAR] = temp_df[temp_df['연도'] == YEAR][['평균', '연도', '월']].reset_index(drop = True)
            temp_dict_2[YEAR] = temp_df[temp_df['연도'] == YEAR]['평균'].reset_index(drop = True)
        temp_dict[2022]['평균'] = pd.Series(temp_predictions_2022)
        temp_dict_2[2022] = pd.Series(temp_predictions_2022)
        print(temp_dict_2.values())

        name = name_dict[i]
        if i == 0:
            fig1, ax = plt.subplots(figsize = (10, 6))
            boxes = ax.boxplot(temp_dict_2.values(),
                labels = ['2018', '2019', '2020', '2021', '2022'],
                patch_artist = True)
            colors = ['red', 'blue', 'green', 'orange', 'pink']
            for box, color in zip(boxes['boxes'], colors):
                box.set(facecolor = color)
            ax.set_title('연도별 월평균 데이터 분포')
            ax.set_xlabel('연도')
            ax.set_ylabel(f'{name}')
            ax.grid(False)
            fig1.tight_layout()
            fig1.savefig('./data/물가/전체지출_예측.png', bbox_inches = 'tight')
            image1 = Image.open('./data/물가/전체지출_예측.png')

            temp_sum = pd.concat(temp_dict.values(), axis = 0, ignore_index = True)
            temp_sum['날짜'] = temp_sum['연도'].astype(str) + '-' + temp_sum['월'].apply(lambda x: '%02d' % x)
            fig3, axx = plt.subplots(figsize = (10, 6))
            axx.plot(temp_sum['날짜'], temp_sum['평균'], marker = 'o', linestyle = '-', color = 'b')
            axx.set_xlabel('월별')
            axx.set_ylabel('전체 비용의 평균값')
            axx.set_title('전체 비용의 월별 평균값')
            axx.set_xticklabels(axx.get_xticklabels(), rotation = 45)
            axx.grid(True)
            fig3.tight_layout()
            fig3.savefig('./data/물가/전체지출_예측_point.png', bbox_inches = 'tight')
            image3 = Image.open('./data/물가/전체지출_예측_point.png')

        else:
            boxes = axs[(i-1)//2, (i-1)%2].boxplot(temp_dict_2.values(),
                labels = ['2018', '2019', '2020', '2021', '2022'],
                patch_artist = True)
            colors = ['red', 'blue', 'green', 'orange', 'pink']
            for box, color in zip(boxes['boxes'], colors):
                box.set(facecolor = color)
            axs[(i-1)//2, (i-1)%2].set_title('연도별 월평균 데이터 분포')
            axs[(i-1)//2, (i-1)%2].set_xlabel('연도')
            axs[(i-1)//2, (i-1)%2].set_ylabel(f'{name}')
            axs[(i-1)//2, (i-1)%2].grid(False)
            fig2.tight_layout()

            temp_sum = pd.concat(temp_dict.values(), axis = 0, ignore_index = True)
            temp_sum['날짜'] = temp_sum['연도'].astype(str) + '-' + temp_sum['월'].apply(lambda x: f'{x:02d}')
            axes[(i-1)//2, (i-1)%2].plot(temp_sum['날짜'], temp_sum['평균'], marker = 'o', linestyle = '-', color = 'b')
            axes[(i-1)//2, (i-1)%2].set_xlabel('월별')
            axes[(i-1)//2, (i-1)%2].set_ylabel(f'{name}의 평균값')
            axes[(i-1)//2, (i-1)%2].set_title(f'{name}의 월별 평균값')
            axes[(i-1)//2, (i-1)%2].set_xticklabels(axes[(i-1)//2, (i-1)%2].get_xticklabels(), rotation = 45)
            axes[(i-1)//2, (i-1)%2].grid(True)
            fig4.tight_layout()
        
        # 예측 결과를 출력하기
        temp_predict_list = []
        for j in range(12):
            temp_predict_list.append([j + 1, temp_predictions_2022[j], temp_df.iloc[j + 48]['평균']])
            # st.markdown(f'2022년 {month}월 예측 월평균 지출금액: {prediction}')
        temp_predict_df = pd.DataFrame(temp_predict_list,
        columns = ['월', '2022년 예측 월평균 지출금액', '4년간 월평균 지출금액'])
        result_dict[i] = temp_predict_df

    st.markdown('#### 전체')
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.image(image1)
        st.image(image3)

    st.markdown('#### 부문별')
    st.pyplot(fig2)
    st.pyplot(fig4)

    for i in range(len(result_dict.values())):
        temp_predict_df = result_dict[i]
        name = name_dict[i]
        st.markdown(f'#### 종속변수: {name}')
        st.dataframe(temp_predict_df, hide_index = True)

        # 학습 데이터를 사용하여 모델의 예측값을 계산하고 MSE를 계산하기
        mse = mean_squared_error(temp_predict_df.iloc[:, 2], temp_predict_df.iloc[:, 1])

        # RMSE를 계산하여 모델의 예측 정확성을 평가하기
        rmse = np.sqrt(mse)
        st.markdown(f'- Root Mean Squared Error (RMSE): {rmse}')

        # pred_cost_list.append(temp_df)
        # pred_cost_list[i]['날짜'] = pd.to_datetime(pred_cost_list[i]['연도'].astype(str) + pred_cost_list[i]['월'].astype(str).str.zfill(2) + '01', format = '%Y%m%d')
        # st.dataframe(pred_cost_list[i])

def vis_app():
    cost_list = data_preprocessing()
    tabs = ['분산분석', '회귀분석', '군집분석', '요인분석']
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    with tab1:
        plt.rcParams['font.family'] = 'NanumGothic'
        labels = [str(year) for year in range(2018, 2022)]
        colors = ['red', 'blue', 'green', 'orange']
        box_width = 0.4
        name_dict = {0 : '전체 지출 비용',
        1 : '숙박 비용',
        2 : '식음료 비용',
        3 : '항공, 선박 비용',
        4 : '관광, 문화 지출 비용'}

        total_cost_comparison = cost_list[0].values()
        fig1, ax = plt.subplots(figsize = (9, 6))
        positions_1 = range(1, len(total_cost_comparison) + 1)
        for i, df in enumerate(total_cost_comparison):
            ax.boxplot(df['평균'], positions = [positions_1[i]], vert = True, patch_artist = True, boxprops = dict(facecolor = colors[i]), widths = box_width)

        ax.set_xlabel('연도')
        ax.set_xticks(positions_1, labels)
        ax.set_ylabel('전체 지출의 월별 평균값')
        ax.set_title('연도별 전체 지출의 월별 평균값 분포')
        ax.grid(False)
        fig1.savefig('./data/물가/전체지출_분산.png', bbox_inches = 'tight')
        image = Image.open('./data/물가/전체지출_분산.png')

        st.markdown('#### 전체')
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(image)

        fig2, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))
        for idx, cost in enumerate(cost_list[1:]):
            temp_comparison = cost.values()
            positions_2 = range(1, len(temp_comparison) + 1)
            name = name_dict[idx + 1]
            for i, df in enumerate(temp_comparison):
                axs[idx//2, idx%2].boxplot(df['평균'], positions = [positions_2[i]], vert = True, patch_artist = True, boxprops = dict(facecolor = colors[i]), widths = box_width)
                axs[idx//2, idx%2].set_xlabel('연도')
                axs[idx//2, idx%2].set_xticks(positions_2, labels)
                axs[idx//2, idx%2].set_ylabel(f'{name}의 월별 평균값')
                axs[idx//2, idx%2].set_title(f'연도별 {name}의 월별 평균값 분포')
                axs[idx//2, idx%2].grid(False)

        st.markdown('#### 부문별')
        st.pyplot(fig2)

    with tab2:
        indep_var = st.radio('독립변수를 하나 선택해주세요.',
        ['환율', '전기세', '유류할증료', '최저시급', '공산품 수출 지수'],
        horizontal = True)
        name_dict = {0 : '종속변수: 전체 지출 비용',
        1 : '종속변수: 숙박 비용',
        2 : '종속변수: 식음료 비용',
        3 : '종속변수: 항공, 선박 비용',
        4 : '종속변수: 관광, 문화 지출 비용'}

        image_dict = reg_vis(indep_var)
        st.markdown('#### 전체')
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(image_dict[0])
        st.markdown('#### 부문별')
        st.image(image_dict[1])

    with tab3:
        dis_df = cluster_df()

        marker0_ind = dis_df[dis_df['cluster'] == 0].index
        marker1_ind = dis_df[dis_df['cluster'] == 1].index
        marker2_ind = dis_df[dis_df['cluster'] == 2].index
        marker3_ind = dis_df[dis_df['cluster'] == 3].index
        marker4_ind = dis_df[dis_df['cluster'] == 4].index

        plt.rcParams['font.family'] = 'NanumGothic'
        fig, ax = plt.subplots()
        ax.scatter(x = dis_df.loc[marker0_ind, 'pca_x'], y = dis_df.loc[marker0_ind, 'pca_y'], marker = 'o', c = 'b', label = 'Cluster 0')
        ax.scatter(x = dis_df.loc[marker1_ind, 'pca_x'], y = dis_df.loc[marker1_ind, 'pca_y'], marker = 'o', c = 'g', label = 'Cluster 1')
        ax.scatter(x = dis_df.loc[marker2_ind, 'pca_x'], y = dis_df.loc[marker2_ind, 'pca_y'], marker = 'o', c = 'r', label = 'Cluster 2')
        ax.scatter(x = dis_df.loc[marker3_ind, 'pca_x'], y = dis_df.loc[marker3_ind, 'pca_y'], marker = 'o', c = 'y', label = 'Cluster 3')
        ax.scatter(x = dis_df.loc[marker4_ind, 'pca_x'], y = dis_df.loc[marker4_ind, 'pca_y'], marker = 'o', c = 'k', label = 'Cluster 4')
        ax.get_legend()
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_title('불만족점 군집분석')
        ax.grid(False)
        fig.savefig('./data/물가/설문조사/PCA.png', bbox_inches = 'tight')
        image = Image.open('./data/물가/설문조사/PCA.png')

        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(image)

    with tab4:
        merge_budget = pd.concat(survey_data_preprocessing()[2], axis = 0)
        efa_result = fa_df()
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize = (6, 10))
        sns.heatmap(efa_result, cmap = "Reds", annot = True, fmt = '.2f')
        ax.grid(False)
        fig.savefig('./data/물가/설문조사/fa.png', bbox_inches = 'tight')
        image = Image.open('./data/물가/설문조사/fa.png')

        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(image)

def stat_app():
    cost_list = data_preprocessing()
    tabs = ['분산분석', '회귀분석', '군집분석', '요인분석']
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    with tab1:
        st.markdown('#### 일원 분산 분석')
        st.markdown('- 귀무가설 : 4개년의 월 평균 지출액은 통계적으로 유의한 차이가 없다.')
        st.markdown('- 대립가설 : 적어도 한 연도의 월 평균 지출액은 통계적으로 유의한 차이가 있다.')
        st.divider()

        name_dict = {0 : '월별 전체 지출 비용',
        1 : '월별 숙박 비용',
        2 : '월별 식음료 비용',
        3 : '월별 항공, 선박 비용',
        4 : '월별 관광, 문화 지출 비용'}

        for i in range (len(cost_list)):
            name = name_dict[i]
            st.markdown(f'#### {name}')
            merged = pd.concat(cost_list[i].values(), axis = 0, ignore_index = True)
            merged['평균'] = pd.to_numeric(merged['평균'], errors = 'coerce')
            aov = pg.anova(dv = '평균', between = '연도', data = merged).round(4)
            st.dataframe(aov, hide_index = True)
            pval = float(aov.iloc[0, 4])
            if pval < 0.05:
                st.markdown(f'- p-value값은 {pval}로 0.05미만이기 때문에 귀무가설을 기각합니다.')
                tukey_df = pg.pairwise_tukey(data = merged, dv = '평균', between = '연도').round(4)
                st.markdown('##### 사후분석')
                st.dataframe(tukey_df, hide_index = True)
                p_list = []
                for j in range(len(tukey_df)):
                    pval_tukey = tukey_df.loc[j, 'p-tukey']
                    if pval_tukey < 0.05:
                        p_list.append(j)
                    else:
                        pass
                st.markdown('- p-value값이 0.05미만인')
                for k in p_list:
                    year1 = tukey_df.loc[k, 'A']
                    year2 = tukey_df.loc[k, 'B']
                    st.markdown(f'{year1}년과 {year2}년')
                st.markdown('의 경우 평균이 통계적으로 유의미한 차이를 보여 그대로 귀무가설을 기각하고,  \n 그 외에는 귀무가설을 기각하지 않는 것이 더 적절해보입니다.')
            else:
                st.markdown(f'- p-value값은 {pval}로 0.05미만이기 때문에 귀무가설을 기각하지 못합니다.')
            st.divider()

    with tab2:
        st.markdown('#### 선형 회귀 분석')
        indep_var = st.radio('독립변수를 선택해주세요.',
        ['환율', '전기세', '유류할증료', '최저시급', '공산품 수출 지수'],
        horizontal = True)
        name_dict = {0 : '종속변수 - 전체 지출 비용',
        1 : '종속변수: 숙박 비용',
        2 : '종속변수: 식음료 비용',
        3 : '종속변수: 항공, 선박 비용',
        4 : '종속변수: 관광, 문화 지출 비용'}
        
        lin_reg_df_dict, corr_dict = reg_stat(indep_var)
        for i in range(len(lin_reg_df_dict)):
            division = name_dict[i]
            st.markdown(f'- {division}')
            st.dataframe(lin_reg_df_dict[i], hide_index = True)
            reg_corr = corr_dict[i]
            st.markdown(f'상관계수는 {reg_corr}입니다.')
            st.divider()
        st.markdown('''
        - **전반적으로 모든 독립변수가 강한 상관관계를 띄는 것을 찾기가 어려웠지만, 공산품 수출 지수는 전체 지출 비용과 숙박 비용에서** 다른 회귀분석들에 비해 **강한 상관관계를 보인다.**
        - 모든 독립변수를 기준으로 해서 회귀분석 결과를 봤을 때, **최저시급은** 다른 독립변수에 비해 세부적으로 분류한 지출비용과도 **골고루 상관계수 값이 비교적 크게 나타나는 추이를 보인다.**
        - 환율과 전기세는 전체적으로 상관관계가 거의 없는 것으로 나타났다.
        - 유류할증료는 환율과 전기세에 비하면 약간의 상관관계가 있는 것으로 보인다.''')
        st.divider()
    
    with tab3:
        st.markdown('#### 불만족 요인에 대한 설문조사 결과 분석')
        dis_df = cluster_df()
        st.dataframe(dis_df, hide_index = True)
        st.divider()
        st.markdown('''
        > 클러스터 설명
        - 클러스터 0: "기타" 컬럼에서 높은 응답률을 보임
        - 클러스터 1: "물가가 비싸다" 컬럼에서 비교적 높은 응답률을 보임
        - 클러스터 2: "관광종사원이 불친절하다", "안내표지판이 부정확하다", "식당과 음식이 불결하다", "음식이 입에 맞지 않는다", "관광가이드의 서비스가 좋지 않다", "상품구입을 강요한다" 컬럼에서 높은 응답률을 보임
        - 클러스터 3: "불만족하거나 불편했던 점이 없다" 컬럼에서 높은 응답률을 보임
        - 클러스터 4: "물가가 비싸다", "쇼핑품목이 다양하지 않다" 컬럼에서 높은 응답률을 보임''')
        st.divider()
        st.markdown('''
        > 클러스터 재 정의
        - 클러스터 0: 설문지 외의 불만족
        - 클러스터 1: 물가에 대한 불만족
        - 클러스터 2: 관광지 서비스에 대한 불만족
        - 클러스터 3: 여행 만족
        - 클러스터 4: 물가 및 상품 부족에 대한 불만족''')
        st.divider()

    with tab4:
        st.markdown('#### 여행경비 만족에 대한 설문조사 결과 분석')
        efa_result = fa_df()
        st.dataframe(efa_result, hide_index = True)

def data_app():
    tabs = ['분산분석', '회귀분석', '군집분석', '요인분석']
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    with tab1:
        YEARS = [2018, 2019, 2020, 2021]
        name_dict = {0 : '종속변수 - 전체 지출 비용',
        1 : '종속변수 - 숙박 비용',
        2 : '종속변수 - 식음료 비용',
        3 : '종속변수 - 항공, 선박 비용',
        4 : '종속변수 - 관광, 문화 지출 비용'}

        for i in range(len(data_preprocessing())):
            name = name_dict[i]
            st.markdown(f'#### {name}')
            col1, col2, col3, col4 = st.columns(4)
            for idx, YEAR in enumerate(YEARS):
                col_dict = {0: col1, 1: col2, 2: col3, 3: col4}
                with col_dict[idx]:
                    st.markdown(f'{YEAR}년')
                    st.dataframe(data_preprocessing()[i][YEAR][['세부 카테고리', '평균']], hide_index = True)
    with tab2:
        cost_list = data_preprocessing()
        for i in range(len(cost_list)):
            name = name_dict[i]
            st.markdown(f'#### {name}')
            st.dataframe(reg_data_preprocessing()[i], hide_index = True)
    with tab3:
        dis_dict = survey_data_preprocessing()[1]
        merge_dis = pd.concat(dis_dict.values(), axis = 0).reset_index(drop = True)
        st.dataframe(merge_dis, hide_index = True)
    with tab4:
        merge_budget = pd.concat(survey_data_preprocessing()[2], axis = 0)
        merge_budget = merge_budget.reset_index(drop = True)
        price = pd.read_csv('./data/물가/제주물가.csv')
        price = price.iloc[:40, :]
        price['전국대비물가'] = price['제주-전국'].apply(lambda x: '높음' if x > 0 else '낮음')
        price = price['전국대비물가']

        columns = ['세부 카테고리', '매우 불만족', '불만족', '보통', '만족', '매우 만족']
        budget_df = merge_budget[columns]
        budget_df = pd.concat([budget_df, price], axis = 1)
        st.dataframe(budget_df, hide_index = True)

def data_preprocessing():
    DATA_PATH = './data/물가/'
    YEARS = [2018, 2019, 2020, 2021]
    total_cost = {}
    accommodation_cost = {}
    food_cost = {}
    transportation_cost = {}
    culture_cost = {}
    for YEAR in YEARS:
        df_total = pd.read_excel(DATA_PATH + f'{str(YEAR)}년/{str(YEAR)}년 1인당 지출경비 - 개별 전체.xlsx')
        df_accommodation = pd.read_excel(DATA_PATH + f'{str(YEAR)}년/{str(YEAR)}년 1인당 지출경비 - 개별 숙박비.xlsx')
        df_food = pd.read_excel(DATA_PATH + f'{str(YEAR)}년/{str(YEAR)}년 1인당 지출경비 - 개별 식음료비.xlsx')
        df_transportation = pd.read_excel(DATA_PATH + f'{str(YEAR)}년/{str(YEAR)}년 1인당 지출경비 - 개별 항공,선박비.xlsx')
        df_culture = pd.read_excel(DATA_PATH + f'{str(YEAR)}년/{str(YEAR)}년 1인당 지출경비 - 개별 관광,문화 지출비.xlsx')
        
        df_total = drop_row(rename_df(df_total), YEAR)
        df_total['연도'] = YEAR
        df_accommodation = drop_row(rename_df(df_accommodation), YEAR)
        df_accommodation['연도'] = YEAR
        df_food = drop_row(rename_df(df_food), YEAR)
        df_food['연도'] = YEAR
        df_transportation = drop_row(rename_df(df_transportation), YEAR)
        df_transportation['연도'] = YEAR
        df_culture = drop_row(rename_df(df_culture), YEAR)
        df_culture['연도'] = YEAR

        total_cost[YEAR] = df_total
        accommodation_cost[YEAR] = df_accommodation
        food_cost[YEAR] = df_food
        transportation_cost[YEAR] = df_transportation
        culture_cost[YEAR] = df_culture

    # Year 2020, 2021 interpolation
    total_cost = interpolation(total_cost, 2020)
    accommodation_cost = interpolation(accommodation_cost, 2020)
    food_cost = interpolation(food_cost, 2020)
    transportation_cost = interpolation(transportation_cost, 2020)
    culture_cost = interpolation(culture_cost, 2020)

    total_cost = interpolation(total_cost, 2021)
    accommodation_cost = interpolation(accommodation_cost, 2021)
    food_cost = interpolation(food_cost, 2021)
    transportation_cost = interpolation(transportation_cost, 2021)
    culture_cost = interpolation(culture_cost, 2021)

    return [total_cost, accommodation_cost, food_cost, transportation_cost, culture_cost]

def rename_df(df):
    df.iloc[0, 0] = '분석 카테고리'
    df.iloc[0, 1] = '세부 카테고리'
    df.iloc[0, -1] = '평균'
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop = True)
    return df

def drop_row(df, YEAR):
    if YEAR == 2020:
        df = df.drop(index = range(0, 3)).reset_index(drop = True)
    else:
        df = df.drop(index = 0).reset_index(drop = True)

    if YEAR == 2021:
        df = df.drop(index = range(9, 42)).reset_index(drop = True)
    elif YEAR == 2020:
        df = df.drop(index = range(7, 40)).reset_index(drop = True)
    else:
        df = df.drop(index = range(12, 43)).reset_index(drop = True)

    return df

def interpolation(df_dict, YEAR):
    if YEAR == 2020:
        months_to_add = ['03월', '04월', '05월', '06월', '07월']
    if YEAR == 2021:
        months_to_add = ['03월', '04월', '05월']
    else:
        pass
    
    to_concat = []
    for month in months_to_add:
        if month not in df_dict[YEAR]['세부 카테고리'].values:
            new_row = {'분석 카테고리': '월별', '세부 카테고리': month, '평균': np.nan}
            to_concat.append(pd.DataFrame([new_row]))

    df_dict[YEAR] = pd.concat([df_dict[YEAR]] + to_concat, ignore_index = True)
    df_dict[YEAR] = df_dict[YEAR].sort_values(by = '세부 카테고리')
    df_dict[YEAR] = df_dict[YEAR].get(['분석 카테고리', '세부 카테고리', '평균'])
    df_dict[YEAR]['평균'] = pd.to_numeric(df_dict[YEAR]['평균'], errors = 'coerce', downcast = 'integer')
    df_dict[YEAR] = df_dict[YEAR].reset_index(drop = True)

    # Spline Interpolation
    x = df_dict[YEAR].index.values
    y = df_dict[YEAR]['평균'].values
    mask = ~np.isnan(y)
    cs = CubicSpline(x[mask], y[mask])
    interpolated_values = cs(x)
    df_dict[YEAR]['평균'] = interpolated_values
    df_dict[YEAR]['평균'] = np.round(df_dict[YEAR]['평균']).astype(int)
    df_dict[YEAR]['연도'] = YEAR

    return df_dict

def reg_data_preprocessing():
    cost_list = data_preprocessing()
    Regression_analysis = pd.read_excel('./data/물가/회귀분석용.xlsx')
    Regression_analysis['날짜'] = pd.to_datetime(Regression_analysis['날짜'])
    Regression_analysis['월'] = Regression_analysis['날짜'].dt.month
    Regression_analysis = Regression_analysis[['월', '환율', '저압 전기요금', '유류할증료', '최저시급', '공산품 지수']]
    Regression_df = Regression_analysis.copy()
    Regression_df = Regression_analysis.iloc[:48]

    reg_df_dict = {}
    for i in range(len(cost_list)):
        merged = pd.concat(cost_list[i].values(), axis = 0, ignore_index = True)
        reg_df = pd.concat([merged, Regression_df], axis = 1)
        reg_df = reg_df[['연도', '월', '평균', '환율', '저압 전기요금', '유류할증료', '최저시급', '공산품 지수']]
        reg_df.columns = ['연도', '월', '평균', '환율', '전기세', '유류할증료', '최저시급', '공산품 수출 지수']
        reg_df['평균'] = reg_df['평균'].astype('float64')
        reg_df_dict[i] = reg_df

    return reg_df_dict

def reg_vis(indep_var):
    reg_df_dict = reg_data_preprocessing()
    image_dict = {}
    plt.rcParams['font.family'] = 'NanumGothic'
    fig2, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))
    for i in range(len(reg_df_dict)):
        idx_dict = {0: '전체 지출 비용',
        1: '숙박 비용',
        2: '식음료 비용',
        3: '항공, 선박 비용',
        4: '관광, 문화 지출 비용'}
        cost = idx_dict[i]
        X = reg_df_dict[i][indep_var]

        if i == 0:
            fig1, ax = plt.subplots(figsize = (6, 6))
            sns.regplot(x = indep_var, y = '평균', data = reg_df_dict[i], ci = None, ax = ax)
            ax.set_title(indep_var + '에 따른 월평균 ' + cost + '에 대한 회귀 분석')
            ax.set_xlabel(indep_var)
            ax.set_ylabel('월평균 ' + cost)
            ax.set_xticks(np.linspace(int(min(X)), int(max(X))+1, 6))
            ax.grid(False)
            fig1.savefig(f'./data/물가/regression/{cost}_{indep_var}.png', bbox_inches = 'tight')
            image = Image.open(f'./data/물가/regression/{cost}_{indep_var}.png')
            image_dict[i] = image
        if i !=0:
            name = idx_dict[i]
            sns.regplot(x = indep_var, y = '평균', data = reg_df_dict[i], ci = None, ax = axs[(i-1)//2, (i-1)%2])
            axs[(i-1)//2, (i-1)%2].set_title(indep_var + '에 따른 월평균 ' + cost + '에 대한 회귀 분석')
            axs[(i-1)//2, (i-1)%2].set_xlabel(indep_var)
            axs[(i-1)//2, (i-1)%2].set_ylabel('월평균 ' + cost)
            axs[(i-1)//2, (i-1)%2].set_xticks(np.linspace(int(min(X)), int(max(X))+1, 6))
            axs[(i-1)//2, (i-1)%2].grid(False)
    fig2.savefig(f'./data/물가/regression/부문별_{indep_var}.png', bbox_inches = 'tight')
    image = Image.open(f'./data/물가/regression/부문별_{indep_var}.png')
    image_dict[1] = image

    return image_dict

def reg_stat(indep_var):
    reg_df_dict = reg_data_preprocessing()
    lin_reg_df_dict = {}
    corr_dict = {}
    for i in range(len(reg_df_dict)):
        X = reg_df_dict[i][indep_var]
        y = reg_df_dict[i]['평균']
        regression_result = pg.linear_regression(X, y)
        correlation = reg_df_dict[i][indep_var].corr(reg_df_dict[i]['평균']).round(4)
        lin_reg_df_dict[i] = regression_result
        corr_dict[i] = correlation
    return [lin_reg_df_dict, corr_dict]

def survey_data_preprocessing():
    DATA_PATH = './data/물가/설문조사/'
    YEARS = [2018, 2019, 2020, 2021]
    info_dict = {}
    dis_dict = {}
    budget_dict = {}

    for YEAR in YEARS:
        temp_info = pd.read_excel(DATA_PATH + f'{YEAR}년 정보습득경로.xlsx')
        temp_info = survey_drop_row(survey_rename_df(temp_info), YEAR)
        info_dict[YEAR] = temp_info

    for YEAR in YEARS:
        temp_dis = pd.read_excel(DATA_PATH + f'{YEAR}년 여행 시 불만족점.xlsx')
        temp_dis = survey_drop_row(survey_rename_df(temp_dis), YEAR)
        temp_dis['연도'] = YEAR
        dis_dict[YEAR] = temp_dis
    
    for YEAR in YEARS:
        temp_budget = pd.read_excel(DATA_PATH + f'{YEAR}년 제주여행 평가_여행경비.xlsx')
        if YEAR == 2019:
            del temp_budget['Unnamed: 8']
            del temp_budget['Unnamed: 12']
        else:
            pass
        temp_budget.columns = ['분석 카테고리', '세부 카테고리', '사례수', '매우 불만족', '불만족', '보통', '만족', '매우 만족', '매우 불만족 & 불만족', 'mid', '매우 만족 & 만족', '[5점 평균]', '[100점 평균]']
        temp_budget = temp_budget.drop(temp_budget.index[0]).reset_index(drop = True)
        temp_budget = survey_drop_row(temp_budget, YEAR)
        budget_dict[YEAR] = temp_budget
    
    return [info_dict, dis_dict, budget_dict]

def survey_rename_df(df):
    df.iloc[0, 0] = '분석 카테고리'
    df.iloc[0, 1] = '세부 카테고리'
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop = True)
    return df

def survey_drop_row(df, YEAR):
    if YEAR == 2020:
        df = df.drop(index = range(0, 3)).reset_index(drop = True)
    else:
        df = df.drop(index = 0).reset_index(drop = True)

    if YEAR == 2021:
        df = df.drop(index = range(9, 44)).reset_index(drop = True)
    elif YEAR == 2020:
        df = df.drop(index = range(7, 42)).reset_index(drop = True)
    else:
        df = df.drop(index = range(12, 45)).reset_index(drop = True)

    return df

def cluster_df():
    dis_dict = survey_data_preprocessing()[1]
    merge_dis = pd.concat(dis_dict.values(), axis = 0).reset_index(drop = True)
    features = merge_dis.iloc[:, 3:-1]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    dis_df = pd.DataFrame(scaled_features, columns = features.columns)
    kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, random_state = 0)
    kmeans.fit(dis_df)
    dis_df['cluster'] = kmeans.labels_
    pca = PCA(n_components = 2)
    pca_transformed = pca.fit_transform(dis_df)
    dis_df['pca_x'] = pca_transformed[:, 0]
    dis_df['pca_y'] = pca_transformed[:, 1]

    score_samples = silhouette_samples(dis_df, dis_df['cluster'])
    dis_df['silhouette_coeff'] = score_samples
    average_score = silhouette_score(dis_df, dis_df['cluster'])
    st.markdown('- Silhouette Analyses Score:{0:.3f}'.format(average_score))
    dis_df = dis_df.sort_values(by = 'cluster').reset_index(drop = True)

    return dis_df

def fa_df():
    merge_budget = pd.concat(survey_data_preprocessing()[2], axis = 0)
    merge_budget = merge_budget.reset_index(drop = True)
    price = pd.read_csv('./data/물가/제주물가.csv')
    price = price.iloc[:40, :]
    price['전국대비물가'] = price['제주-전국'].apply(lambda x: '높음' if x > 0 else '낮음')
    price = price['전국대비물가']

    columns = ['매우 불만족', '불만족', '보통', '만족', '매우 만족']
    budget_df = merge_budget[columns]
    budget_df = pd.concat([budget_df, price], axis = 1)
    
    x = budget_df.drop(columns = ['전국대비물가'])
    # 요인 분석 모델 생성
    fa = FactorAnalyzer(n_factors = 2, method = 'ml', rotation = None)  # 요인 수 및 다른 설정을 조정 가능
    # 모델 적합
    fa.fit(x)
    efa_result = pd.DataFrame(fa.loadings_, index = columns)
    return efa_result