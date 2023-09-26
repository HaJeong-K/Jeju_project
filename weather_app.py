# -*- coding:utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

def vis_app():
    final = data_preprocessing()
    fig, ax1 = plt.subplots(figsize = (12, 6))
    ax1.bar(final['구분연월'], final['강수량(mm)'], color = 'blue', alpha = 0.7, label = '강수량(mm)')
    ax1.set_ylabel('강수량(mm)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 90)
    ax1.legend(loc = 'upper left', bbox_to_anchor = (0.0, 0.94))
    ax1.grid(False)
    ax2 = ax1.twinx()
    ax2.plot(final['구분연월'], final['전체 방문객'], color = 'red', marker = 'o', label = '전체 방문객')
    ax2.set_ylabel('전체 방문객')
    ax2.set_title('강수량과 전체 방문객 추이')
    ax2.legend(loc = 'upper left')
    ax2.ticklabel_format(axis = 'y', style = 'plain')
    ax2.grid(False)
    st.pyplot(fig)

def data_app():
    st.markdown('데이터')
    st.dataframe(data_preprocessing(), hide_index = True)

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
        df_temp['전체 방문객'] = df_temp.iloc[:, -3:].sum(axis = 1)
        df_temp = df_temp.loc[:, ['구분연월', '전체 방문객']]
        df_dict[i] = df_temp
    df_5years = pd.concat(df_dict.values(), ignore_index = True)
    
    weather = pd.read_excel('./data/제주 강수량.xlsx')
    weather = weather.drop(index = range(0, 13)).reset_index(drop = True)
    column_names = {'Unnamed: 2': '일시', 'Unnamed: 3': '강수량(mm)', 'Unnamed: 4': '일최다강수량(mm)'}
    weather = weather.rename(columns = column_names)
    weather = weather.drop(columns = ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'])
    weather[['년', '월', '나머지']] = weather['일시'].astype(str).str.split('-', n = 2, expand = True)
    weather['구분연월'] = weather['년'] + '-' + weather['월']
    weather = weather.drop(columns = ['일시', '년', '월', '나머지'])
    final = df_5years.merge(weather, on = '구분연월', how = 'inner')

    return final