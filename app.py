# -*- coding:utf-8 -*-

import streamlit as st
from recommend_app import run_recommend_app
from data_app import run_data_app
from reference_app import run_reference_app
from about_app import run_about_app

def main():
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("http://www.jeju.go.kr/pub/site/jejuwnh/images/sub/sub01_img04.png");
        background-size: 100% 100%;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="stVerticalBlock"] {
        background-color: Ivory;
    }
    
    [data-testid="stSidebar"] {
        background-image: url("https://gongu.copyright.or.kr/gongu/wrt/cmmn/wrtFileImageView.do?wrtSn=13296755&filePath=L2Rpc2sxL25ld2RhdGEvMjAyMS8yMS9DTFMxMDAwNC8xMzI5Njc1NV9XUlRfMjFfQ0xTMTAwMDRfMjAyMTEyMTNfMQ==&thumbAt=Y&thumbSe=b_tbumb&wrtTy=10004");
        background-size: contain;
        background-position: bottom;
        background-repeat: no-repeat;
        background-color: Ivory;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html = True)

    st.title('Jeju_project')

    menu = ['Home', '분석내용', '자료출처', 'About']
    choice = st.sidebar.selectbox('메뉴', menu)

    if choice == 'Home':
        run_recommend_app()
    elif choice == '분석내용':
        run_data_app()
    elif choice == '자료출처':
        run_reference_app()
    elif choice == 'About':
        run_about_app()
    else:
        pass

if __name__ == '__main__':
    main()