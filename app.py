# -*- coding:utf-8 -*-

import streamlit as st
import os
import matplotlib.font_manager as fm
import base64
from recommend_app import run_recommend_app
from data_app import run_data_app
from reference_app import run_reference_app
from about_app import run_about_app

# @st.cache_data
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = """
#     <style>
#     st.App {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     """ % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/customFonts']
    font_files = fm.findSystemFonts(fontpaths = font_dirs)

    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache = False)

def main():
    page_bg_img = """
    <style>
    [title~="st.iframe"] {width: 100%}

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="block-container"] {
        background-color: CadetBlue;
        background-size: contain;
        background-position: top;
        background-repeat: no-repeat;
    }

    [data-testid="stVerticalBlock"] {
        background-color: Ivory;
    }
    
    [data-testid="stSidebar"] {
        background-image: url("https://github.com/HWANHEECHO/practice/assets/139515758/55339303-7b7a-4908-b8d9-d372da038d28");
        background-size: contain;
        background-position: bottom;
        background-repeat: no-repeat;
        background-color: Ivory;
    }
    """
    st.set_page_config(layout = 'wide')
    st.markdown(page_bg_img, unsafe_allow_html = True)
    # set_png_as_page_bg('./data/streamlit/background.png')

    fontRegistered()
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
    
    # st.sidebar.image('./data/streamlit/image_01.png', use_column_width = True)
        # background-position = bottom;
        # background-image: url("data:image/png;base64,{img}");
        # background-size: contain;
        # background-position: bottom;
        # background-repeat: no-repeat;

    # [data-testid="stAppViewContainer"] {
    #     background-image: url("http://www.jeju.go.kr/pub/site/jejuwnh/images/sub/sub01_img04.png");
    #     background-size: 100% 100%;
    # }

if __name__ == '__main__':
    main()