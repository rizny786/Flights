import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Engineering", page_icon="⚙️", layout="wide", initial_sidebar_state="auto", menu_items=None,)



st.title("Data Engineering ⚙️")

@st.cache_data
def load_data():
    return pd.read_csv('Data/sdf91.csv'),  pd.read_csv('Data/sdf01.csv'), pd.read_csv('Data/pdf91.csv'),  pd.read_csv('Data/pdf01.csv')

sdf91, sdf01, pdf91, pdf01 = load_data()
col_l, col_r = st.columns([1, 1], gap="medium")
with col_l:
    st.header('1991', divider='rainbow')

with col_r:
    st.header('2001',divider='rainbow')


st.header("Actual data")
col_1l, col_1r = st.columns([1, 1], gap="medium")
with col_1l:
    st.dataframe(sdf91.sample(20))

with col_1r:
    st.dataframe(sdf01.sample(20))

st.markdown('''
            ### Data Preprocessing Steps:
                1. Removing repetative information
                2. Remove null values
                3. Label encoding for Categorical columns
                4. Normalizing numerical columns
                5. Define target variable 
                6. Convert target into Binary class
            ''')

st.header("Processed Data")
col_1l, col_1r = st.columns([1, 1], gap="medium")
with col_1l:
    st.dataframe(pdf91.sample(20))

with col_1r:
    st.dataframe(pdf01.sample(20))
