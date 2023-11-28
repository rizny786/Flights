import streamlit as st
from app import df_01, df_91

st.set_page_config(page_title="Data Exploration", page_icon="🔍", layout="wide", initial_sidebar_state="auto", menu_items=None,)
st.markdown("# Data Exploration 🔍")

@st.cache_data
def load_data():
    st.dataframe(df_01.head(10))

def eda():
    st.title("show me")    

load_data()