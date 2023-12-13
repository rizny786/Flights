import streamlit as st

st.set_page_config(page_title="Modeling", page_icon="🤖", layout="wide", initial_sidebar_state="auto", menu_items=None,)


st.title("Modeling 🤖")

l1, l2, r1, r2 = st.columns([1, 2, 2, 1], gap="small")
with l2:
    st.header("Decision Tree Classifier")
    

with r1:
    st.header("Extreme Gradient Boosting Classifer")
    



l1, m, r2 = st.columns([1, 2, 1], gap="small")

