import streamlit as st
import visualizations as v
import pandas as pd
import xgboost as xgb
import joblib

st.set_page_config(page_title="Results", page_icon="💡", layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("Results💡")

accuracies = pd.read_csv('Data/accuracies.csv')


col_l, col_r = st.columns([1, 1], gap="medium")
with col_l:
    st.header('1991', divider='rainbow')
    st.dataframe(accuracies[['Model','Year 91']],use_container_width=True)
    col_1l, col_1r = st.columns([1, 1], gap="medium")
   
    fig = v.plot_model_accuracy(accuracies['Year 91'].to_list())
    st.plotly_chart(fig, use_container_width= True)
   
    fig = v.plot_feature_importance(pd.read_csv('Data/features_importance_91.csv'))
    st.plotly_chart(fig, use_container_width= True) 
    
    

with col_r:
    st.header('2001',divider='rainbow')
    st.dataframe(accuracies[['Model','Year 01']],use_container_width=True)

    fig = v.plot_model_accuracy(accuracies['Year 01'].to_list())
    st.plotly_chart(fig, use_container_width= True)

    fig = v.plot_feature_importance(pd.read_csv('Data/features_importance_01.csv'))
    st.plotly_chart(fig, use_container_width= True)

    


