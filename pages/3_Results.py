import streamlit as st
import visualizations as v
import pandas as pd
import xgboost as xgb
import joblib

st.set_page_config(page_title="Results", page_icon="💡", layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("Results💡")

accuracies_cls = pd.read_csv('Data/accuracies_cls.csv')
accuracies_reg = pd.read_csv('Data/accuracies_reg.csv')


col_l, col_r = st.columns([1, 1], gap="medium")
with col_l:
    st.header('1991', divider='rainbow')
    col_l1, col_r1 = st.columns([1, 1], gap="medium")
    
    with col_l1:
        st.subheader("Classifier")
        st.dataframe(accuracies_cls[['Model','Year 91']],use_container_width=True)
        st.subheader("Accuracy")
        fig = v.plot_model_accuracy(accuracies_cls['Year 91'].to_list())
        st.plotly_chart(fig, use_container_width= True)
        fig = v.plot_feature_importance(pd.read_csv('Data/features_importance_cls_91.csv'))
        st.plotly_chart(fig, use_container_width= True) 

    with col_r1:
        st.subheader("Regressor")
        st.dataframe(accuracies_reg[['Model','Year 91']],use_container_width=True,column_config = {'Year 91': st.column_config.NumberColumn(format="%.10f")})
        st.subheader("Mean Squared Error")
        fig = v.plot_model_accuracy(accuracies_reg['Year 91'].to_list())
        st.plotly_chart(fig, use_container_width= True)
        fig = v.plot_feature_importance(pd.read_csv('Data/features_importance_reg_91.csv'))
        st.plotly_chart(fig, use_container_width= True) 
 

with col_r:
    st.header('2001',divider='rainbow')
    col_l1, col_r1 = st.columns([1, 1], gap="medium")
    
    with col_l1:
        st.subheader("Classifier")
        st.dataframe(accuracies_cls[['Model','Year 01']],use_container_width=True)
        st.subheader("Accuracy")
        fig = v.plot_model_accuracy(accuracies_cls['Year 01'].to_list())
        st.plotly_chart(fig, use_container_width= True)
        fig = v.plot_feature_importance(pd.read_csv('Data/features_importance_cls_01.csv'))
        st.plotly_chart(fig, use_container_width= True) 

    with col_r1:
        st.subheader("Regressor")
        st.dataframe(accuracies_reg[['Model','Year 01']],use_container_width=True,column_config = {'Year 01': st.column_config.NumberColumn(format="%.10f")})
        st.subheader("Mean Squared Error")
        fig = v.plot_model_accuracy(accuracies_reg['Year 01'].to_list())
        st.plotly_chart(fig, use_container_width= True)
        fig = v.plot_feature_importance(pd.read_csv('Data/features_importance_reg_01.csv'))
        st.plotly_chart(fig, use_container_width= True) 

    


