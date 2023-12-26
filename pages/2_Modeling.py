import streamlit as st

st.set_page_config(page_title="Modeling", page_icon="🤖", layout="wide", initial_sidebar_state="auto", menu_items=None,)


st.title("Modeling 🤖")

l1, l2, r1, r2 = st.columns([1, 2, 1, 2], gap="small")
with l2:
    st.header("Decision Tree")
with r2:
    st.header("Extreme Gradient Boosting ")

c1, c2 = st.columns([1,1], gap="small")
with c1: 
    st.subheader("Classifier")
    st.markdown('''
                A Decision Tree Classifier is a supervised learning algorithm used for classification tasks. It recursively splits the dataset into subsets based on the features, 
                aiming to create a tree-like structure of decision nodes. Each node represents a feature and each branch represents a possible decision rule. 
                The model is easy to interpret and visualize, allowing for an intuitive understanding of the decision-making process. Feature importance in decision trees is derived 
                from how frequently or deeply a feature is used to make decisions across various branches. Analyzing feature importance helps identify the most influential features 
                in predicting the target class, providing insights into the underlying relationships within the dataset.
                ''')
with c2: 
    st.subheader("Classifier") 
    st.markdown('''An advanced and popular implementation of gradient boosting algorithms. It excels in performance due to its efficient computation, regularization techniques, and 
                handling missing values. XGBoost builds multiple weak learners (decision trees) sequentially, where each subsequent tree corrects the errors made by the previous one. 
                It automatically determines feature importance by evaluating the gain of each feature across all splits in all trees. The higher the gain, the more important the feature is in predicting the target class.
                ''')    


c1, c2 = st.columns([1,1], gap="small")
with c1:
    st.subheader("Regressor")    
    st.markdown('''
                A Decision Tree Regressor is used for regression tasks. Instead of predicting classes, it predicts continuous values. It works by partitioning the dataset into subsets
                     and fitting a simple model (like a constant value) for each subset. Decision tree regressors are also interpretable and easy to visualize. Feature importance in 
                    regression trees is calculated by assessing how much each feature contributes to reducing the variance within the data. It helps in understanding which features have the most impact on predicting the target variable..
                ''')
with c2: 
    st.subheader("Regressor") 
    st.markdown('''
                XGBoost Regressor applies gradient boosting for regression problems. It works well with large datasets and is robust against overfitting. XGBoost determines feature 
                importance by calculating the average gain of each feature across all trees in the ensemble. This allows the identification of the most influential features in predicting the continuous target variable.    
                ''')


