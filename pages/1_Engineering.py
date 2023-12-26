import streamlit as st
import pandas as pd
import visualizations as dv

st.set_page_config(page_title="Data Engineering", page_icon="⚙️", layout="wide", initial_sidebar_state="auto", menu_items=None,)



st.title("Data Engineering ⚙️")


@st.cache_data
def load_data():
    cols_91 = ['Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','UniqueCarrier','FlightNum','ActualElapsedTime','CRSElapsedTime','ArrDelay','DepDelay','Origin','Dest','Distance','Cancelled','Diverted']
    cols_01 = ['Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','UniqueCarrier','FlightNum','TailNum','ActualElapsedTime','CRSElapsedTime','ArrDelay','DepDelay','Origin','Dest','Distance','TaxiIn','TaxiOut','Cancelled','Diverted']
    return pd.read_csv("Data/1991.csv.gz", encoding='cp1252', compression="gzip", usecols=cols_91), pd.read_csv("Data/2001.csv.gz", encoding='cp1252', compression="gzip", usecols=cols_01)


@st.cache_data
def load_data_t():
    return pd.read_csv('Data/sdf91.csv'),  pd.read_csv('Data/sdf01.csv'), pd.read_csv('Data/pdf91.csv'),  pd.read_csv('Data/pdf01.csv')

df91, df01 = load_data()

sdf91, sdf01, pdf91, pdf01 = load_data_t()
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
st.header("Correlation Analysis")
col_1l, col_1r = st.columns([1, 1], gap="medium")
with col_1l:
    st.plotly_chart(dv.correlation_heatmap_plot(df91))

with col_1r:
    st.plotly_chart(dv.correlation_heatmap_plot(df01))


st.header("Processed Data")
col_1l, col_1r = st.columns([1, 1], gap="medium")
with col_1l:
    st.dataframe(pdf91.sample(20))

with col_1r:
    st.dataframe(pdf01.sample(20))
