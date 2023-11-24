import streamlit as st
import pandas as pd
import numpy as np
from numerize.numerize import numerize
import functions 


st.set_page_config(page_title="Business Intelligence", page_icon=":airplane:", layout="wide", initial_sidebar_state="auto", menu_items=None)
with st.spinner("Loading"):
    @st.cache_resource
    def load_data():
        return pd.read_csv("Data/1991.csv.gz",encoding='ANSI', compression="gzip").dropna(axis=1, how='all').reset_index(drop=True), pd.read_csv("Data/2001.csv.gz",encoding='ANSI', compression="gzip").dropna(axis=1, how='all').reset_index(drop=True)

df_91, df_01 = load_data()

filter_1, filter_2, filter_3,filter_4, filter_5 = st.columns([1,1,1,1,1], gap="medium")
@st.cache_resource
def getFilters():
    return np.union1d(df_91['Origin'].unique(), df_01['Origin'].unique()), np.union1d(df_91['Dest'].unique(),df_01['Dest'].unique()),np.union1d(df_91['UniqueCarrier'].unique(), df_01['UniqueCarrier'].unique()),df_91['Month'].unique(), df_01['DayofMonth']
                    

origin_options,dest_options,airline_options,months_options, days_options = getFilters()


with filter_1: 
    airline_filter = st.multiselect(label="Select Airline", options=list(airline_options))

with filter_2: 
    origin_filter = st.multiselect(label="Select Origin", options=list(origin_options))

with filter_3: 
    destination_filter = st.multiselect(label="Select Destination", options=list(dest_options))

with filter_4: 
    month_filter = st.multiselect(label="Select Month", options= months_options)

with filter_5: 
    day_filter = st.multiselect(label="Select Day", options= days_options)


# df = pd.concat([df_91, df_01]) if '1991' in dataset_filter and '2001' in dataset_filter else df_91 if '1991' in dataset_filter else df_01 if '2001' in dataset_filter else pd.DataFrame()

query_filters = []

if airline_filter:
    query_filters.append(f'UniqueCarrier in {airline_filter}')

if origin_filter:
    query_filters.append(f'Origin in {origin_filter}')
    
if destination_filter:
    query_filters.append(f'Dest in {destination_filter}')

if month_filter:
    query_filters.append(f'Month in {month_filter}')

if day_filter:
    query_filters.append(f'DayofMonth in {day_filter}')

# Build the query condition
query_condition = ' & '.join(query_filters)

df_q_91 = df_91.query(query_condition) if query_condition else df_91
df_q_01 = df_01.query(query_condition) if query_condition else df_01

st.divider()
r_c1, r_c2 = st.columns([1,1], gap="small")

with r_c1:
    st.title("1991")
    # st.image('Images/91.png',use_column_width=True)
    r1_c1, r1_c2, r1_c3,r1_c4,r1_c5,r1_c6 = st.columns([1,1,1,1,1,1], gap="small")
    r2_c1, r2_c2 = st.columns([1,1], gap="small")
    total = float(len(df_q_91['Year']))
    delayed = float(len(df_q_91[df_q_91['ArrDelay'] > 0]))
    non_delayed = float(len(df_q_91[df_q_91['ArrDelay'] <= 0]))

    
    with r1_c1:
        st.image('Images/airline.png')
        st.metric(label="Airlines", value = df_q_91['UniqueCarrier'].nunique())
    with r1_c2:
        st.image('Images/origin.png', width=100)
        st.metric(label="Origins", value = df_q_91['Origin'].nunique())
    with r1_c3:
        st.image('Images/dest.png', width=100)
        st.metric(label="Destination", value = df_q_91['Dest'].nunique())

    with r1_c4:
        st.image('Images/total.png')
        st.metric(label="Total", value = numerize(total))

    with r1_c5:
        st.image('Images/status.png')
        st.metric(label="Non Delayed", value = numerize(non_delayed))
    with r1_c6:
        st.image('Images/delayed.png', width=100)
        st.metric(label="Origins", value = numerize(delayed))

    with r2_c1:
        st.plotly_chart(functions.delayed_vs_non_delayed_flights_chart(df_q_91),use_container_width=True)

with r_c2:
    st.title("2001")
    # st.image('Images/01.png',use_column_width=True)
    r1_c1, r1_c2, r1_c3,r1_c4,r1_c5,r1_c6 = st.columns([1,1,1,1,1,1], gap="small")
    r2_c1, r2_c2 = st.columns([1,1], gap="small")
    total = float(len(df_q_01['Year']))
    delayed = float(len(df_q_01[df_q_01['ArrDelay'] > 0]))
    non_delayed = float(len(df_q_01[df_q_01['ArrDelay'] <= 0]))
    
    with r1_c1:
        st.image('Images/airline.png')
        st.metric(label="Airlines", value = df_q_01['UniqueCarrier'].nunique())
    with r1_c2:
        st.image('Images/origin.png', width=100)
        st.metric(label="Origins", value = df_q_01['Origin'].nunique())
    with r1_c3:
        st.image('Images/dest.png', width=100)
        st.metric(label="Destination", value = df_q_01['Dest'].nunique())

    with r1_c4:
        st.image('Images/total.png')
        st.metric(label="Total", value = numerize(total))

    with r1_c5:
        st.image('Images/status.png')
        st.metric(label="Non Delayed", value = numerize(non_delayed))
    with r1_c6:
        st.image('Images/delayed.png', width=100)
        st.metric(label="Origins", value = numerize(delayed))

    with r2_c1:
        st.plotly_chart(functions.delayed_vs_non_delayed_flights_chart(df_q_01),use_container_width=True)
        

# marg_l,col_1, col_2, col_3, marg_r = st.columns([1,12,12,12,1], gap="small")

# if df_q_91.shape[0] > 0:
#     with col_1:
#         st.plotly_chart(functions.delayed_vs_non_delayed_flights_chart(df_q_91),use_container_width=True)
# if df_q_91.shape[0] > 0:
#     with col_2:
#         st.plotly_chart(functions.delayed_vs_non_delayed_flights_chart(df_q_91),use_container_width=True)
# if df_q_91.shape[0] > 0:
#     with col_3:
#         st.plotly_chart(functions.delayed_vs_non_delayed_flights_chart(df_q_91),use_container_width=True)
