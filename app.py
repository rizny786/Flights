import streamlit as st
import pandas as pd
import numpy as np
from numerize.numerize import numerize
import visualizations as dv
from st_pages import Page, show_pages


show_pages(
    [
        Page("app.py", "Exploration", "🔍"),
        Page("pages/1_Engineering.py", "Engineering", "⚙️"),
        Page("pages/2_Modeling.py", "Modeling", "🤖 "),
        Page("pages/3_Results.py", "Results", "💡"),
    ]
)

st.set_page_config(
    page_title="Business Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed")

st.markdown("# Dashboard ✈️")

@st.cache_data
def load_data():
    cols_91 = ['Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','UniqueCarrier','FlightNum','ActualElapsedTime','CRSElapsedTime','ArrDelay','DepDelay','Origin','Dest','Distance','Cancelled','Diverted']
    cols_01 = ['Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','UniqueCarrier','FlightNum','TailNum','ActualElapsedTime','CRSElapsedTime','ArrDelay','DepDelay','Origin','Dest','Distance','TaxiIn','TaxiOut','Cancelled','Diverted']
    return pd.read_csv("Data/1991.csv.gz", encoding='cp1252', compression="gzip", usecols=cols_91), pd.read_csv("Data/2001.csv.gz", encoding='cp1252', compression="gzip", usecols=cols_01)

with st.spinner("Loading"):
    df_91, df_01 = load_data()

    col_af, col_of, col_df, col_mf, col_dmf = st.columns(
        [1, 1, 1, 1, 1], gap="medium")

    with col_af:
        airline_filter = st.multiselect(
            label="Select Airline", options=np.union1d(df_91['UniqueCarrier'].unique(), df_01['UniqueCarrier'].unique()))

    with col_of:
        origin_filter = st.multiselect(
            label="Select Origin", options=np.union1d(df_91['Origin'].unique(), df_01['Origin'].unique()))

    with col_df:
        destination_filter = st.multiselect(
            label="Select Destination", options=np.union1d(df_91['Dest'].unique(), df_01['Dest'].unique()))

    with col_mf:
        month_filter = st.multiselect(label="Select Month", options=np.sort(df_91['Month'].unique()))

    with col_dmf:
        day_filter = st.multiselect(label="Select Day", options=np.sort(df_01['DayofMonth'].unique()))


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
    r_c1, r_m, r_c2 = st.columns([50,5, 50], gap="small")

    with r_c1:
        st.title("1991")
        # st.image('Images/91.png',use_column_width=True)
        r1_c1, r1_c2, r1_c3, r1_c4, r1_c5, r1_c6 = st.columns(
            [1, 1, 1, 1, 1, 1], gap="small")
        r2_c1, r2_c2 = st.columns([1, 1], gap="small")
        r3_c1, r3_c2 = st.columns([1, 1], gap="small")
        total = float(len(df_q_91['Year']))
        delayed = float(len(df_q_91[df_q_91['ArrDelay'] > 0]))
        non_delayed = float(len(df_q_91[df_q_91['ArrDelay'] <= 0]))

        with r1_c1:
            st.image('Images/airline.png')
            st.metric(label="Airlines", value=df_q_91['UniqueCarrier'].nunique())
        with r1_c2:
            st.image('Images/origin.png', width=100)
            st.metric(label="Origins", value=df_q_91['Origin'].nunique())
        with r1_c3:
            st.image('Images/dest.png', width=100)
            st.metric(label="Destination", value=df_q_91['Dest'].nunique())

        with r1_c4:
            st.image('Images/total.png')
            st.metric(label="Total", value=numerize(total))

        with r1_c5:
            st.image('Images/delayed.png', width=100)
            st.metric(label="Origins", value=numerize(delayed))
        with r1_c6:
            st.image('Images/status.png')
            st.metric(label="Non Delayed", value=numerize(non_delayed))

        with r2_c1:
            st.plotly_chart(dv.pie_chart_delayed_vs_non_delayed(
                df_q_91), use_container_width=True)
        with r2_c2:
            st.plotly_chart(dv.by_airline(
                df_q_91), use_container_width=True)

        with r3_c1:
            st.plotly_chart(dv.by_origin(
                df_q_91), use_container_width=True)
        with r3_c2:
            st.plotly_chart(dv.by_dest(
                df_q_91), use_container_width=True)


    with r_c2:
        st.title("2001")
        # st.image('Images/01.png',use_column_width=True)
        r1_c1, r1_c2, r1_c3, r1_c4, r1_c5, r1_c6 = st.columns(
            [1, 1, 1, 1, 1, 1], gap="small")
        r2_c1, r2_c2 = st.columns([1, 1], gap="small")
        r3_c1, r3_c2 = st.columns([1, 1], gap="small")
        total = float(len(df_q_01['Year']))
        delayed = float(len(df_q_01[df_q_01['ArrDelay'] > 0]))
        non_delayed = float(len(df_q_01[df_q_01['ArrDelay'] <= 0]))

        with r1_c1:
            st.image('Images/airline.png')
            st.metric(label="Airlines", value=df_q_01['UniqueCarrier'].nunique(),)
        with r1_c2:
            st.image('Images/origin.png', width=100)
            st.metric(label="Origins", value=df_q_01['Origin'].nunique())
        with r1_c3:
            st.image('Images/dest.png', width=100)
            st.metric(label="Destination", value=df_q_01['Dest'].nunique())

        with r1_c4:
            st.image('Images/total.png')
            st.metric(label="Total", value=numerize(total))

        with r1_c5:
            st.image('Images/delayed.png', width=100)
            st.metric(label="Delayed", value=numerize(delayed))

        with r1_c6:
            st.image('Images/status.png')
            st.metric(label="Non Delayed", value=numerize(non_delayed))

        with r2_c1:
            st.plotly_chart(dv.pie_chart_delayed_vs_non_delayed(
                df_q_01), use_container_width=True)
        with r2_c2:
            st.plotly_chart(dv.by_airline(
                df_q_01), use_container_width=True)
        with r3_c1:
            st.plotly_chart(dv.by_origin(
                df_q_01), use_container_width=True)
        with r3_c2:
            st.plotly_chart(dv.flight_trend_by_airline(df_q_01), use_container_width=True)
            # st.plotly_chart(dv.by_dest(
            #     df_q_01), use_container_width=True)