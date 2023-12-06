import streamlit as st
import f_processing as dp

st.set_page_config(layout="wide")

@st.cache_resource
def processing():
    df1, df2 = dp.load_data()
    return dp.preprocessing(df1,df2)


st.title("Data Preprocessing")
df_91, df_01 = processing()

col_1l, col_1r = st.columns([1, 1], gap="medium")

with col_1l:
    st.dataframe(df_91.limit(10))

with col_1r:
    st.dataframe(df_01.limit(10))

st.write(df_91.dtypes)

# pdf_91 = dp.preprocess_data(df_91,['UniqueCarrier','TailNum','Origin','Dest'],['Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','CRSArrTime','FlightNum','ActualElapsedTime','CRSElapsedTime','DepDelay','Distance','Cancelled','Diverted'])

# pdf_01 = dp.preprocess_data(df_01,['UniqueCarrier','TailNum','Origin','Dest'],['Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','CRSArrTime','FlightNum','ActualElapsedTime','CRSElapsedTime','DepDelay','Distance','TaxiIn','TaxiOut','Cancelled','Diverted'])


# col_2l, col_2r = st.columns([1, 1], gap="medium")
# with col_2l:
#     st.dataframe(pdf_91.limit(10))

# with col_2r:
#     st.dataframe(pdf_01.limit(10))