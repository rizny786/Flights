import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
from numerize.numerize import numerize
from pyspark.storagelevel import StorageLevel
from pyspark.sql import functions as F

st.set_page_config(page_title="Data Exploration", page_icon="🔍", layout="wide", initial_sidebar_state="auto", menu_items=None,)
st.markdown("# Data Exploration 🔍")

# Creating a SparkSession
spark = SparkSession.builder \
    .appName('my_app') \
    .config('spark.master', 'local[*]') \
    .getOrCreate()

# # Setting the log level to ERROR
spark.sparkContext.setLogLevel("ERROR")

# Setting the timeParserPolicy to LEGACY for backward compatibility
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")



@st.cache_resource
def load_data():
    # Read the gzip files into PySpark DataFrames
    df_1991 = spark.read.option("header", True).option("inferSchema", True).option("encoding", "cp1252").option("compression", "gzip").csv("../Data/1991.csv.gz")
    df_2001 = spark.read.option("header", True).option("inferSchema", True).option("encoding", "cp1252").option("compression", "gzip").csv("../Data/2001.csv.gz")
    return df_1991, df_2001

def eda():
    st.title("show me")    

df_91, df_01 = load_data()
df_91.write.csv("df_1991.csv.gz",header="true", mode="overwrite")
