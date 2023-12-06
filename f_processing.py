from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType,DoubleType
from pyspark.sql.functions import when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, MinMaxScaler, VectorAssembler
from pyspark.sql.functions import col



# schema_91 = StructType([
#     StructField('Year', IntegerType(), True),
#     StructField('Month', IntegerType(), True),
#     StructField('DayofMonth', IntegerType(), True),
#     StructField('DayOfWeek', IntegerType(), True),
#     StructField('DepTime', DoubleType(), True),
#     StructField('CRSDepTime', IntegerType(), True),
#     StructField('ArrTime', DoubleType(), True),
#     StructField('CRSArrTime', IntegerType(), True),
#     StructField('UniqueCarrier', StringType(), True),
#     StructField('FlightNum', IntegerType(), True),
#     StructField('TailNum', StringType(), True),
#     StructField('ActualElapsedTime', IntegerType(), True),
#     StructField('CRSElapsedTime', DoubleType(), True),
#     StructField('AirTime', DoubleType(), True),
#     StructField('ArrDelay', DoubleType(), True),
#     StructField('DepDelay', DoubleType(), True),
#     StructField('Origin', StringType(), True),
#     StructField('Dest', StringType(), True),
#     StructField('Distance', IntegerType(), True),
#     StructField('Cancelled', DoubleType(), True),
#     StructField('Diverted', DoubleType(), True),
# ])
# schema_01 = StructType([
#     StructField('Year', IntegerType(), True),
#     StructField('Month', IntegerType(), True),
#     StructField('DayofMonth', IntegerType(), True),
#     StructField('DayOfWeek', IntegerType(), True),
#     StructField('DepTime', DoubleType(), True),
#     StructField('CRSDepTime', IntegerType(), True),
#     StructField('ArrTime', DoubleType(), True),
#     StructField('CRSArrTime', IntegerType(), True),
#     StructField('UniqueCarrier', StringType(), True),
#     StructField('FlightNum', IntegerType(), True),
#     StructField('TailNum', StringType(), True),
#     StructField('ActualElapsedTime', DoubleType(), True),
#     StructField('CRSElapsedTime', DoubleType(), True),
#     StructField('AirTime', DoubleType(), True),
#     StructField('ArrDelay', DoubleType(), True),
#     StructField('DepDelay', DoubleType(), True),
#     StructField('Origin', StringType(), True),
#     StructField('Dest', StringType(), True),
#     StructField('Distance', IntegerType(), True),
#     StructField('TaxiIn', IntegerType(), True),
#     StructField('TaxiOut', IntegerType(), True),
#     StructField('Cancelled', DoubleType(), True),
#     StructField('Diverted', DoubleType(), True),
# ])

# Creating a SparkSession
spark = SparkSession.builder \
    .appName('my_app') \
    .config('spark.master', 'local[*]') \
    .getOrCreate()

# # Setting the log level to ERROR
spark.sparkContext.setLogLevel("ERROR")


def load_data():
    df_91 = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("compression", "gzip") \
        .load('Data/1991.csv.gz') 

    df_01 = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("compression", "gzip") \
        .load('Data/2001.csv.gz')
    
    return df_91,df_01

def preprocessing(df91, df01):
    return df91.withColumn("status", when(df91["ArrDelay"] > 1, "1").otherwise("0")), df01.withColumn("status", when(df01["ArrDelay"] > 1, "1").otherwise("0"))

def preprocess_data(df, categorical_cols, numerical_cols):
    # Remove null values
    for col_name in df.columns:
        df = df.filter(col(col_name).isNotNull())

    df = df.drop(*['Year','ArrTime','ArrDelay'])
    # Indexing and One-Hot Encoding for categorical columns
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
        for col_name in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_encoded")
        for col_name in categorical_cols
    ]

    # Assemble all encoded categorical columns and numerical columns into a single feature vector
    assembler = VectorAssembler(
        inputCols=[f"{col_name}_encoded" for col_name in categorical_cols] + numerical_cols,
        outputCol="features"
    )

    # MinMax scaling for numerical columns
    scalers = [
        MinMaxScaler(inputCol=col_name, outputCol=f"{col_name}_scaled")
        for col_name in numerical_cols
    ]

    # Assemble all stages into a pipeline
    stages = indexers + encoders + [assembler] + scalers
    pipeline = Pipeline(stages=stages)

    # Fit the pipeline to the DataFrame
    pipeline_model = pipeline.fit(df)

    # Transform the DataFrame using the fitted pipeline
    transformed_df = pipeline_model.transform(df)

    return transformed_df