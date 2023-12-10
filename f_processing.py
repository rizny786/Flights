from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType,DoubleType
from pyspark.sql.functions import when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, MinMaxScaler, VectorAssembler
from pyspark.sql.functions import col



schema_91 = StructType([
    StructField("Year", IntegerType(), True),
    StructField("Month", IntegerType(), True),
    StructField("DayofMonth", IntegerType(), True),
    StructField("DayOfWeek", IntegerType(), True),
    StructField("DepTime", DoubleType(), True),
    StructField("CRSDepTime", IntegerType(), True),
    StructField("ArrTime", DoubleType(), True),
    StructField("CRSArrTime", IntegerType(), True),
    StructField("UniqueCarrier", StringType(), True),
    StructField("FlightNum", IntegerType(), True),
    StructField("ActualElapsedTime", DoubleType(), True),
    StructField("CRSElapsedTime", IntegerType(), True),
    StructField("ArrDelay", DoubleType(), True),
    StructField("DepDelay", DoubleType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("Distance", DoubleType(), True),
    StructField("Cancelled", IntegerType(), True),
    StructField("Diverted", IntegerType(), True)
])

schema_01 = StructType([
    StructField('Year', IntegerType(), True),
    StructField('Month', IntegerType(), True),
    StructField('DayofMonth', IntegerType(), True),
    StructField('DayOfWeek', IntegerType(), True),
    StructField('DepTime', DoubleType(), True),
    StructField('CRSDepTime', IntegerType(), True),
    StructField('ArrTime', DoubleType(), True),
    StructField('CRSArrTime', IntegerType(), True),
    StructField('UniqueCarrier', StringType(), True),
    StructField('FlightNum', IntegerType(), True),
    StructField('TailNum', StringType(), True),
    StructField('ActualElapsedTime', DoubleType(), True),
    StructField('CRSElapsedTime', DoubleType(), True),
    StructField('AirTime', DoubleType(), True),
    StructField('ArrDelay', DoubleType(), True),
    StructField('DepDelay', DoubleType(), True),
    StructField('Origin', StringType(), True),
    StructField('Dest', StringType(), True),
    StructField('Distance', IntegerType(), True),
    StructField('TaxiIn', IntegerType(), True),
    StructField('TaxiOut', IntegerType(), True),
    StructField('Cancelled', DoubleType(), True),
    StructField('Diverted', DoubleType(), True),
])


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
        .schema(schema_91) \
        .load('Data/1991.csv.gz') 

    df_01 = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("compression", "gzip") \
        .schema(schema_01) \
        .load('Data/2001.csv.gz')
    
    return df_91,df_01

def preprocessing(df91, df01):
    return df91.withColumn("status", when(df91["ArrDelay"] > 1, 1).otherwise(0)), df01.withColumn("status", when(df01["ArrDelay"] > 1, 1).otherwise(0))

def preprocess_data(df, categorical_cols, numerical_cols):
    # Remove null values
    all_columns = list()
    for col_name in df.columns:
        all_columns.append(col_name)
        df = df.filter(col(col_name).isNotNull())

    # Indexing and One-Hot Encoding for categorical columns
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="keep")
        for col_name in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_encoded")
        for col_name in categorical_cols
    ]

    # Assemble all encoded categorical columns and numerical columns into a single feature vector
    assembler = VectorAssembler(
        inputCols=[col_name + "_encoded" for col_name in categorical_cols] + [num_cols for num_cols in numerical_cols],
        outputCol="features"
    )

    # Assemble all stages into a pipeline
    stages = indexers + encoders + [assembler]
    pipeline = Pipeline(stages=stages)

    # Fit the pipeline to the DataFrame
    pipeline_model = pipeline.fit(df)

    # Transform the DataFrame using the fitted pipeline
    transformed_df = pipeline_model.transform(df)

    transformed_df = transformed_df.drop(*[col_name + "_index" for col_name in categorical_cols] + [col_name + "_encoded" for col_name in categorical_cols]) \
    .withColumnRenamed("status", "target") \
    
    # Combine categorical_cols, numerical_cols, 'features', and 'target' into a single list
    selected_columns = categorical_cols + numerical_cols + ['features', 'target']

    # Construct the select expression using list comprehension and col() function
    select_expr = [col(column) for column in selected_columns]
    
    return transformed_df.select(select_expr)


