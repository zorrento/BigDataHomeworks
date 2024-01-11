from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, max, mean, std, median
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType
from math import radians, sin, cos, sqrt, atan2

def calculate_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    difflng = lng2 - lng1
    difflat = lat2 - lat1
    a = sin(difflat / 2)**2 + cos(lat1) * cos(lat2) * sin(difflng / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371.0
    dist = R * c
    return dist*1000

sc = SparkContext("local", "Homework 2")
spark = SparkSession(sc)

# Схема данных
schema = StructType([
    StructField("tripduration", IntegerType()),
    StructField("starttime", DateType()),
    StructField("stoptime", DateType()),
    StructField("start station id", IntegerType()),
    StructField("start station name", StringType()),
    StructField("start station latitude", FloatType()),
    StructField("start station longitude", FloatType()),
    StructField("end station id", IntegerType()),
    StructField("end station name", StringType()),
    StructField("end station latitude", FloatType()),
    StructField("end station longitude", FloatType()),
    StructField("bikeid", IntegerType()),
    StructField("usertype", StringType()),
    StructField("birth year", IntegerType()),
    StructField("gender", IntegerType())
])

# Чтение данных из CSV
data = spark.read.csv("file:///home/ubuntu/HW1/Part1/201902-citibike-tripdata.csv", header=True, schema=schema)

# Регистрация функции в качестве UDF в Spark
calculate_distance_udf = udf(calculate_distance, FloatType())

data = data.filter(data['start station id'] != data['end station id']).withColumn('distance', calculate_distance_udf('start station latitude', 'start station longitude', 'end station latitude', 'end station longitude'))
max_dist = data.select(max(data.distance)).collect()[0][0]
mean_dist = data.select(mean(data.distance)).collect()[0][0]
std_dist = data.select(std(data.distance)).collect()[0][0]
median_dist = data.select(median(data.distance)).collect()[0][0]

print(f"Макс. дистанция: {max_dist}\nСредняя дистанция: {mean_dist}\nСтандартное отклонение дистанции: {std_dist}\nМедиана дистанции {median_dist}")

# Закрытие Spark
sc.stop()