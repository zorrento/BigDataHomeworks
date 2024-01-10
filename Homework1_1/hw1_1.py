from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from math import radians, sin, cos, sqrt, atan2


def calculate_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    difflng = lng2 - lng1
    difflat = lat2 - lat1
    a = sin(difflat / 2)**2 + cos(lat1) * cos(lat2) * sin(difflng / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371.0
    dist = R * c
    return dist


sc = SparkContext("local", "Homework 1.1")
spark = SparkSession(sc)

# Schema
schema = StructType([
    StructField("ID", IntegerType()),
    StructField("Name", StringType()),
    StructField("global_id", IntegerType()),
    StructField("IsNetObject", StringType()),
    StructField("OperatingCompany", StringType()),
    StructField("TypeObject", StringType()),
    StructField("AdmArea", StringType()),
    StructField("District", StringType()),
    StructField("Address", StringType()),
    StructField("PublicPhone", StringType()),
    StructField("SeatsCount", IntegerType()),
    StructField("SocialPrivileges", StringType()),
    StructField("Longitude_WGS84", DoubleType()),
    StructField("Latitude_WGS84", DoubleType()),
    StructField("geoData", StringType())
])

data = spark.read.csv("file:///home/ubuntu/Homeworks/Homework1_1/places.csv", header=False, schema=schema)
target_lat, target_lng = 55.751244, 37.618423

# Рассчитайте расстояние от заданной точки (lat=55.751244, lng=37.618423) до каждого заведения общепита из набора данных. Выведите первые 10.
target_dist = data.rdd.map(lambda row: (
    row["ID"],
    row["Name"],
    calculate_distance(row["Latitude_WGS84"], row["Longitude_WGS84"], target_lat, target_lng)
))

print("Расстояние от заданной точки до каждого заведения общепита из набора данных (первые 10):")
for dist in target_dist.takeOrdered(10, key=lambda x: x[2]):
    print(f"{dist[1]}: {dist[2]}")

# Рассчитайте расстояние между всеми заведениями общепита из набора данных. Выведите первые 10.
all_distances = data.rdd.cartesian(data.rdd).filter(lambda x: x[0]["ID"] < x[1]["ID"]).map(lambda x: (
    (x[0]["Name"], x[1]["Name"]),
    calculate_distance(x[0]["Latitude_WGS84"], x[0]["Longitude_WGS84"], x[1]["Latitude_WGS84"], x[1]["Longitude_WGS84"])
))

print("\n\nРасстояние между всеми заведениями общепита из набора данных (первые 10):")
for dist in all_distances.takeOrdered(10, key=lambda x: x[1]):
    print(f"{dist[0]}: {dist[1]}")

# Выведите топ-10 наиболее близких и наиболее отдаленных заведений.
print("\n\nТоп-10 наиболее близких и наиболее отдаленных заведений:")
for dist in all_distances.takeOrdered(10, key=lambda x: x[1]) + all_distances.takeOrdered(10, key=lambda x: -x[1]):
    print(f"{dist[0]}: {dist[1]}")

sc.stop()
