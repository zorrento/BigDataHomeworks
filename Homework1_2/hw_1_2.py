import csv
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType


spark_session = SparkSession(SparkContext("local", "HomeWork 1.2"))
ratings_schema = StructType([
    StructField("item", StringType()),
    StructField("user", StringType()),
    StructField("rating", DoubleType()),
    StructField("timestamp", IntegerType())
])
meta_schema = StructType([
    StructField("asin", StringType()),
    StructField("title", StringType()),
    StructField("feature", StringType()),
    StructField("description", StringType()),
    StructField("price", StringType()),
    StructField("imageURL", StringType()),
    StructField("imageURLHighRes", StringType()),
    StructField("also_buy", ArrayType(elementType=StringType())),
    StructField("also_viewed", ArrayType(elementType=StringType())),
    StructField("salesRank", StringType()),
    StructField("brand", StringType()),
    StructField("tech1", StringType()),
    StructField("tech2", StringType()),

])
ratings = spark_session.read.csv("file:///home/ubuntu/Homeworks/1_2/Appliances.csv", header=False, schema=ratings_schema)
meta_data = spark_session.read.json("file:///home/ubuntu/Homeworks/1_2/meta_Appliances.json", schema=meta_schema)

print("Средний рейтинг товаров:")
print(ratings.rdd.map(lambda x: (x["rating"])).reduce(lambda x, y: x + y) / ratings.count())

with open("result.csv", 'w', newline='') as re:
    writer = csv.writer(re)
    writer.writerow(['rating', 'title'])
    result = ratings.rdd.map(
                lambda x: (x["item"], x["rating"])
            ).filter(
                lambda x: float(x[1]) < 3.0
            ).join(
                meta_data.rdd.
                map(lambda x: (x["asin"], x["title"])).
                filter(lambda x: x[0] != "" and x[1] != "").
                map(lambda x: (x[0], x[1]))
            ).map(
                lambda x: (x[1], x[1])
            ).reduceByKey(
                lambda x, y: x
            ).map(
                lambda x: x[0]
            ).sortBy(
                lambda x: x[0]
            ).take(10)
    print("Топ-10 товаров с наименьшим рейтингом:")
    for item in result:
        print(item)
    writer.writerows(result)
spark_session.stop()
