from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col
import numpy as np

def cosine_sim(rating1, rating2):
    return np.dot(rating1, rating2) / (np.linalg.norm(rating1) * np.linalg.norm(rating2))

spark_session = SparkSession(SparkContext("local", "1.3"))
schema_ratings = StructType([
    StructField("userId", IntegerType()),
    StructField("movieId", IntegerType()),
    StructField("rating", DoubleType()),
    StructField("timestamp", IntegerType())
])
schema_movies = StructType([
    StructField("movieId", IntegerType()),
    StructField("title", StringType()),
    StructField("genres", StringType())
])

ratings = spark_session.read.csv("file:///home/ubuntu/Homeworks/1_3/ratings.csv", header=True, schema=schema_ratings)
movies = spark_session.read.csv("file:///home/ubuntu/Homeworks/1_3/movies.csv", header=True, schema=schema_movies)

movies_vectors = ratings.groupBy("movieId").agg({"rating": "avg"}).rdd.map(lambda row: (row["movieId"], np.array([row["avg(rating)"]])))
target_mov_vector = ratings.filter(ratings.movieId == 589).rdd.map(lambda row: np.array([row["rating"]])).first()[0]

similarities_rdd = movies_vectors.map(lambda x:  (x[0], cosine_sim(x[1], target_mov_vector))).map(lambda x: (x[0], float(x[1])))
result_rdd = similarities_rdd.toDF(["SimilarMovieId", "Similarity"]) \
    .join(movies, col("SimilarMovieId") == col("movieId")) \
    .select("SimilarMovieId", "title", col("Similarity").cast(DoubleType())) \
    .orderBy(col("Similarity"), ascending=False)
print("Топ-10 наиболее похожих фильмов")
result_rdd.show(10, truncate=False)
spark_session.stop()
