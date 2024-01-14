from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

spark = SparkSession.builder.appName("4.1").getOrCreate()
movies_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True)
])
ratings_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", DoubleType(), True),
    StructField("timestamp", IntegerType(), True)
])

# movies_df = spark.read.csv("file:///home/ubuntu/Homeworks/4/ml-latest-small/movies.csv", header=True, schema=movies_schema)
# ratings_df = spark.read.csv("file:///home/ubuntu/Homeworks/4/ml-latest-small/ratings.csv", header=True, schema=ratings_schema)
movies_df = spark.read.csv("file:///home/ubuntu/Homeworks/4/ml-latest/movies.csv", header=True, schema=movies_schema)
ratings_df = spark.read.csv("file:///home/ubuntu/Homeworks/4/ml-latest/ratings.csv", header=True, schema=ratings_schema)

movies_df = movies_df.withColumn("genres", F.split("genres", "\\|"))
movies_df = movies_df.select("movieId", "title", F.explode("genres").alias("genres"))

selected_genres = ['Drama', 'Comedy', 'Musical']
genres_filtered = movies_df.filter(F.col("genres").isin(selected_genres))

genre_counts = genres_filtered.groupBy("genres").count().orderBy("count", ascending=False)

print("=======================================================================")
print("Жанры и количество фильмов:")
genre_counts.show()

# Merge ratings and movies DataFrames
merged_df = ratings_df.join(movies_df, on="movieId")

# Group by genres, movieId, and title, then aggregate ratings information
genre_ratings_info = merged_df.groupBy("genres", "movieId", "title").agg(
    F.mean("rating").alias("average_rating"),
    F.count("rating").alias("num_ratings")
).filter("num_ratings > 10").orderBy("average_rating", ascending=False)


print("=======================================================================")
print("Первые 10 фильмов с наибольшим количеством рейтингов для каждого жанра:")
for genre in selected_genres:
    top_rated_films = genre_ratings_info.filter(F.col("genres") == genre).orderBy("num_ratings", ascending=False).limit(10)
    print(f"\nТоп 10 фильмов жанра {genre} с наибольшим числом рейтингов:")
    top_rated_films.select("movieId", "title", "num_ratings").show(truncate=False)

print("=======================================================================")
print("Первые 10 фильмов с наименьшим количеством рейтингов для каждого жанра (если рейтингов больше 10):")
for genre in selected_genres:
    least_rated_films = genre_ratings_info.filter((F.col("genres") == genre) & (F.col("num_ratings") > 10)).orderBy("num_ratings").limit(10)
    print(f"\nТоп 10 фильмов жанра {genre} с наименьшим числом рейтингов:")
    least_rated_films.select("movieId", "title", "num_ratings").show(truncate=False)

print("=======================================================================")
print("Первые 10 фильмов с наибольшим средним рейтингом при количестве рейтингов больше 10 для каждого жанра:")
for genre in selected_genres:
    print(f"\nТоп 10 фильмов жанра {genre} с наибольшим средним рейтингом:")
    top_films = genre_ratings_info.filter(F.col("genres") == genre).limit(10)
    top_films.show(truncate=False)

print("=======================================================================")
print("Первые 10 фильмов с наименьшим средним рейтингом при количестве рейтингов больше 10 для каждого жанра:")
for genre in selected_genres:
    print(f"\nТоп 10 фильмов жанра {genre} с наименьшим средним рейтингом:")
    anti_top_films = genre_ratings_info.filter(F.col("genres") == genre).orderBy("average_rating").limit(10)
    anti_top_films.show(truncate=False)

# Stop the Spark session
spark.stop()
