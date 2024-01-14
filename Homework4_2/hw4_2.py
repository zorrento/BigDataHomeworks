from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("CollaborativeFiltering").getOrCreate()

#ratings_df = spark.read.csv("file:///home/ubuntu/Homeworks/4/ml-latest-small/ratings.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("file:///home/ubuntu/Homeworks/4/ml-latest/ratings.csv", header=True, inferSchema=True)

train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

average_rating = train_df.selectExpr("avg(rating) as average_rating").collect()[0]["average_rating"]
print(f"Среднее значение рейтинга в обучающем подмножестве: {average_rating}")

test_df = test_df.withColumn("predicted_rating", F.lit(average_rating))

evaluator = RegressionEvaluator(labelCol="rating", predictionCol="predicted_rating", metricName="rmse")
rmse = evaluator.evaluate(test_df)
print(f"RMSE для тестового подмножества (для всех значений из test предсказывается среднее): {rmse}")

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)
model = als.fit(train_df)

predictions = model.transform(test_df)

evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE для тестового подмножества (для коллаборативной фильтрации): {rmse}")

spark.stop()
