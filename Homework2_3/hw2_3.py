from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofmonth, hour, when, dayofweek
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, FloatType
import folium
from folium.plugins import HeatMapWithTime
import pandas as pd

sc = SparkContext("local", "Homework 2")
spark = SparkSession(sc)

# Схема данных
schema = StructType([
    StructField("tripduration", IntegerType()),
    StructField("starttime", TimestampType()),
    StructField("stoptime", TimestampType()),
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

start_station_count_by_day = data.groupBy('start station id', dayofmonth('starttime').alias('day')).agg({"starttime": "count"}).orderBy('day')
end_station_count_by_day = data.groupBy('end station id', dayofmonth('stoptime').alias('day')).agg({"stoptime": "count"}).orderBy('day')

mean_start_by_day = start_station_count_by_day.groupBy('start station id').agg({'count(starttime)': 'mean'}).withColumnRenamed('avg(count(starttime))', 'mean start count')
mean_end_by_day = end_station_count_by_day.groupBy('end station id').agg({'count(stoptime)': 'mean'}).withColumnRenamed('avg(count(stoptime))', 'mean end count')

res_mean_by_day = mean_start_by_day.join(mean_end_by_day, (mean_start_by_day['start station id'] == mean_end_by_day['end station id']), 'inner')
res_mean_by_day = res_mean_by_day.select('start station id', 'mean start count', 'mean end count').withColumnRenamed('start station id', 'station id')
print("Среднее количество начала и окончания поездок по станциям за день")
res_mean_by_day.show()

data = data.withColumn("starthour", hour(data['starttime']))
data = data.withColumn("start time of day", 
                   when((data.starthour >= 6) & (data.starthour <= 11), "morning")
                   .when((data.starthour >= 12) & (data.starthour <= 17), "afternoon")
                   .when((data.starthour >= 18) & (data.starthour <= 23), "evening")
                   .otherwise("night"))
data = data.withColumn("endhour", hour(data['stoptime']))
data = data.withColumn("end time of day", 
                   when((data.endhour >= 6) & (data.endhour <= 11), "morning")
                   .when((data.endhour >= 12) & (data.endhour <= 17), "afternoon")
                   .when((data.endhour >= 18) & (data.endhour <= 23), "evening")
                   .otherwise("night"))


count_by_day_start = data.withColumn('startday', dayofmonth(data['starttime'])).groupBy('start time of day', 'start station id', 'startday').count().orderBy('startday','start station id')
mean_starts = count_by_day_start.groupBy('start station id', 'start time of day').agg({"count": "mean"}).withColumnRenamed('avg(count)', 'mean starts')

start_stations_coords = data.select('start station id', 'start station latitude', 'start station longitude').distinct()

mean_starts = mean_starts.join(start_stations_coords, ['start station id'], 'left')
print("Среднее значение стартов по станциям по временным диапазонам")
mean_starts.orderBy('start station id', 'start time of day').show()

count_by_day_end = data.withColumn('endday', dayofmonth(data['stoptime'])).groupBy('end time of day', 'end station id', 'endday').count().orderBy('endday','end station id')
mean_ends = count_by_day_end.groupBy('end station id', 'end time of day').agg({"count": "mean"}).withColumnRenamed('avg(count)', 'mean ends')

end_stations_coords = data.select('end station id', 'end station latitude', 'end station longitude').distinct()

mean_ends = mean_ends.join(end_stations_coords, ['end station id'], 'left')
print("Среднее значение окончаний поездок по станциям по временным диапазонам")
mean_ends.orderBy('end station id', 'end time of day').show()

data = data.withColumn('day of week', dayofweek(data['starttime']))
data_wed = data.filter(data['day of week'] == 4)
data_sun = data.filter(data['day of week'] == 1)

wed_count_by_day_start = data_wed.withColumn('startday', dayofmonth(data_wed['starttime'])).groupBy('start time of day', 'start station id', 'startday').count().orderBy('startday','start station id')
wed_mean_starts = wed_count_by_day_start.groupBy('start station id', 'start time of day').agg({"count": "mean"}).withColumnRenamed('avg(count)', 'mean starts')
wed_count_by_day_end = data_wed.withColumn('endday', dayofmonth(data_wed['stoptime'])).groupBy('end time of day', 'end station id', 'endday').count().orderBy('endday','end station id')
wed_mean_ends = wed_count_by_day_end.groupBy('end station id', 'end time of day').agg({"count": "mean"}).withColumnRenamed('avg(count)', 'mean ends')
wed_res_df = wed_mean_starts.join(wed_mean_ends, (wed_mean_starts['start station id'] == wed_mean_ends['end station id']) & (wed_mean_starts['start time of day'] == wed_mean_ends['end time of day']), 'inner')
wed_res_df = wed_res_df.select('start station id', 'start time of day', 'mean starts', 'mean ends').withColumnRenamed('start station id', 'station id').withColumnRenamed('start time of day', 'time of day')
print("Среднее количество начала и окончания поездок по средам по временным интервалам:")
wed_res_df.orderBy('station id', 'time of day').show()

sun_count_by_day_start = data_sun.withColumn('startday', dayofmonth(data_sun['starttime'])).groupBy('start time of day', 'start station id', 'startday').count().orderBy('startday','start station id')
sun_mean_starts = sun_count_by_day_start.groupBy('start station id', 'start time of day').agg({"count": "mean"}).withColumnRenamed('avg(count)', 'mean starts')
sun_count_by_day_end = data_sun.withColumn('endday', dayofmonth(data_sun['stoptime'])).groupBy('end time of day', 'end station id', 'endday').count().orderBy('endday','end station id')
sun_mean_ends = sun_count_by_day_end.groupBy('end station id', 'end time of day').agg({"count": "mean"}).withColumnRenamed('avg(count)', 'mean ends')
sun_res_df = sun_mean_starts.join(sun_mean_ends, (sun_mean_starts['start station id'] == sun_mean_ends['end station id']) & (sun_mean_starts['start time of day'] == sun_mean_ends['end time of day']), 'inner')
sun_res_df = sun_res_df.select('start station id', 'start time of day', 'mean starts', 'mean ends').withColumnRenamed('start station id', 'station id').withColumnRenamed('start time of day', 'time of day')
print("Среднее количество начала и окончания поездок по воскресеньям по временным интервалам:")
sun_res_df.orderBy('station id', 'time of day').show()

# mean_starts = mean_starts.toPandas()
# m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
# data = []
# for time in mean_starts["start time of day"].unique():
#     data.append(list(zip(mean_starts[mean_starts["start time of day"]==time]["start station latitude"],
#                          mean_starts[mean_starts["start time of day"]==time]["start station longitude"],
#                          mean_starts[mean_starts["start time of day"]==time]["mean starts"])))
# plugins.HeatMapWithTime(data, index=list(mean_starts["start time of day"].unique()), auto_play=True).add_to(m)
# m.save('starts.html')
# m

# Закрытие Spark
sc.stop()