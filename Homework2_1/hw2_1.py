import json
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, udf, sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, DateType, FloatType, MapType
import geopandas as gpd
from shapely.geometry import shape, Point
import plotly.express as px

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

# Чтение данных из geojson
with open("NYC Taxi Zones.geojson") as f:
    nyc_geojson = json.load(f)

# Создание полигонов для каждого района
polygons = {}
for feature in nyc_geojson['features']:
    zone_name = feature['properties']['zone']
    polygons[zone_name] = shape(feature['geometry'])

# Функция для определения района по координатам
def find_zone(latitude, longitude):
    point = Point(longitude, latitude)
    for zone_name, polygon in polygons.items():
        if polygon.contains(point):
            return zone_name
    return None
# Регистрация функции в качестве UDF в Spark
find_zone_udf = udf(find_zone, StringType())

# определите для каждой станции количество начала поездок и количество завершения поездок
start_count = data.groupBy("start station id", "start station latitude", "start station longitude").count()
end_count = data.groupBy("end station id", "end station latitude", "end station longitude").count()

# сопоставьте станции с кварталами города (zones) и определите суммы количества начала и завершения для каждого квартала (НАЧАЛО)
start_stations_data = data.select('start station id', 'start station latitude', 'start station longitude').distinct()
start_stations_data = start_stations_data.withColumn("start station zone", find_zone_udf(col('start station latitude'), col('start station longitude')))
start_stations_data = start_stations_data.join(start_count, (start_stations_data["start station id"] == start_count["start station id"]) &
                                               (start_stations_data["start station latitude"] == start_count["start station latitude"])&
                                               (start_stations_data["start station longitude"] == start_count["start station longitude"]), 'right')
count_by_zone_start = start_stations_data.groupBy('start station zone').sum('count')

# сопоставьте станции с кварталами города (zones) и определите суммы количества начала и завершения для каждого квартала (ОКОНЧАНИЕ)
end_stations_data = data.select('end station id', 'end station latitude', 'end station longitude').distinct()
end_stations_data = end_stations_data.withColumn("end station zone", find_zone_udf(col('end station latitude'), col('end station longitude')))
end_stations_data = end_stations_data.join(end_count, (end_stations_data["end station id"] == end_count["end station id"]) &
                                               (end_stations_data["end station latitude"] == end_count["end station latitude"])&
                                               (end_stations_data["end station longitude"] == end_count["end station longitude"]), 'right')
count_by_zone_end = end_stations_data.groupBy('end station zone').sum('count')

print("Количество начала поездок и количество завершения поездок для каждой станции:")
# выведите по убыванию количества поездок -  по отдельности начало/окончание
count_by_zone_start.orderBy(count_by_zone_start['sum(count)'].desc()).show(truncate=False)
count_by_zone_end.orderBy(count_by_zone_end['sum(count)'].desc()).show(truncate=False)

# объединение данных
count_by_zone_start = count_by_zone_start.withColumnRenamed('sum(count)', 'start count')
count_by_zone_end = count_by_zone_end.withColumnRenamed('sum(count)', 'end count')
count_all = count_by_zone_start.join(count_by_zone_end, (count_by_zone_start['start station zone'] == count_by_zone_end['end station zone']), 'fullouter')\
    .select('start station zone', 'start count', 'end count')
count_all = count_all.dropna('any').withColumnRenamed('start station zone', 'station zone')
count_all = count_all.withColumn('start plus end', count_all['start count'] + count_all['end count']).select('station zone', 'start plus end')

print("\n\nСтанции по убыванию количества поездок:")
# выведите по убыванию количества поездок - объединенные данные
count_all.orderBy(count_all['start plus end'].desc()).show(truncate=False)

# отобразите в виде картограмм (Choropleth) - открывается карта США - необходимо приблизить к Нью-Йорку
fig = px.choropleth(count_all.toPandas(), geojson=nyc_geojson, color='start plus end', locations='station zone', 
                    labels={'start plus end': 'count'}, featureidkey='properties.zone', scope='usa')
fig.show()

# Закрытие Spark
sc.stop()