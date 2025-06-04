from pyspark.sql import SparkSession

import pyspark as spark
from pyspark import SparkConf, SparkContext


conf = SparkConf().setAppName("RDDExample").setMaster("local")

sc = SparkContext(conf=conf)


data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)
squared = rdd.map(lambda x: x * x)
result = squared.collect()
print(result)


result = squared.collect()
print(result)


data = [(1, "Ali"), (2, "Sara")]
df = spark.createDataFrame(data, ["id", "name"])


df.filter(df.id > 1).show()

spark.stop()