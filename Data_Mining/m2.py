from pyspark.sql import SparkSession


spark_local = SparkSession.builder \
    .appName("LocalApp") \
    .master("local[2]") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()
print(spark_local.sparkContext.getConf().getAll())


spark_cluster = SparkSession.builder \
    .appName("ClusterApp") \
    .master("yarn") \
    .config("spark.executor.memory", "4g") \
    .config("spark.submit.deployMode", "cluster") \
    .getOrCreate()
print(spark_cluster.sparkContext.getConf().getAll())

spark_local.stop()
spark_cluster.stop()