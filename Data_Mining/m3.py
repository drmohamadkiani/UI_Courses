from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline


spark = SparkSession.builder \
    .appName("LogisticRegressionExample") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()


data = [
    (0, 1.0, 2.0, 3.0),
    (0, 2.0, 1.0, 4.0),
    (1, 3.0, 4.0, 1.0),
    (1, 4.0, 3.0, 2.0),
    (0, 1.5, 2.5, 3.5)
]
columns = ["label", "feature1", "feature2", "feature3"]
df = spark.createDataFrame(data, columns)


assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features_raw")


scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)


lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.01)


pipeline = Pipeline(stages=[assembler, scaler, lr])


train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)


model = pipeline.fit(train_data)


predictions = model.transform(test_data)


evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Area Under ROC: {auc:.3f}")


new_data = spark.createDataFrame([(2.5, 3.5, 1.5)], ["feature1", "feature2", "feature3"])
new_predictions = model.transform(new_data)
new_predictions.select("features", "prediction", "probability").show()


spark.stop()