from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


spark = SparkSession.builder \
    .appName("SparkTensorFlowBatchTraining") \
    .master("yarn") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()


df = spark.read.csv("hdfs://path/to/large_dataset.csv", header=True, inferSchema=True)


assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)


pipeline = Pipeline(stages=[assembler, scaler])


processed_df = pipeline.fit(df).transform(df)


batch_size = 1000
num_partitions = processed_df.rdd.getNumPartitions()
batches = processed_df.repartition(num_partitions).rdd.mapPartitions(lambda rows: [list(rows)[i:i + batch_size] for i in range(0, len(list(rows)), batch_size)])


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),  # لایه مخفی با 16 نورون
    tf.keras.layers.Dense(8, activation='relu'),                     # لایه مخفی با 8 نورون
    tf.keras.layers.Dense(1, activation='sigmoid')                   # لایه خروجی برای دسته‌بندی باینری
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epoches = 1000
for j in range(epoches):
    for batch in batches.collect():
        batch_data = [(row["features"], row["label"]) for row in batch]
        if not batch_data:
            continue
        X_batch = np.array([row[0] for row in batch_data])
        y_batch = np.array([row[1] for row in batch_data])


        model.fit(X_batch, y_batch, epochs=1, batch_size=32, verbose=1)


test_sample = processed_df.sample(fraction=0.1, seed=1234).collect()
X_test = np.array([row["features"] for row in test_sample])
y_test = np.array([row["label"] for row in test_sample])


y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")


new_data = spark.createDataFrame([(2.5, 3.5, 1.5)], ["feature1", "feature2", "feature3"])
new_processed = pipeline.fit(df).transform(new_data)
new_features = np.array([row["features"] for row in new_processed.select("features").collect()])
new_pred = (model.predict(new_features) > 0.5).astype(int)
print(f"Prediction for new data: {new_pred[0]}")


spark.stop()