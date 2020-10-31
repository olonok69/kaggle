

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'



from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
#from xgboost import XGBoostEstimator

spark = SparkSession.builder.appName("PySpark XGBOOST").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


spark.sparkContext.addPyFile("sparkxgb.zip")


schema = StructType(
  [StructField("PassengerId", DoubleType()),
    StructField("Survival", DoubleType()),
    StructField("Pclass", DoubleType()),
    StructField("Name", StringType()),
    StructField("Sex", StringType()),
    StructField("Age", DoubleType()),
    StructField("SibSp", DoubleType()),
    StructField("Parch", DoubleType()),
    StructField("Ticket", StringType()),
    StructField("Fare", DoubleType()),
    StructField("Cabin", StringType()),
    StructField("Embarked", StringType())
  ])


df_raw = spark.read.option("header", "true").schema(schema).csv("train.csv")
df = df_raw.na.fill(0)

sexIndexer = StringIndexer().setInputCol("Sex").setOutputCol("SexIndex").setHandleInvalid("keep")
cabinIndexer = StringIndexer().setInputCol("Cabin").setOutputCol("CabinIndex").setHandleInvalid("keep")
embarkedIndexer = StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex").setHandleInvalid("keep")

vectorAssembler = VectorAssembler().setInputCols(["Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "CabinIndex", "EmbarkedIndex"]).setOutputCol("features")

from xgboost import XGBoostEstimator

xgboost = XGBoostEstimator(featuresCol="features", labelCol="Survival", predictionCol="prediction", alpha=0.05, min_child_weight=0.1, eta=0.7, max_depth= 1)

pipeline = Pipeline().setStages([sexIndexer, cabinIndexer, embarkedIndexer, vectorAssembler, xgboost])


trainDF, testDF = df.randomSplit([0.8, 0.2], seed=24)

trainDF.show()
testDF.show()

model = pipeline.fit(trainDF)
predictions=model.transform(testDF)

predictions.select("PassengerId", "prediction", "Survival").show(n = 10, truncate = 30)


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="Survival")
print("Accuracy: "+str(evaluator.evaluate(predictions)))




