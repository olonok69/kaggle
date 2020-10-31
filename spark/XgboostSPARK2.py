####################################################################
# Hyperparamether Tunning XGboost/PYSPARK
#
# https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html#pipeline-with-hyper-parameter-tunning
# 
#
#
#####################################################################

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell' # probably no need of this import as we defined as paramether on the submit

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd

#Session and configuration parameter run from command line spark-submit --jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar XgboostSPARK2.py

spark = SparkSession.builder.appName("PySpark XGBOOST").config("spark.jars", "xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.addPyFile("sparkxgb.zip")

#Dataset structure
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

#read the observations file
df_raw = spark.read.option("header", "true").schema(schema).csv("train.csv")
#fill NaNs
df = df_raw.na.fill(0)
# Clean UP
sexIndexer = StringIndexer().setInputCol("Sex").setOutputCol("SexIndex").setHandleInvalid("keep")
cabinIndexer = StringIndexer().setInputCol("Cabin").setOutputCol("CabinIndex").setHandleInvalid("keep")
embarkedIndexer = StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex").setHandleInvalid("keep")

#Define vector of Features
vectorAssembler = VectorAssembler().setInputCols(["Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "CabinIndex", "EmbarkedIndex"]).setOutputCol("features")

from xgboost import XGBoostEstimator, XGBoostClassificationModel
from pipeline import XGBoostPipeline

######################
#  cv
######################


#Transform imput dataset to fit the model

pipeline = Pipeline().setStages([sexIndexer, cabinIndexer, embarkedIndexer, vectorAssembler])
pipelineFit = pipeline.fit(df)
dataset = pipelineFit.transform(df)

#Split transformed dataset
trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=24)

# Define Xgboostestimator just with default paramethers and mandatory features and labels columns if they have different names from standard
xg = XGBoostEstimator(featuresCol="features", labelCol="Survival")

#Define metric , default ROC

evaluator = BinaryClassificationEvaluator(rawPredictionCol='Survival', labelCol='prediction')

# Create pipeline of XG Estimators for CV
pipelineXG=Pipeline().setStages([xg])

# Create ParamGrid for Cross Validation. Personalice to your space of search and your problem

paramGrid = (ParamGridBuilder()
			 .addGrid(xg.min_child_weight, [.1, .2, .3]) # regularization parameter
             .addGrid(xg.eta, [0.2, 0.5, 0.7,0.9]) # Elastic Net Parameter (Ridge = 0)
			 .addGrid(xg.max_depth, [1, 2, 3]) #Number of iterations
        	 .addGrid(xg.alpha, [0.01, 0.05, 0.1])# Number of features
			 .addGrid(xg.subsample, [.6,.65,.7])
			 .addGrid(xg.gamma,[0.0, 0.01, 0.02])
             .build())

			 # Create 5-fold CrossValidator
cv = CrossValidator(estimator=pipelineXG, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)


# fit the Crossvalidation objetc . Be aware of using the PipelineObject with XGboost estimators . Data shall be already transformed 
cvmodel = cv.fit(trainDF)
bestModel = cvmodel.bestModel
print(str(bestModel))

# Evaluate best model
predictions = cvmodel.transform(testDF)
print("\n AREA UNDER ROC")
print(str(evaluator.evaluate(predictions)))
print("\n")

#from the best model Print all paramethers maps 
bestPipeline = cvmodel.bestModel
bestLRModel = bestPipeline.stages[-1]
bestParams = bestLRModel.extractParamMap()
#print(str(bestParams))

java_model = bestPipeline.stages[-1]._java_obj
parameters={param.name: java_model.getOrDefault(java_model.getParam(param.name)) for param in paramGrid[0]}

print("Best Hyperparamethers")
print(parameters)

# Evaluate best model, ACCURACY
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="Survival")
print("Accuracy: "+ str(evaluator.evaluate(predictions)))

# print the importance of each feature on the model. !! for XGBOOST doesnt work

#model = pd.DataFrame(cvmodel.bestModel.stages[-1].featureImportances.toArray(), columns=["values"])
#features_col = pd.Series(features)
#model["features"] = features_col
#print(model)


spark.stop()

