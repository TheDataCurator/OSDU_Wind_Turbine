# Databricks notebook source
# MAGIC %md 
# MAGIC # Wind Turbine Predictive Maintenance 
# MAGIC 
# MAGIC *tldr: In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines.*

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Overview 
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection to identify signals that a wind turbine will fail. A single, damaged, inactive wind turbine costs energy utility companies thousands of dollars per day in losses. A 2016 study of unplanned downtime by Baker-Hughes states unexpected equipment failure costs Energy companies an estimated $38 million in losses per year, and an average of approximately 27 days of unplanned downtime. 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Dataset
# MAGIC Our dataset is generated within the workbook based on vibration readings from sensors located in the gearboxes of wind turbines. The gearbox is the most costly component in the drivetrain to maintain throughout the 20 year lifespan of a wind turbine. The original data set was collected as part of a study by the National Renewable Energy Laboratory (NREL) to research the root causes of premature failure of wind turbines. 
# MAGIC 
# MAGIC **Turbine Sensor Locations**
# MAGIC 
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/databricks-demo-images/wind_turbine/wind_small.png" width=400 />
# MAGIC 
# MAGIC **Turbine Sensor Descriptions**
# MAGIC 
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/databricks-demo-images/wind_turbine/wtsmall.png" width=400 />
# MAGIC 
# MAGIC **Original Dataset Details**
# MAGIC   * Dataset title: Wind Turbine Gearbox Condition Monitoring Vibration Analysis Benchmarking Datasets
# MAGIC   * Dataset source URL: https://data.openei.org/submissions/738
# MAGIC   * Dataset attribution: US Department of Energy: National Renewable Energy Laboratory (NREL)
# MAGIC   * Relevant Papers: S. Sheng, Editor, 'Wind Turbine Gearbox Condition
# MAGIC Monitoring Round Robin Study – Vibration Analysis', National Renewable Energy Laboratory, 2012 https://www.nrel.gov/docs/fy12osti/54530.pdf

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### OSDU schema for WindTurbines

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Template for masterdata
# MAGIC An example of a custom schema created by Pariveda for this project for windturbine maintenance for the OSDU portal created is "osdu:ts:master-data--WindTurbine:1.0.0". The properties on the schema are selected with guidance from AWS renewable energy expertise, see papers "JT Rand, et al., A continuously updated, geospatially rectified database of utility-scale wind turbines in the United States. Sci Data 7, 15 (2020)" and "J. Eriksson, Machine Learning for Predictive Maintenance on Wind Turbines, 2020"
# MAGIC 
# MAGIC ##### See below for extended properties. Schema can be edited to suit needs.
# MAGIC 
# MAGIC `{
# MAGIC   "kind": "osdu:ts:master-data--WindTurbine:1.0.0",
# MAGIC     "acl": acl,
# MAGIC     "legal": {
# MAGIC       "legaltags": [
# MAGIC         "osdu-public-usa-dataset"
# MAGIC       ],
# MAGIC       "otherRelevantDataCountries": [
# MAGIC         "US"
# MAGIC       ],
# MAGIC       "status": "compliant"
# MAGIC     },`
# MAGIC     `"data": {
# MAGIC       "location": {
# MAGIC         "latitude": "",
# MAGIC         "longitude": ""
# MAGIC       },
# MAGIC       "OEM": "",
# MAGIC       "ContractedMaintenanceProvider": "",
# MAGIC       "Model": "",
# MAGIC       "PitchRange": "",
# MAGIC       "YawRange": "",
# MAGIC       "TowerHeight": "",
# MAGIC       "Blade": {
# MAGIC         "Model": "",
# MAGIC         "Pitch": "",
# MAGIC         "Material": "",
# MAGIC       },`
# MAGIC       `"HubHeight": "",
# MAGIC       "TurbineCapacity": "",
# MAGIC       "RotorDiameter": "",
# MAGIC       "RotorSweptArea": "",
# MAGIC       "TotalHeight": "",
# MAGIC       "GearboxManufacturer": "",
# MAGIC       "GearboxBearingManufacturer": "",
# MAGIC       "LastMaintenanceDate": "",
# MAGIC       "InstallationDate": ""
# MAGIC     }
# MAGIC   }`
# MAGIC 
# MAGIC This data can be used in conjunction with a company's data to increase the number of features and details for more accurate predictive models. Additional examples of application of OSDU data can be found in the [OSDU™ Data Platform on AWS](https://aws.amazon.com/energy/osdu-data-platform/) and examples of the API, including uploading metadata and syntax at the [OSDU™ Quick Start Guide](https://community.opengroup.org/osdu/documentation/-/wikis/OSDU-API-Quick-start-guide#schema-creation) or the 

# COMMAND ----------

# MAGIC %md As this data is limited to OSDU members, this is left out of modeling for the reproducible notebook.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### About This Notebook
# MAGIC 
# MAGIC * This notebooks is intended to help you ingest sensor data to use with classification algorithms to predict equipment failure.
# MAGIC 
# MAGIC * In support of this goal, we will:
# MAGIC  * Load labeled sensor data that contains readings from healthy and damaged turbines
# MAGIC  * Create one pipeline for streaming and batch to predict failure in near real-time and/or on an ad-hoc basis. 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/ML-workflow.png" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Installations

# COMMAND ----------

# MAGIC %pip install dbldatagen

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation and ETL

# COMMAND ----------

# MAGIC %md Import libraries

# COMMAND ----------

import shutil
import numpy as np # linear algebra
import pandas as pd # data processing 
from pyspark.ml import *
# from pyspark.sql.functions import *
# from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, rand, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# COMMAND ----------

# MAGIC %md ##### Generate data damaged data into a pyspark dataframe

# COMMAND ----------

import dbldatagen as dg
import dbldatagen.distributions as dist

column_count = 10
data_rows = 250000
df_d_spec = (dg.DataGenerator(spark, name="turbine_damaged", rows=data_rows)
                            .withIdOutput()
                            .withColumn("AN3", DoubleType(), minValue=-10.7, maxValue=10.96, step=0.001, random=True, distribution=dist.Normal(-0.03,2.82))
                            .withColumn("AN4", DoubleType(), minValue=-11.64, maxValue=11.61, step=0.001, random=True, distribution=dist.Normal(-0.02,2.72))
                            .withColumn("AN5", DoubleType(), minValue=-16.46, maxValue=17.81, step=0.001, random=True, distribution=dist.Normal(-0.14,3.98))
                            .withColumn("AN6", DoubleType(), minValue=-20.31, maxValue=19.1, step=0.001, random=True, distribution=dist.Normal(-0.05,4.45))
                            .withColumn("AN7", DoubleType(), minValue=-17.83, maxValue=16.71, step=0.001, random=True, distribution=dist.Normal(-0.06,4.01))
                            .withColumn("AN8", DoubleType(), minValue=-36.83, maxValue=35.56, step=0.001, random=True, distribution=dist.Normal(-0.04,8.3))
                            .withColumn("AN9", DoubleType(), minValue=-29.99, maxValue=30.48, step=0.001, random=True, distribution=dist.Normal(-0.09,7.23))
                            .withColumn("AN10", DoubleType(), minValue=-16.96, maxValue=16.65, step=0.001, random=True, distribution=dist.Normal(-0.02,4.07))
                            .withColumn("ReadingType", StringType(), values=['DAMAGED'])
                            )
                            
turbine_damaged = df_d_spec.build()
# num_rows=df_d_spec.count()  

# COMMAND ----------

display(turbine_damaged)

# COMMAND ----------

import dbldatagen as dg
import dbldatagen.distributions as dist
from pyspark.sql.types import DoubleType, StringType
column_count = 10
data_rows = 24000000
df_h_spec = (dg.DataGenerator(spark, name="turbine_healthy", rows=data_rows)
                            .withIdOutput()
                            .withColumn("AN3", DoubleType(), minValue=-9.42, maxValue=9.51, step=0.001, random=True, distribution=dist.Normal(0,1.13))
                            .withColumn("AN4", DoubleType(), minValue=-10.37, maxValue=10.86, step=0.001, random=True, distribution=dist.Normal(0,1.91))
                            .withColumn("AN5", DoubleType(), minValue=-9.29, maxValue=10.15, step=0.001, random=True, distribution=dist.Normal(-0.09,1.72))
                            .withColumn("AN6", DoubleType(), minValue=-14.78, maxValue=14.88, step=0.001, random=True, distribution=dist.Normal(-0.24,2.81))
                            .withColumn("AN7", DoubleType(), minValue=-14.9, maxValue=15.28, step=0.001, random=True, distribution=dist.Normal(0.02,2.62))
                            .withColumn("AN8", DoubleType(), minValue=-21.93, maxValue=23.08, step=0.001, random=True, distribution=dist.Normal(-0.09,4.75))
                            .withColumn("AN9", DoubleType(), minValue=-8.08, maxValue=8.72, step=0.001, random=True, distribution=dist.Normal(-0.38,2.04))
                            .withColumn("AN10", DoubleType(), minValue=-9.87, maxValue=11.32, step=0.001,random=True, distribution=dist.Normal(0.01,1.91))
                            .withColumn("ReadingType", StringType(), values=['HEALTHY'])
                            )
                            
turbine_healthy = df_h_spec.build()
#num_rows=df_h_spec.count()  

# COMMAND ----------

display(turbine_healthy)

# COMMAND ----------

# Create some temp views for easy SQL queries
turbine_damaged.createOrReplaceTempView("turbine_damaged")
turbine_healthy.createOrReplaceTempView("turbine_healthy")

# COMMAND ----------

# Create a unified data set too with HEALTHY/DAMAGED label:
df = turbine_healthy.union(turbine_damaged)

# Randomly shuffle the data to ensure even distribution of healthy/damaged
df = df.orderBy(rand()).cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Create a database and table structure for the data.

# COMMAND ----------

# Set database name, file paths, and table names
database_name = 'wind_turbine'

# Set table paths the Delta tables
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
bronze_tbl_path = '/home/{}/wind_turbine/bronze/'.format(user)

# Set Delta table names
bronze_tbl_name = 'bronze_wt_data'

# Delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %md ##### Write the data to a Delta table

# COMMAND ----------

df.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("wind_turbine.bronze_wt_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC Select * from wind_turbine.bronze_wt_data

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC Select count(ReadingType) from wind_turbine.bronze_wt_data group by ReadingType

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like? Notice the much larger stdev in AN9.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Summary statistics by label

# COMMAND ----------

display(turbine_healthy.describe())

# COMMAND ----------

display(turbine_damaged.describe())

# COMMAND ----------

# Compare AN9 value for healthy/damaged; varies much more for damaged ones
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Q-Q Plot by Label
# MAGIC 
# MAGIC The Q-Q plot shows how a field of values are distributed.

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md ##### Box Plot by Label
# MAGIC 
# MAGIC The boxplot displays the expected range of values and shows the outliers.

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Creation: Workflows with Pyspark.ML Pipeline

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
stages = [VectorAssembler(inputCols=featureCols, outputCol="va"), StandardScaler(inputCol="va", outputCol="features"), StringIndexer(inputCol="ReadingType", outputCol="label")]
pipeline = Pipeline(stages=stages)

featurizer = pipeline.fit(df)
featurizedDf = featurizer.transform(df)

# COMMAND ----------

# MAGIC %md Review the featurized data, the far right columns are the label and features fields we will use for modeling.

# COMMAND ----------

display(featurizedDf)

# COMMAND ----------

# MAGIC %md Next we will split the data into testing and training sets

# COMMAND ----------

train, test = featurizedDf.select(["label", "features"]).randomSplit([0.8, 0.2])
train.cache()
test.cache()
print(train.count())
print(test.count())

# COMMAND ----------

# MAGIC %md Here we are using the [Spark MLlib library](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html) but this could be switched out to sklearn or your machine learning library of choice.

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)

grid = ParamGridBuilder().addGrid(
    gbt.maxDepth, [4, 5, 6]
).build()

ev = BinaryClassificationEvaluator()

# 3-fold cross validation
cv = CrossValidator(estimator=gbt, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=3)

cvModel = cv.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Model
# MAGIC 
# MAGIC We'll look at several ways to get several binary classifier evaluation metrics. First is AUROC (Area Under ROC curve), as shown below.

# COMMAND ----------

predictions = cvModel.transform(test)
# Prints AUROC
ev.evaluate(predictions)

# COMMAND ----------

bestModel = cvModel.bestModel
print(bestModel.toDebugString)

# COMMAND ----------

# MAGIC %md Next we will review the feature importance for this model and using sql check the accuracy of the predictions.

# COMMAND ----------

bestModel.featureImportances

# COMMAND ----------

# convert numpy.float64 to str for spark.createDataFrame()
weights = map(lambda w: '%.10f' % w, bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)).toDF("weight", "feature")

display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False))

# COMMAND ----------

predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql SELECT avg(CASE WHEN prediction = label THEN 1.0 ELSE 0.0 END) AS accuracy FROM predictions

# COMMAND ----------

# MAGIC %md After the gradient boost model, we will try a tree based model for classification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features")

grid = ParamGridBuilder().addGrid(
    rf.maxDepth, [4, 5, 6]
).build()


# we use the featurized dataframe here; we'll use the full pipeline when we save the model
# we use estimator=rf to access the trained model to view feature importance
# stages += [rf]
# p = Pipeline(stages = stages)

# 3-fold cross validation
cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=3)

cvModel = cv.fit(train)

# COMMAND ----------

predictions = cvModel.transform(test)
# Prints AUROC
ev.evaluate(predictions)

# COMMAND ----------

bestModel = cvModel.bestModel
print(bestModel.toDebugString)

# COMMAND ----------

bestModel.featureImportances

# COMMAND ----------

# convert numpy.float64 to str for spark.createDataFrame()
weights = map(lambda w: '%.10f' % w, bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)).toDF("weight", "feature")

display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False))

# COMMAND ----------

display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False))

# COMMAND ----------

predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql SELECT avg(CASE WHEN prediction = label THEN 1.0 ELSE 0.0 END) AS accuracy FROM predictions

# COMMAND ----------

stages += [rf]
stages

train.unpersist()
test.unpersist()
train, test = df.randomSplit([0.8, 0.2])
train.cache()
test.cache()

# COMMAND ----------

cv = CrossValidator(estimator=Pipeline(stages=stages), \
                    estimatorParamMaps=grid, \
                    evaluator=BinaryClassificationEvaluator(), \
                    numFolds=3)

cvModel = cv.fit(train)
cvModel.bestModel

# COMMAND ----------

# MAGIC %md In practice you'd save the model and load it later

# COMMAND ----------


# cvModel.bestModel.write().overwrite().save(...)
# ...
# model = PipelineModel.load(...)

# ... for now just use the already loaded model
model = cvModel.bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC In this Notebook, we focused on analyzing the sensor data to understand the indicators of a potential failure, and to build predictive maintenance predictive models.
# MAGIC 
# MAGIC In reality this is a just a portion of a predictive maintenance pipeline. For a end to end IOT wind turbine demo, including dashboarding, you can visit the Databricks demo site [dbdemos.ai](https://www.dbdemos.ai/demo.html?demoName=lakehouse-iot-platform)
