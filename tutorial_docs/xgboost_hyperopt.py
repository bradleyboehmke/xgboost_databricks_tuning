# Databricks notebook source
# MAGIC %md # Advanced XGBoost Hyperparameter Tuning
# MAGIC 
# MAGIC Gradient boosting machines (GBMs), and more specifically the XGBoost variant, are an extremely popular machine learning algorithm that have proven successful across many domains and is one of the leading algorithmic methods for tabular data. Whereas random forests build an ensemble of deep independent trees, GBMs build an ensemble of shallow trees in sequence with each tree learning and improving on the previous one. Although shallow trees by themselves are rather weak predictive models, they can be “boosted” to produce a powerful “committee” that, ___when appropriately tuned___, is often hard to beat with other algorithms.
# MAGIC 
# MAGIC The key phrase above to note is -- "_when appropriately tuned_." XGBoost has many hyperparameters to refine, which can be overwhelming. Moreover, refining these hyperparameters can take significant time; and the tracking of model performance based on hyperparameter values can get unwieldy if done inefficiently. Consequently, the objectives of this tutorial is to illustrate:
# MAGIC <br><br>
# MAGIC 1. Best practices for tuning XGBoost hyperparameters
# MAGIC 2. Leveraging Hyperopt for an effective and efficient XGBoost grid search
# MAGIC 3. Using MLflow for tracking and organizing grid search performance

# COMMAND ----------

# MAGIC %md ## Assumptions
# MAGIC 
# MAGIC This tutorial makes the assumption that the reader...
# MAGIC <br><br>
# MAGIC 1. Is comfortable using Python for basic data wrangling processes
# MAGIC 2. Is comfortable writing Python functions and understands context managers
# MAGIC 3. Understands the basics of the GBM and XGBoost algorithm
# MAGIC 
# MAGIC This tutorial also introduces the use of Hyperopt and MLflow for parts of the hyperparameter tuning process; however, this tutorial does not provide a comprehensive introduction to these packages. When and where appropriate this tutorial will hyperlink to additional resources that provide more comprehensive documentation.

# COMMAND ----------

# MAGIC %md ## Prerequisites
# MAGIC 
# MAGIC This tutorial leverages the Databricks Community Edition with an 8.3 ML runtime cluster. 
# MAGIC 
# MAGIC ### Packages
# MAGIC 
# MAGIC The following packages are leveraged in this tutorial with the emphasis on [xgboost](https://xgboost.readthedocs.io/en/latest/), [hyperopt](http://hyperopt.github.io/hyperopt/), and [mlflow](https://mlflow.org/); all of which are pre-installed on the 8.3 ML cluster.

# COMMAND ----------

# helper packages
import pandas as pd
import numpy as np
import time
import warnings

# modeling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# hyperparameter tuning
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from hyperopt.pyll import scope

# model/grid search tracking
import mlflow

# COMMAND ----------

# typically not advised but doing this to minimize excessive messaging 
# during the grid search
warnings.filterwarnings("ignore") 

# COMMAND ----------

# MAGIC %md ### Data
# MAGIC 
# MAGIC For simplicity we will use the well known [wine quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). Prior to modeling we will do a few simple data wrangling tasks. Our objective in this modeling task is to use wine characteristics in order to predict the ___quality___ of the wine. In this example we will convert this to a classification problem with the objective of predicting if a wine is high quality ( \\(quality \\geq 7\\) ) or low quality ( \\(quality < 7\\) ). Consquently, our data prep tasks include:
# MAGIC <br><br>
# MAGIC 1. Read in the data
# MAGIC 2. Create indicator column for red vs. white wine
# MAGIC 3. Combine the red and white wine data sets
# MAGIC 4. Clean up column names
# MAGIC 5. Convert wine quality column to a binary response variable

# COMMAND ----------

# 1. read in data
white_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

# 2. create indicator column for red vs. white wine
red_wine['is_red'] = 1
white_wine['is_red'] = 0

# 3. combine the red and white wine data sets
data = pd.concat([red_wine, white_wine], axis=0)
 
# 4. remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# 5. convert "quality" column to 0 vs. 1 to make this a classification problem
data["quality"] = (data["quality"] >= 7).astype(int)

# COMMAND ----------

data.head(10)

# COMMAND ----------

# MAGIC %md Next, we'll split our data into train and test sets using the default 75% (train) 25% (train) ratio.

# COMMAND ----------

# split data into train (75%) and test (25%) sets
train, test = train_test_split(data, random_state=123)
X_train = train.drop(columns="quality")
X_test = test.drop(columns="quality")
y_train = train["quality"]
y_test = test["quality"]

# COMMAND ----------

X_train.head()

# COMMAND ----------

y_test.head()

# COMMAND ----------

# MAGIC %md And finally we will convert our data to a DMatrix object which is an XGBoost internal data structure that is optimized for both memory efficiency and training speed.

# COMMAND ----------

train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)

# COMMAND ----------

# MAGIC %md ## XGBoost Grid Search
# MAGIC 
# MAGIC As previously mentioned XGBoost has many hyperparameters. These hyperparameters can be grouped into 4 categories. The following provides a quick overview of the purpose of the primary hyperparameters along with general guidelines regarding the range of values to typically search across.
# MAGIC 
# MAGIC __Boosting hyperparameters__ include those parameters that are central to the gradient boosting algorithm. This primarily focuses on the [gradient descent process](https://bradleyboehmke.github.io/HOML/gbm.html#gbm-gradient). The two main parameters include learning rate (discussed below) and the number of trees. However, the number of trees is, technically, not a hyperparameter. We primarily want to ensure we are training for the right number of trees to minimize the loss function while not overfitting. XGBoost allows for some early stopping procedures (discussed later) that allows us to train for a large number of trees but then stop training once we no longer improve on the loss. Consequently, the primary boosting hyperparamter of concern is learning rate.
# MAGIC 
# MAGIC * __learning_rate__: Determines the contribution of each tree on the final outcome and controls how quickly the algorithm learns. Values range from 0–1 with typical values between 0.001–0.3. Smaller values make the model robust to the specific characteristics of each individual tree, thus allowing it to generalize well. Smaller values also make it easier to stop prior to overfitting; however, they increase the risk of not reaching the optimum with a fixed number of trees and are more computationally demanding. __Recommendation__: Search across values ranging from 0.0001-1 on a log scale (i.e. 0.0001, 0.001, 0.01, 0.1, 1).
# MAGIC 
# MAGIC __Tree hyperparameters__ - although we can perform boosting on non-tree based algorithms, using decision trees as the base learning for boosting is the most common procedure. Consequently, there are several hyperparamters that steer how the decision trees are built. The most common tree hyperparameters to tune include:
# MAGIC 
# MAGIC * __max_depth__: Controls the depth of the individual trees. Typical values range from a depth of 3–8 but it is not uncommon to see a tree depth of 1 ([J. Friedman, Hastie, and Tibshirani 2001](https://web.stanford.edu/~hastie/ElemStatLearn/)). Smaller depth trees such as decision stumps are computationally efficient (but require more trees); however, higher depth trees allow the algorithm to capture unique interactions but also increase the risk of over-fitting. Note that larger (both by number of features and observations) training data sets are more tolerable to deeper trees. __Recommendation__: Uniformly search across values ranging from 1-10 but be willing to increase the high value range for larger datasets.
# MAGIC 
# MAGIC * __min_child_weight__: Also, controls the complexity of each tree by requiring the minimum number of instances (measured by [hessian](https://stats.stackexchange.com/questions/317073/explanation-of-min-child-weight-in-xgboost-algorithm) within XGBoost) to be greater than a certain value for further partitioning to occur. Since we tend to use shorter trees this rarely has a large impact on performance but tuning it should not be overlooked. Typical values range from 5–15 where higher values helping to prevent a model from learning relationships which might be highly specific to the particular sample selected for a tree (overfitting) but smaller values can help with imbalanced target classes in classification problems. __Recommendation__: Uniformly search across values ranging from near zero-20 but be willing to increase the high value range for larger datasets.
# MAGIC 
# MAGIC __Stochastic hyperparameters__ - Stochastic gradient boosting is the act of randomly subsampling the training data set, whether that be by sampling rows or columns. Subsampling helps to reduce tree correlation and also helps reduce the chances of getting stuck in local minimas, plateaus, and other irregular terrain of the loss function so that we may find a near global optimum loss. There are four main ways we subsample while implementing XGBoost:
# MAGIC 
# MAGIC * __subsample__: Subsampling rows before creating each tree. This form of subsampling primarily helps to reduce tree correlation. Generally, aggressive subsampling of rows, such as selecting only 50% or less of the training data, has shown to be beneficial and typical values range between 0.5–0.8. __Recommendation__: Uniformly search across values ranging from 0.5-1.0.
# MAGIC 
# MAGIC * __colsample_bytree__: Subsampling of columns and the impact to performance largely depends on the nature of the data and if there is strong multicollinearity or a lot of noisy features. If there are fewer relevant predictors (more noisy data) higher values of column subsampling tends to perform better because it makes it more likely to select those features with the strongest signal. When there are many relevant predictors, a lower values of column subsampling tends to perform well. __Recommendation__: Uniformly search across values ranging from 0.5-1.0.
# MAGIC 
# MAGIC * __colsample_bylevel__ & __colsample_bynode__: These additional subsampling procedures act similar to `colsample_bytree`; however, they do additional subsampling of columns every time we achieve a new level of depth in a tree (`colsample_bylevel`) and/or every node split within a given tree depth level (`colsample_bynode`). These hyperparameters are less common but can be useful when dealing with extremely large data sets or with data sets that have very high multicollinearity. __Recommendation__: Uniformly search across values ranging from 0.5-1.0.
# MAGIC 
# MAGIC __Regularization hyperparameters__ - XGBoost provides multiple regularization parameters to help reduce model complexity and guard against overfitting. These include:
# MAGIC 
# MAGIC * __gamma__: A pseudo-regularization hyperparameter known as a Lagrangian multiplier and controls the complexity of a given tree. `gamma` specifies a minimum loss reduction required to make a further partition on a leaf node of the tree. When gamma is specified, XGBoost will grow the tree to the max depth specified but then prune the tree to find and remove splits that do not meet the specified gamma. gamma tends to be worth exploring as your trees in your GBM become deeper and when you see a significant difference between the train and validation/test error. The value of gamma typically ranges from \\(0 - ∞ \\)
# MAGIC   (0 means no constraint while large numbers mean a higher regularization). What quantifies as a large gamma value is dependent on the loss function but generally lower values between 1–20 will do if gamma is influential. __Recommendation__: Search across values ranging from 0-some large number on a log scale (i.e. 0, 1, 10, 100, 1000, etc.).
# MAGIC 
# MAGIC * __alpha__: Provides an \\( L_{2} \\) regularization to the loss function, which is similar to the [Ridge penalty](https://bradleyboehmke.github.io/HOML/regularized-regression.html#ridge) commonly used for regularized regression. Typically values range from \\(0 - ∞ \\) with larger values causing more conservative models. Setting both `alpha` and `lambda` to greater than 0 results in an [elastic net regularization](https://bradleyboehmke.github.io/HOML/regularized-regression.html#elastic). **Recommendation**: Search across values ranging from 0-some large number on a log scale (i.e. 0, 1, 10, 100, 1000, etc.).
# MAGIC 
# MAGIC * __lambda__: Provides an \\(L_{1} \\) regularization to the loss function, which is similar to the [Lasso penalty](https://bradleyboehmke.github.io/HOML/regularized-regression.html#lasso) commonly used for regularized regression. Typically values range from \\(0 - ∞ \\) with larger values causing more conservative models. Setting both `alpha` and `lambda` to greater than 0 results in an [elastic net regularization](https://bradleyboehmke.github.io/HOML/regularized-regression.html#elastic). **Recommendation**: Search across values ranging from 0-some large number on a log scale (i.e. 0, 1, 10, 100, 1000, etc.).

# COMMAND ----------

# MAGIC %md ### Hyperopt
# MAGIC 
# MAGIC There are [several approaches](https://en.wikipedia.org/wiki/Hyperparameter_optimization) you can use for performing a hyperparameter grid search -- full cartesian grid search, random grid search, Bayesian optimization, etc. However, when performing a grid search over a large search space with many hyperparameters and value ranges, using a full cartesian search or a random unguided search will be very time costly or lead to non-optimal results. To perform a grid search of this scale we prefer to us a ___Bayesian optimizer___, meaning it is not merely randomly searching or searching a grid, but intelligently learning which combinations of values work well as it goes, and focusing the search there.  
# MAGIC 
# MAGIC Multiple packages exist that provide Bayesian optimization grid search capabilities; however, [Hyperopt](https://github.com/hyperopt/hyperopt) is one of the most popular packages and is also tightly integrated with Databricks and MLFlow. The benefits of using Hyperopt include:
# MAGIC 
# MAGIC * Open source
# MAGIC * Bayesian optimizer – smart searches over hyperparameters (using a [Tree of Parzen Estimators](https://optunity.readthedocs.io/en/latest/user/solvers/TPE.html)), not grid or random search
# MAGIC * Integrates with Apache Spark for parallel hyperparameter search
# MAGIC * Integrates with MLflow for automatic tracking of the search results
# MAGIC * Included already in the Databricks ML runtime
# MAGIC * Maximally flexible: can optimize literally any Python model with any hyperparameters

# COMMAND ----------

# MAGIC %md ### Hyperparameter search space
# MAGIC 
# MAGIC Our first objective is to define our hyperparameter search space. Hyperopt provides a [variety of expressions](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/) to create a range of values (i.e. `hp.loguniform`, `hp.uniform`). When establishing the range of values choose bounds that are extreme and let Hyperopt learn what values aren’t working well. For example, if a regularization parameter is typically between 1 and 10, try values from 0 to 100. The range should include the default value, certainly. At worst, it may spend time trying extreme values that do not work well at all, but it should learn and stop wasting trials on bad values. This may mean subsequently re-running the search with a narrowed range after an initial exploration to better explore reasonable values.
# MAGIC 
# MAGIC A best practice strategy for a Hyperopt workflow is as follows:
# MAGIC 
# MAGIC 1. Choose what hyperparameters are reasonable to optimize
# MAGIC 2. Define broad ranges for each of the hyperparameters (including the default where applicable)
# MAGIC 3. Run a small number of trials
# MAGIC 4. Observe the results in an MLflow parallel coordinate plot and select the runs with lowest loss
# MAGIC 5. Move the range towards those higher/lower values when the best runs’ hyperparameter values are pushed against one end of a range
# MAGIC 6. Determine whether certain hyperparameter values cause fitting to take a long time (and avoid those values)
# MAGIC 7. Re-run with more trials
# MAGIC 8. Repeat until the best runs are comfortably within the given search bounds and none are taking excessive time
# MAGIC 
# MAGIC **NOTE**: when using `hp.loguniform` the values you specify for min and max should be expressed in `exp(x)`. For example, if you want the `learning_rate` to range from a min of 0.0001 and max of 1 on a log scale you would specify `hp.loguniform(-9, 0)` because `exp(-9) = 0.0001` and `exp(0) = 1`.
# MAGIC 
# MAGIC | Hyperparameter     | Suggested Hyperopt Expression | Value Ranges |
# MAGIC | ------------------ | --------------- | ------------------------------------ |
# MAGIC | `learning_rate`    | `hp.loguniform` | min = -9; max = 0 (because `exp(-9) = 0.0001` and `exp(0) = 1`) |
# MAGIC | `max_depth`        | `hp.uniform`    | min = 1; max = 100 (we wrap with `scope.int` since this must be an integer) |
# MAGIC | `min_child_weight` | `hp.loguniform` | min = -2; max = 3 (includes default value of 1) |
# MAGIC | `subsample`        | `hp.uniform`    | min = 0.5; max = 1 |
# MAGIC | `colsample_bytree` | `hp.uniform`    | min = 0.5; max = 1 |
# MAGIC | `gamma`            | `hp.loguniform` | min = -10; max = 10 (because `exp(-10) ~ 0` and `exp(10) > 10000`) |
# MAGIC | `alpha`            | `hp.loguniform` | min = -10; max = 10 (because `exp(-10) ~ 0` and `exp(10) > 10000`) |
# MAGIC | `lambda`           | `hp.loguniform` | min = -10; max = 10 (because `exp(-10) ~ 0` and `exp(10) > 10000`) |
# MAGIC   

# COMMAND ----------

search_space = {
  'learning_rate': hp.loguniform('learning_rate', -7, 0),
  'max_depth': scope.int(hp.uniform('max_depth', 1, 100)),
  'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
  'subsample': hp.uniform('subsample', 0.5, 1),
  'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
  'gamma': hp.loguniform('gamma', -10, 10),
  'alpha': hp.loguniform('alpha', -10, 10),
  'lambda': hp.loguniform('lambda', -10, 10),
  'objective': 'binary:logistic',
  'eval_metric': 'auc',
  'seed': 123,
}

# COMMAND ----------

search_space

# COMMAND ----------

# MAGIC %md ### Defining search process
# MAGIC 
# MAGIC When using Hyperopt, we need to create an objective function that will be optimized by Hyperopt's Bayesian optimizer. The function definition typically only requires one parameter (`params`), which hyperopt will use to pass a set of hyperparameter values. So given a set of hyperparameter values that Hyperopt chooses, the function trains our given model and computes the loss for the model built with those hyperparameters. Consequently, we need our function to return a dictionary with at least two items:
# MAGIC 
# MAGIC * status: One of the keys from `hyperopt.STATUS_STRINGS` to signal successful vs. failed completion. Most common is `STATUS_OK`.
# MAGIC * loss: The loss score to be optimized. Note that hyperopt will always try to minimize the loss so if you choose a loss where the objective is to maximize (i.e. \\( R^2 \\), AUC) then you will need to negate this value.
# MAGIC 
# MAGIC A big benefit of using hyperopt is that it is automatically integrated with MLFlow. Consequently, we can leverage the many benefits of MLFlow within this process. For example, we can use:
# MAGIC 
# MAGIC * [`mlflow.xgboost.autolog`](https://www.mlflow.org/docs/latest/python_api/mlflow.xgboost.html#mlflow.xgboost.autolog) to automatically log the parameters that hyperopt chooses
# MAGIC * [`mlflow.start_run`](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run) to start a new MLFlow run to log to. This allows us to run different variations of hyperparameters runs and keep our logged values organized.
# MAGIC * [`mlflow.log_metric`](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric) to record specific metrics of interest for each model run.
# MAGIC 
# MAGIC **Tip**: Since our learning rate hyperparameter can very from very small to quite large, it is always recommended to crank up the number of boosted trees (`num_boost_round`) and use `early stopping_rounds` to end training once the loss has stopped improving for a given model run.  Also, some hyperparameters have a large impact on runtime. A very small learning rate and large max tree depth can cause it to fit models that are large and expensive to train. Worse, sometimes models take a long time to train because they are overfitting the data! Hyperopt does not try to learn about runtime of trials or factor that into its choice of hyperparameters. Consequently, it can be beneficial to time the run time for each model instance so you can make a tradeoff between performance and compute time.

# COMMAND ----------

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog(silent=True)
  
  # However, we can log additional information by using an MLFlow tracking context manager 
  with mlflow.start_run(nested=True):

    # Train model and record run time
    start_time = time.time()
    booster = xgb.train(params=params, dtrain=train, num_boost_round=5000, evals=[(test, "test")], early_stopping_rounds=50, verbose_eval=False)
    run_time = time.time() - start_time
    mlflow.log_metric('runtime', run_time)
    
    # Record AUC as primary loss for Hyperopt to minimize
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -auc_score, 'booster': booster.attributes()}

# COMMAND ----------

# MAGIC %md ### Executing search
# MAGIC 
# MAGIC To execute the search we use `fmin` and supply it our model training (objective) function along with the hyperparameter search space. `fmin` can use different algorithms to search across the hyperparameter search space (i.e. random, Bayesian); however, we suggest using the Tree of Parsen Estimators (`tpe.suggest`) which will perform a smart Bayesian optimization grid search.  
# MAGIC 
# MAGIC Also, as previously mentioned, its is smart to run an initial small number of trials to find a range of hyperparameter values that appear to perform well and then refine the search space. The following example runs 25 trials for the initial search (note that 'small' will be relative to compute time and size of the search space).
# MAGIC 
# MAGIC **Note**: Hyperopt can parallelize its trials across a Spark cluster, which is a great feature. Building and evaluating a model for each set of hyperparameters is inherently parallelizable, as each trial is independent of the others. Using Spark to execute trials is simply a matter of using “SparkTrials” in Hyperopt. This is a great idea in environments like Databricks where a Spark cluster is readily available. However, considering this notebook is using a single node cluster there are no additional benefits in trying to parallelize.

# COMMAND ----------

#spark_trials = SparkTrials(parallelism=4)

# COMMAND ----------

with mlflow.start_run(run_name='initial_search'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=25,
    rstate=np.random.RandomState(123),
    #trials=spark_trials
  )

# COMMAND ----------

best_params

# COMMAND ----------

# MAGIC %md ### Assess results
# MAGIC 
# MAGIC As shown above, `best_params` will contain the hyperparameter values that results in the highest AUC. However, we can also click on the "experiment in MLflow" link which will take us to MLFlow experiment board where additional details can be found. 
# MAGIC 
# MAGIC For example, we can compare all the model runs and hyperparameter values with a parallel coordinates plot to assess which hyperparameter values tend to be most related high AUC scores. In the below plot we can see that those models with higher AUCs tend to have:
# MAGIC 
# MAGIC * higher `max_depth` values
# MAGIC * lower `min_child_weight`, `learning_rate`, `lambda`, `gamma`, and `alpha` values
# MAGIC * and it does not appear that `subsample` and `colsample_bytree` have much effect
# MAGIC 
# MAGIC ![hyperparameter_results](https://user-images.githubusercontent.com/6753598/132388405-fb243df5-4a6d-43c8-a33a-f8e82ffc1f3a.png)
# MAGIC 
# MAGIC Also, we can look at a scatter plot of AUC vs. runtime and we see that some models tend to to perform quite well but with substantially less runtime demands than other high performing models. The objective is to use the findings from these plots to identify refined values for our search space and then perform a second iteraction of executing a search to try and find a more optimal model. However, for brevity we will leave this as an exercise for our reader.
# MAGIC 
# MAGIC ![runtime_performance](https://user-images.githubusercontent.com/6753598/132387421-875873d1-f415-46bb-ac30-7203f7e93aef.png)

# COMMAND ----------

# MAGIC %md ### Alternative early stopping procedures
# MAGIC 
# MAGIC With GBMs (and a few other resource demanding algorithms such as deep learning), compute time can increase significantly as the dimensions of the data and/or hyperparameter grid search expand. Hyperopt provides a few alternative methods for early stopping procedures. In addition of using `max_evals` for limiting how many models to run, we can also stop our hyperparameter search using `timeout` (stop based on time) and `loss_threshold` (stop once we've met a certain loss value), or a combination of any of the above.

# COMMAND ----------

with mlflow.start_run(run_name='xgb_timeout'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    timeout=60*5, # stop the grid search after 5 * 60 seconds == 5 minutes
    #trials=spark_trials, 
    rstate=np.random.RandomState(123)
  )

# COMMAND ----------

with mlflow.start_run(run_name='xgb_loss_threshold'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    loss_threshold=-0.92, # stop the grid search once we've reached an AUC of 0.92 or higher
    timeout=60*10,        # stop after 5 minutes regardless if we reach an AUC of 0.92
    #trials=spark_trials, 
    rstate=np.random.RandomState(123)
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Summary
# MAGIC 
# MAGIC Tuning an XGBoost algorithm is no small feat. This tutorial outlined the primary hyperparameters that tend to impact model performance along with recommended values to explore for each hyperparameter. In addition, this tutorial illustrated how to use Hyperopt for an intelligent, Bayesian optimization approach to explore the search space along with MLFlow to log and organize the hyperparameter exploration within Databricks.
# MAGIC 
# MAGIC However, this tutorial should only serve as an introduction to these tools and processes. To go further and learn more please explore the following additional resources.
# MAGIC 
# MAGIC 
# MAGIC ## Additional resources
# MAGIC 
# MAGIC * XGBoost
# MAGIC    - [Getting started with XGBoost on Databricks](https://docs.databricks.com/applications/machine-learning/train-model/xgboost.html) (Databricks docs)
# MAGIC    - [Additional notes on XGBoost hyperparameter tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html) (package docs)
# MAGIC    - [A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/) (blog)
# MAGIC    
# MAGIC * Hyperopt
# MAGIC    - [Hyperopt best practices documentation from Databricks](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices.html?_gl=1*wlclzc*_gcl_aw*R0NMLjE2Mjg3OTM0NTQuQ2p3S0NBandqZE9JQmhBX0Vpd0FIejh4bTF0d0hKSFdvYzgzUDBHTWd5Z3duMFZMb1dueUpDYVU0aDZSOHdDVE91UU05VWZ4QTdEaUxSb0NPN1FRQXZEX0J3RQ..&_ga=2.244655938.636196059.1630927467-1465050559.1628793454&_gac=1.16404164.1628793456.CjwKCAjwjdOIBhA_EiwAHz8xm1twHJHWoc83P0GMgygwn0VLoWnyJCaU4h6R8wCTOuQM9UfxA7DiLRoCO7QQAvD_BwE) (Databricks docs)
# MAGIC    - [How (Not) to Tune Your Model With Hyperopt](https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html) (blog)
# MAGIC    - [Scaling Hyperopt to Tune Machine Learning Models in Python](https://databricks.com/blog/2019/10/29/scaling-hyperopt-to-tune-machine-learning-models-in-python.html) (blog)
# MAGIC * MLFlow
# MAGIC    - [Python MLFlow API documentation](https://www.mlflow.org/docs/latest/python_api/index.html) (package docs)
# MAGIC    - [MLFlow guide](https://docs.databricks.com/applications/mlflow/index.html) (Databricks docs)
# MAGIC    - [Best Practices for Hyperparameter Tuning with MLflow](https://databricks.com/session/best-practices-for-hyperparameter-tuning-with-mlflow) (talk)

# COMMAND ----------


