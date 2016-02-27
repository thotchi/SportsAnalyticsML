# Launch EC2 instance with Anaconda AMI
# AMI: anaconda3-2.4.1-on-ubuntu-14.04-lts - ami-39ff9759

# connect to EC2 instance
# ssh -i "anaconda_2.pem" ubuntu@ec2-52-53-223-200.us-west-1.compute.amazonaws.com
# ssh -i "anaconda_2.pem" ubuntu@ec2-54-67-17-194.us-west-1.compute.amazonaws.com
# ssh -i "anaconda_2.pem" ubuntu@ec2-54-153-74-94.us-west-1.compute.amazonaws.com

#/home/ubuntu/data_science/nba_ml.py

# sudo apt-get install python-qt4
# exec(open("./nba_ml.py").read())

# Import data analysis packages
import math
import numpy as np
import scipy
import matplotlib
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import pymongo
from sqlalchemy import create_engine
from datetime import date

# import sklearn datasets and algorithms
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sknn.mlp import Regressor, Layer, MultiLayerPerceptron

# import data from database into data frame
engine = create_engine('postgresql://postgres:postgres@dc-db.c0idyhwyzcjh.us-west-1.rds.amazonaws.com:5432/nba')

fd = open('/home/ubuntu/data_science/nba_boxscore_query.sql', 'r')
sqlFile = fd.read()
fd.close()

dataset = pd.read_sql(sqlFile, engine)

dataset_cols = DataFrame(dataset.columns.values)

# Load features and targets
features = dataset.iloc[:,7:42]
targets = dataset.iloc[:,6]

#-------------------------------------------------------------- 
# Normalize data
#--------------------------------------------------------------

# Load into Pandas
features_df = DataFrame(features)
features_shape = features_df.shape
features_stats = features_df.describe()

# Cleanse data

# Create normalized data set
features_columns = features_shape[1]
features_rows = features_shape[0]
features_normalized = DataFrame()

for col in range(0, features_columns-1):
	features_normalized[col] = (features_df.ix[:,col] - features_df.ix[:,col].mean()) / features_df.ix[:,col].std(ddof=0)

#-------------------------------------------------------------- 
# Loop through games of season
#--------------------------------------------------------------

# Use 100n-games to predict remaining games.
game_increment = 100
n = round(features_rows/game_increment + 0.5)


for x in range(0, n):
	#-------------------------------------------------------------- 
	# Create Test Train datasets
	#--------------------------------------------------------------

	# Split dataframe into training and test arrays
	features_train = features_normalized.ix[0:x*game_increment,].values
	targets_train = targets.ix[0:x*game_increment,].values
	
	features_test = features_normalized.ix[x*game_increment+1:features_rows,].values
	targets_test =  targets.ix[x*game_increment+1:features_rows,].values

	#-------------------------------------------------------------- 
	# Regression models
	#--------------------------------------------------------------

	# Linear regression
	linear_regr = linear_model.LinearRegression()
	linear_regr.fit(features_train, targets_train)
	linear_prediction = linear_regr.predict(features_test)
	# mean square error
	linear_regr_error = np.mean((linear_prediction-targets_test)**2)

	# comment out other models
	if False:
		# Ridge
		ridge_regr = linear_model.Ridge(alpha = 0.01)
		ridge_regr.fit(features_train, targets_train)
		ridge_prediction = ridge_regr.predict(features_test)
		# mean square error
		ridge_regr_error = np.mean((ridge_prediction-targets_test)**2)

		# Lasso
		lasso_regr = linear_model.Lasso(alpha = 0.01)
		lasso_regr.fit(features_train, targets_train)
		lasso_prediction = lasso_regr.predict(features_test)
		# mean square error
		lasso_regr_error = np.mean((lasso_prediction-targets_test)**2)

		# SGD Regressor
		sgd_regr = linear_model.SGDRegressor()
		sgd_regr.fit(features_train, targets_train)
		sgd_prediction = sgd_regr.predict(features_test)
		# mean square error
		sgd_regr_error = np.mean((sgd_prediction-targets_test)**2)

	#-------------------------------------------------------------- 
	# Write error to DataFrame
	#--------------------------------------------------------------

	table_val = pd.DataFrame({
		'training_date_start' : [dataset['date_game'][0].isoformat()], 
		'training_date_end' : [dataset['date_game'][x].isoformat()],
		'test_date_start' : [dataset['date_game'][x+1].isoformat()],
		'test_date_end' : [dataset['date_game'][features_rows-1].isoformat()],
		'training_sample_size' : [x*game_increment],
		'test_sample_size' : [features_rows-x*game_increment],
		'model_parameter_settings' : ['linear regression, 20 game moving avg, increasing training size'],
		'model' : ['linear regression'],
		'model_parameters' : ['N/A'],
		'mean_square_error' : [linear_regr_error],
		'root_mean_square_error' : [math.sqrt(linear_regr_error)],
		'date_tm' : [datetime.datetime.now().isoformat()]
	})

	table_val.to_sql('model_perf', engine, index = False, if_exists = 'append')
	
#-------------------------------------------------------------- 
# Use prior 1000 games to predict next 100 games using linear regression
#--------------------------------------------------------------

# Use 1000n-games to predict remaining games.
training_increment = 1000
test_increment = 100
n = round(features_rows/test_increment + 0.5)


for x in range(0, n):
	#-------------------------------------------------------------- 
	# Create Test Train datasets
	#--------------------------------------------------------------

	train_start = min(x * test_increment, features_rows - 2)
	train_end = min(x * test_increment + training_increment, features_rows-2)
	test_start = min(x * test_increment + training_increment + 1, features_rows-2)
	test_end = min(x * test_increment + training_increment + test_increment, features_rows-1)
	
	# Split dataframe into training and test arrays
	features_train = features_normalized.ix[train_start:train_end,].values
	targets_train = targets.ix[train_start:train_end,].values
	
	features_test = features_normalized.ix[test_start:test_end,].values
	targets_test =  targets.ix[test_start:test_end,].values

	#-------------------------------------------------------------- 
	# Regression models
	#--------------------------------------------------------------

	# Linear regression
	linear_regr = linear_model.LinearRegression()
	linear_regr.fit(features_train, targets_train)
	linear_prediction = linear_regr.predict(features_test)
	# mean square error
	linear_regr_error = np.mean((linear_prediction-targets_test)**2)

	# comment out other models
	if False:
		# Ridge
		ridge_regr = linear_model.Ridge(alpha = 0.01)
		ridge_regr.fit(features_train, targets_train)
		ridge_prediction = ridge_regr.predict(features_test)
		# mean square error
		ridge_regr_error = np.mean((ridge_prediction-targets_test)**2)

		# Lasso
		lasso_regr = linear_model.Lasso(alpha = 0.01)
		lasso_regr.fit(features_train, targets_train)
		lasso_prediction = lasso_regr.predict(features_test)
		# mean square error
		lasso_regr_error = np.mean((lasso_prediction-targets_test)**2)

		# SGD Regressor
		sgd_regr = linear_model.SGDRegressor()
		sgd_regr.fit(features_train, targets_train)
		sgd_prediction = sgd_regr.predict(features_test)
		# mean square error
		sgd_regr_error = np.mean((sgd_prediction-targets_test)**2)

	#-------------------------------------------------------------- 
	# Write error to DataFrame
	#--------------------------------------------------------------

	table_val = pd.DataFrame({
		'training_date_start' : [dataset['date_game'][train_start].isoformat()], 
		'training_date_end' : [dataset['date_game'][train_end].isoformat()],
		'test_date_start' : [dataset['date_game'][test_start].isoformat()],
		'test_date_end' : [dataset['date_game'][test_end].isoformat()],
		'training_sample_size' : [training_increment],
		'test_sample_size' : [test_increment],
		'model_parameter_settings' : ['linear regression, 20 game moving avg, 1000 game training and size'],
		'model' : ['linear regression'],
		'model_parameters' : ['N/A'],
		'mean_square_error' : [linear_regr_error],
		'root_mean_square_error' : [math.sqrt(linear_regr_error)],
		'date_tm' : [datetime.datetime.now().isoformat()]
	})

	table_val.to_sql('model_perf', engine, index = False, if_exists = 'append')

#-------------------------------------------------------------- 
# Use prior 1000 games to predict next 100 games using neural net
#--------------------------------------------------------------

# Use 1000-games to predict next 100 games.
training_increment = 1000
test_increment = 100
n = round(features_rows/test_increment + 0.5)


for x in range(0, n):
	#-------------------------------------------------------------- 
	# Create Test Train datasets
	#--------------------------------------------------------------

	train_start = min(x * test_increment, features_rows - 2)
	train_end = min(x * test_increment + training_increment, features_rows-2)
	test_start = min(x * test_increment + training_increment + 1, features_rows-2)
	test_end = min(x * test_increment + training_increment + test_increment, features_rows-1)
	
	# Split dataframe into training and test arrays
	features_train = features_normalized.ix[train_start:train_end,].values
	targets_train = targets.ix[train_start:train_end,].values
	
	features_test = features_normalized.ix[test_start:test_end,].values
	targets_test =  targets.ix[test_start:test_end,].values

	#-------------------------------------------------------------- 
	# Regression models
	#--------------------------------------------------------------

	# Neural Net regression
	
	nn_regr = Regressor(
	layers=[
		Layer("Rectifier", units=100),
		Layer("Linear")],
	learning_rate=0.02,
	n_iter=10)
	
	nn_regr.fit(features_train, targets_train)
	
	nn_prediction = nn_regr.predict(features_test)
	
	# mean square error
	nn_regr_error = np.mean((nn_prediction-targets_test)**2)


	#-------------------------------------------------------------- 
	# Write error to DataFrame
	#--------------------------------------------------------------

	table_val = pd.DataFrame({
		'training_date_start' : [dataset['date_game'][train_start].isoformat()], 
		'training_date_end' : [dataset['date_game'][train_end].isoformat()],
		'test_date_start' : [dataset['date_game'][test_start].isoformat()],
		'test_date_end' : [dataset['date_game'][test_end].isoformat()],
		'training_sample_size' : [training_increment],
		'test_sample_size' : [test_increment],
		'model_parameter_settings' : ['neural net regression, 20 game moving avg, 1000 game training and 100 game size'],
		'model' : ['neural net regression'],
		'model_parameters' : ['Rectifier, units=100; Linear; learning_rate=0.02; n_iter=10'],
		'mean_square_error' : [nn_regr_error],
		'root_mean_square_error' : [math.sqrt(nn_regr_error)],
		'date_tm' : [datetime.datetime.now().isoformat()]
	})

	table_val.to_sql('model_perf', engine, index = False, if_exists = 'append')