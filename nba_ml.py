# Launch EC2 instance with Anaconda AMI
# AMI: anaconda3-2.4.1-on-ubuntu-14.04-lts - ami-39ff9759

# connect to EC2 instance
# ssh -i "anaconda_2.pem" ubuntu@ec2-52-53-223-200.us-west-1.compute.amazonaws.com
# sudo apt-get install python-qt4

# Import data analysis packages
import numpy as np
import scipy
import matplotlib
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import pymongo

# import sklearn datasets and algorithms
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

def getDatabaseData(hostname,port,collection):
    # Setup connection to DB
    client = pymongo.MongoClient(hostname,port)
    db = client[collection]

    # Initialize List
    dataList = []

    # Pull data from MongoDB database
    boxScoreDict = db.boxScores.find()

    # Load database data as row of values for one game
    for values in boxScoreDict:
        GAME_ID = int(values.get('GAME_ID'))
        TEAM_ID = int(values.get('TEAM_ID'))
        FGM = int(values.get('FGM'))
        FGA = int(values.get('FGA'))
        FG_PCT = int(values.get('FG_PCT'))
        FG3M = int(values.get('FG3M'))
        FG3A = int(values.get('FG3A'))
        FG3_PCT = int(values.get('FG3_PCT'))
        FTM = int(values.get('FTM'))
        FTA = int(values.get('FTA'))
        FT_PCT = int(values.get('FT_PCT'))
        OREB = int(values.get('OREB'))
        DREB = int(values.get('DREB'))
        REB = int(values.get('REB'))
        AST = int(values.get('AST'))
        STL = int(values.get('STL'))
        BLK = int(values.get('BLK'))
        TO = int(values.get('TO'))
        PF = int(values.get('PF'))
        PTS = int(values.get('PTS'))
        PLUS_MINUS = int(values.get('PLUS_MINUS'))
        dataList.append([GAME_ID,TEAM_ID,FGM,FGA,FG_PCT,FG3M,FG3A,FG3_PCT,
			FTM,FTA,FT_PCT,OREB,DREB,REB,AST,STL,BLK,TO,PF,PTS,
			PLUS_MINUS])
    return dataList

#Load dataset from database
dataset = getDatabaseData('localhost',27017,'nba') 
print(dataset)

######Take Out exit()
exit()

# Load dataset
#dataset = datasets.load_diabetes()
# Load features and targets
features = dataset.data
targets = dataset.target

np.unique(targets)

#-------------------------------------------------------------- 
# Normalize data
#--------------------------------------------------------------

# Load into Pandas
features_df = DataFrame(features)
features_stats = features_df.describe()

# Cleanse data

#-------------------------------------------------------------- 
# Create Test Train datasets
#--------------------------------------------------------------

# Split data into training and test datasets
# A random permutation to split the data randomly
np.random.seed()
indices = np.random.permutation(len(features))

dataset_size = len(features)
test_pct = 0.9
test_size = np.int(round(dataset_size * test_pct,0))

features_train = features[indices[:-test_size]]
targets_train = targets[indices[:-test_size]]
features_test = features[indices[-test_size:]]
targets_test = targets[indices[-test_size:]]

#-------------------------------------------------------------- 
# Fit models
#--------------------------------------------------------------

# Linear regression
linear_regr = linear_model.LinearRegression()
linear_regr.fit(features_train, targets_train)
print(linear_regr.coef_)

# mean square error
np.mean((linear_regr.predict(features_test)-targets_test)**2)

# score
linear_regr.score(features_test, targets_test)

# Plot Prediction vs. Actual
