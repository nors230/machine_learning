'''
CAPP30254-ML: HW 2 - Machine Learning Pipeline

Code for Machine Learning Pipeline

Nora Hajjar
'''
import numpy as np
import pandas as pd 
import util
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

####READ/LOAD_DATA####
def load_data(filename):
	'''
	Load data from a csv to pandas df
	'''
	df = pd.read_csv(filename)
	return df


####EXPLORE_DATA####
def get_distribution(df):
	'''
	break this into chunks
	'''
	pass


def get_outliers(df):
	'''
	'''
	pass


def get_summary_statistics(df):
	'''
	model this after:
	https://mit.cs.uchicago.edu/capp30121-aut-18/nhajjar/blob/pa6-grading/pa6/traffic_stops.py

	'''
	pass



####PRE-PROCESS/CLEAN_DATA####
def get_nulls(df):
	'''
	'''
	return df.isnull().sum()


def fill_nas(df, col_name):
	'''
	'''
	

	return df
	






####GENERATE-FEATURES/PREDICTORS####
def convert_var_cont_to_disc(df, var_name):
	'''
	Convert one variable at a time from continuous to discreet 
	Returns: updated df w/ converted column
	'''
	pass

	return df


def convert_var_cat_to_bin(df, var_name):
	'''
	Convert one variable at a time form categorical to binary 
	Returns: updated df with converted column
	'''
	pass

	return df


def split_data(df, pred_vars, dep_var):
	'''
	Choose predictors and dependent variables, create train/test data splits
	'''
	X = df[pred_vars]
	y = df[dep_var]
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
	return X_train, X_test, y_train, y_test
 

####BUILD-CLASSIFIER####
def build_classifier(X_train, X_test, y_train, y_test):
	'''
	Build classifier, return predictions and actual results
	'''
	treereg = DecisionTreeRegressor(random_state=1)
	treereg.fit(X_train, y_train)
	preds = treereg.predict(X_test)
	return y_test, preds


def calc_rmse(y_test, preds):
	'''
	Calculate Root Mean Squared Error
	'''
	return np.sqrt(metrics.mean_squared_error(y_test, preds))


def evaluate_classifier():
	'''
	'''
	pass


#minimum functions to include:
#1) Read/Load Data - DONE
#2) Explore Data
#3) Pre-Process and Clean Data
#4) Generate Features/Predictors - DONE
#5) Build Machine Learning Classifier
#6) Evaluate Classifier










