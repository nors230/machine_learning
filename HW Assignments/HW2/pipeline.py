'''
CAPP30254-ML: HW 2 - Machine Learning Pipeline

Code for Machine Learning Pipeline

Nora Hajjar
'''
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


####READ/LOAD_DATA####  DONE
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


####PRE-PROCESS/CLEAN_DATA####  DONE
def check_null_counts(df):
	'''
	Input df, find nulls
	Return list of null_cols
	'''
	null_counts = df.isnull().sum()
	return null_counts


def get_null_cols(df, null_counts):
	'''
	Input df, find nulls

	Return list of null_cols
	'''
	null_cols = []
	for index, val in null_counts.iteritems():
		if val > 0:
			null_cols.append(index)
	return null_cols


def fill_null_cols(df, null_cols):
	'''
	Input df and list of null cols
	Calc mean for the column, fill NAs with the mean
	Return: updated df 
	'''
	for col in null_cols:
		df[col].fillna(df[col].mean(), inplace=True)


####GENERATE-FEATURES/PREDICTORS####
def convert_var_cont_to_disc(df, new_col, var_name, bins, labels):
	'''
	Convert one variable at a time from continuous to discreet. 
	We are binning/bucketing continuous data into discreet chunks 
	to use as ordinal categorical variables
	use pd.cut

	Returns: none, creates a new column 
	'''
	df[new_col] = pd.cut(df[var_name], bins=bins, labels=labels)


def convert_var_cat_to_bin(df, new_col, var_name, var0, var1):
	'''
	Convert one variable at a time form categorical to binary 
	Returns: none, creates a new column in the df
	'''
	df[new_col] = df[var_name].map({var0:0, var1:1})


def split_data(df, pred_vars, dep_var, test_size=0.25, random_state=1):
	'''
	Choose predictors and dependent variables, create train/test data splits
	'''
	X = df[pred_vars]
	y = df[dep_var]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
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


#################################################################
#MAP #minimum functions to include:
#1) Read/Load Data - DONE

#2) Explore Data - DO THIS LAST -
	#distributions of variables
	#correlations of variables
	#find outliers
	#data summaries

#3) Pre-Process and Clean Data - DONE
	#fill in missing values, use mean to fill in

#4) Generate Features/Predictors 
	#write function for continuous --> discreet variable - DONE
	#write function for categorical --> binary variable - DONE
	#apply each function to at least one variable each in the current data set

#5) Build Machine Learning Classifier
	#select classifier - decision trees

#6) Evaluate Classifier
	#use any metric
	#also use accuracy
	#can evaluate on same data (but bad form)






###################################################

#OUTSTANDING TO-DOS:
#2) Explore Data - DO THIS LAST -
	#distributions of variables
	#correlations of variables
	#find outliers
	#data summaries

#4) Generate Features/Predictors 
	#apply each function to at least one variable each in the current data set

#5) Build Machine Learning Classifier
	#select classifier - decision trees

#6) Evaluate Classifier
	#use any metric
	#also use accuracy
	#can evaluate on same data (but bad form)








