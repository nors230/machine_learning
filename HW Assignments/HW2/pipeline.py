'''
CAPP30254-ML: HW 2 - Machine Learning Pipeline

Code for Machine Learning Pipeline Version 1

Nora Hajjar
'''
import numpy as np
import pandas as pd 
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import cross_val_score


####1)READ/LOAD_DATA####
def load_data(filename):  
	'''
	Load data from a csv to pandas df

	Inputs: filename

	Returns: pandas df
	'''
	df = pd.read_csv(filename)
	return df


####2)EXPLORE_DATA####
def get_sum_stats(df):
	'''
	Generate summary statistics of dataframe

	Inputs: df

	Returns: summary statistics
	'''
	return df.describe()


def get_hist(df):
	'''
	Generate histogram of datafame

	Inputs: df

	Returns: histogram image of data
	'''
	return df.hist()


def get_outliers(df):
	'''
	Generate boxplot to see outliers in the data

	Inputs: df

	Returns: boxplot
	'''
	return df.boxplot()


####3)PRE-PROCESS/CLEAN_DATA####  DONE
def check_null_counts(df):
	'''
	Get null counts in dataframe

	Inputs: df

	Returns: null counts
	'''
	null_counts = df.isnull().sum()
	return null_counts


def get_null_cols(df, null_counts):
	'''
	Get null columns in dataframe

	Inputs: df

	Returns: list of null columns
	'''
	null_cols = []
	for index, val in null_counts.iteritems():
		if val > 0:
			null_cols.append(index)
	return null_cols


def fill_null_cols(df, null_cols):
	'''
	Fill null columns in the dataframe with the mean of the column

	Inputs: df, list of null columns

	Returns: none, updates the dataframe in place 
	'''
	for col in null_cols:
		df[col].fillna(df[col].mean(), inplace=True)


####4)GENERATE-FEATURES/PREDICTORS####
def convert_var_cont_to_disc(df, new_col, var_name, bins, labels=None):
	'''
	Convert one variable at a time from continuous to discreet. 
	We are binning/bucketing continuous data into discreet chunks 
	to use as ordinal categorical variables using pd.cut

	Inputs: df, new col name, current col name, bin number, labels
	
	Returns: none, creates a new column in the df
	'''
	df[new_col] = pd.cut(df[var_name], bins=bins, labels=labels)


def convert_var_cat_to_bin(df, new_col, var_name, var0, var1):
	'''
	Convert one variable at a time from categorical to binary

	Inputs: df, new col name, current col name, variable 0 name, variable 1 name

	Returns: none, creates a new column in the df
	'''
	df[new_col] = df[var_name].map({var0:0, var1:1})


def split_data(df, pred_vars, dep_var):
	'''
	Choose predictors and dependent variables, create train/test data splits

	Inputs: df, list of predicted variables, independent variable

	Returns: X_train, X_test, y_train, y_test
	'''
	X = df[pred_vars]
	y = df[dep_var]
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	return X_train, X_test, y_train, y_test
 

####5)BUILD-CLASSIFIER####
def build_tree_classifier(X_train, X_test, y_train, y_test, criterion):
	'''
	Build classifier, return results, predictions, and tree

	Inputs: X_train, X_test, y_train, y_test, criterion

	Returns: y_test, y_pred, tree
	'''
	tree = DecisionTreeClassifier(criterion)
	tree.fit(X_train, y_train)
	y_pred = tree.predict(X_test)
	return y_test, y_pred, tree


####6)EVALUATE-CLASSIFIER####
def calc_accuracy(y_test, y_pred):
	'''
	Returns the accuracy score based on the given test data 

	Inputs: y_test, y_pred

	Returns: accuracy score * 100
	'''
	return accuracy_score(y_test, y_pred)*100


def calc_confusion_matrix(y_test, y_pred, columns=None, index=None):
	'''
	Calculate the confusion matrix for the model

	Inputs: y_test, y_pred, columns=None, index=None

	Returns: df confusion matrix
	'''
	return pd.DataFrame(confusion_matrix(y_test, y_pred), columns, index)


def calc_feature_importance(pred_vars, tree):
	'''
	Calculate feature important of the predictor variables

	Input: pred_vars, tree

	Returns: dataframe with predictor variables and score of importance
	'''
	return pd.DataFrame({'feature':pred_vars, 'importance':tree.feature_importances_})


def calc_cross_val_score(tree, X, y):
	'''
	Calculate cross validation score for the model

	Inputs: tree, X, y

	Returns: cross validation score
	'''
	return cross_val_score(tree, X, y)


def visualize_tree(tree):
	'''
	Visualize the decision tree

	Inputs: tree

	Returns: image of tree 
	'''
	dot_data = StringIO()
	export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
	return Image(graph.create_png())




