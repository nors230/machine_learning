'''
CAPP30254-ML: HW 3 - Machine Learning Pipeline

Code for Machine Learning Pipeline Version 2

Nora Hajjar
'''
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
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

##DECISION TREE##
def build_tree_classifier(X_train, X_test, y_train, y_test, criterion='gini'):
	'''
	Build classifier, return results, predictions, and tree

	Inputs: X_train, X_test, y_train, y_test, criterion

	Returns: y_test, y_pred, tree
	'''
	tree = DecisionTreeClassifier(criterion)
	tree.fit(X_train, y_train)
	y_pred = tree.predict(X_test)
	return y_test, y_pred, tree


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


##LOGISTIC REGRESSION##
def logistic_regression(X_train, X_test, y_train, y_test):
	'''
	Build logistic regression

	Inputs: X_train, X_test, y_train, y_test

	Returns: y_test, y_pred, pred_scores
	'''
	lr = LogisticRegression()
	lr.fit(X_train, y_train)
	y_pred = lr.predict(X_test)
	pred_scores = lr.predict_proba(X_test)
	return y_test, y_pred, pred_scores


##SVM##
def svm(X_train, X_test, y_train, y_test):
	'''
	Build svm

	Inputs: X_train, X_test, y_train, y_test

	Returns: y_test, y_pred, confidence_score
	'''
	svm = LinearSVC()
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	confidence_score = svm.decision_function(X_test)
	return y_test, y_pred, confidence_score


##K-NEAREST NEIGHBOR##
def knn(X_train, X_test, y_train, y_test, n_neighbors=5, 
	weights='uniform', algorithm='auto', leaf_size=30, p=2, 
	metric='minkowski', metric_params=None):
	'''
	Build knn

	Inputs: X_train, X_test, y_train, y_test

	Returns: y_test, y_pred, pred_scores
	'''
	knn = KNeighborsClassifier(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	pred_scores = knn.predict_proba(X_test)
	return y_test, y_pred, pred_scores


#####ENSEMBLE METHODS#####
##RANDOM FOREST##
def random_forest(X_train, X_test, y_train, y_test):
	'''
	Build random forest

	Inputs: X_train, X_test, y_train, y_test

	Returns: y_test, y_pred, pred_scores
	'''
	rand = RandomForestClassifier()
	rand.fit(X_train, y_train)
	y_pred = rand.predict(X_test)
	pred_scores = rand.predict_proba(X_test)
	return y_test, y_pred, pred_scores


##BOOSTING##
def boosting(X_train, X_test, y_train, y_test):
	'''
	Build boosting

	Inputs: X_train, X_test, y_train, y_test

	Returns: y_test, y_pred, pred_scores
	'''
	boost = AdaBoostClassifier()
	boost.fit(X_train, y_train)
	y_pred = boost.predict(X_test)
	pred_scores = boost.predict_proba(X_test)
	return y_test, y_pred, pred_scores


##BAGGING##
def bagging(X_train, X_test, y_train, y_test):
	'''
	Build bagging

	Inputs: X_train, X_test, y_train, y_test

	Returns: y_test, y_pred, pred_scores
	'''
	bagging = BaggingClassifier()
	bagging.fit(X_train, y_train)
	y_pred = bagging.predict(X_test)
	pred_scores = bagging.predict_proba(X_test)
	return y_test, y_pred, pred_scores


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


def calc_precision(y_test, y_pred):
	'''
	Calculate precision

	Inputs: y_test, y_pred

	Returns: precision score
	'''
	return precision_score(y_test, y_pred)


def calc_recall(y_test, y_pred):
	'''
	Calculate recall

	Inputs: y_test, y_pred

	Returns: recall score
	'''
	return recall_score(y_test, y_pred)


def calc_f1(y_test, y_pred):
	'''
	Calculate f1

	Inputs: y_test, y_pred

	Returns: f1 score
	'''
	return f1_score(y_test, y_pred)


def calc_auc_roc(y_test, pred_scores):
	'''
	Calculate auc_roc

	Inputs: y_test, pred_scores

	Returns: auc_roc score
	'''
	return roc_auc_score(y_test, pred_scores)


def plot_precision_recall(predicted_scores, true_labels):
	'''
	Plot precision-recall curve

	Inputs: predicted_scores, true_labels

	Returns: plots
	'''
	precision, recall, thresholds = precision_recall_curve(true_labels, predicted_scores)
	plt.plot(recall, precision, marker='.')
	plt.show()


#BUILD A CALC TABLE FOR DIFFERENT THRESHOLDS, REUSABLE ACROSS DATA
def calc_scores_pred(model_name, time_period, y_test, pred_scores, threshold_list):
	'''
	Calc scores

	Inputs: model_name, time_period, y_test, pred_scores, threshold_list

	Returns: dataframe with score results
	'''
	COLUMNS = ['model_name', 'time_period',
		'threshold', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']

	output_df = pd.DataFrame()
	for threshold in threshold_list:
		pred_label = [1 if x[1]>threshold else 0 for x in pred_scores]
		accuracy = accuracy_score(pred_label, y_test)
		precision = precision_score(y_test, pred_label)
		recall = recall_score(y_test, pred_label)
		f1 = f1_score(y_test, pred_label)
		auc_roc = roc_auc_score(y_test, pred_label)
		df2 = pd.DataFrame([[model_name, time_period, threshold, accuracy, precision, recall, f1, auc_roc]], 
			columns=COLUMNS)
		output_df = pd.concat([output_df, df2])
	return output_df


#run all models (excluding SVM because it uses confidence_score, runs separately)
def run_models(model_dict, time_dict, threshold_list):
	'''
	Run models

	Inputs: model_dict, time_dict, threshold_list

	Returns: dataframe with full results
	'''
	final_df = pd.DataFrame()
	for time_period, train_test_list in time_dict.items():
		(X_train, X_test, y_train, y_test) = train_test_list
		for model_name, model_class in model_dict.items():
			model = model_class
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			pred_scores = model.predict_proba(X_test)
			results_df = calc_scores_pred(model_name, time_period, y_test, pred_scores, threshold_list)
			final_df = pd.concat([final_df, results_df])
	return final_df.sort_values(by=['model_name', 'time_period'], ascending=[1, 1])


def calc_scores_conf(model_name, time_period, y_test, confidence_score, threshold_list):
	'''
	Calc scores

	Inputs: model_name, time_period, y_test, pred_scores, threshold_list

	Returns: dataframe with score results
	'''
	COLUMNS = ['model_name', 'time_period',
		'threshold', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']

	output_df = pd.DataFrame()
	for threshold in threshold_list:
		pred_label = [1 if x >threshold else 0 for x in confidence_score]
		accuracy = accuracy_score(pred_label, y_test)
		precision = precision_score(y_test, pred_label)
		recall = recall_score(y_test, pred_label)
		f1 = f1_score(y_test, pred_label)
		auc_roc = roc_auc_score(y_test, pred_label)
		df2 = pd.DataFrame([[model_name, time_period, threshold, accuracy, precision, recall, f1, auc_roc]], 
			columns=COLUMNS)
		output_df = pd.concat([output_df, df2])
	return output_df


def run_models_conf(model_dict, time_dict, threshold_list):
	'''
	Run models

	Inputs: model_dict, time_dict, threshold_list

	Returns: dataframe with full results
	'''
	final_df = pd.DataFrame()
	for time_period, train_test_list in time_dict.items():
		(X_train, X_test, y_train, y_test) = train_test_list
		for model_name, model_class in model_dict.items():
			model = model_class
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			confidence_score = model.decision_function(X_test)
			results_df = calc_scores_conf(model_name, time_period, y_test, confidence_score, threshold_list)
			final_df = pd.concat([final_df, results_df])
	return final_df.sort_values(by=['model_name', 'time_period'], ascending=[1, 1])


####TEMPORAL_VALIDATION_FUNCTIONS####
#Create temporal validation function in your pipeline that can create training and test sets over time. 
#You can choose the length of these splits based on analyzing the data. For example, the test sets could 
#be six months long and the training sets could be all the data before each test set.
def temp_val_train(df, date_col, date_val):
	'''
	Create temp val train df

	Inputs: df, date_col, date_val

	Returns: temp val train df
	'''
	return df[df[date_col] <= date_val]


def temp_val_test(df, date_col, date_1, date_2):
	'''
	Create temp val test df

	Inputs: df, date_col, date_1, date_2

	Returns: temp val test df
	'''
	return df[df[date_col].between(date_1, date_2, inclusive=True)]


def extract_train_test(df_train, df_test, dep_var, pred_vars):
	'''
	Create train test splits

	Inputs: df_train, df_test, dep_var, pred_vars

	Returns: train test splits
	'''
	X_train = df_train[pred_vars]
	X_test = df_test[pred_vars]
	y_train = df_train[dep_var]
	y_test = df_test[dep_var]
	return X_train, X_test, y_train, y_test



