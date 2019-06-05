'''
CAPP30254-ML: HW 5 - Machine Learning Pipeline

Code for Machine Learning Pipeline Version 4 (Revision of HW3)

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
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import cross_val_score
import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


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


####3)PRE-PROCESS/CLEAN_DATA####
def check_null_counts(df):
	'''
	Get null counts in dataframe

	Inputs: df

	Returns: null counts
	'''
	null_counts = df.isnull().sum()
	return null_counts


def fill_null_auto(df):
	'''
	Input: df

	Returns: None, updates df in place
	'''
	cols = df.columns.tolist()
	for col in cols:
		if df[col].isnull().any():
			df[col].fillna(df[col].mean(), inplace=True)


####4)GENERATE-FEATURES/PREDICTORS####
def convert_column_type(df, col, new_type):
	'''
	Convert a column to a new type 

	Inputs: df, column, new_type

	Returns: None, updates the column in df
	'''
	df[col] = df[col].astype(new_type)


def convert_to_datetime(df, cols):
	'''
	'''
	for col in cols:
		df[col] =  pd.to_datetime(df[col])


def convert_tf_to_binary(df):
	'''
	'''
	cols = df.columns.tolist()
	for col in cols:
		if is_string_dtype(df[col]):
			if (df[col] == 't').any():
				df[col] = df[col].map({'f': 0, 't': 1})


def convert_to_categorical(df, cat_cols):
	'''
	'''
	for col in cat_cols:
		df[col] = df[col].astype('category').cat.codes


def convert_to_disc(df, disc_cols, bins=4, labels=[0,1,2,3]):
	'''
	Discretize continuous variables into discrete variables

	Inputs: df, cols, bins, labels

	Returns: None, updates the columns in df
	'''
	for col in disc_cols:
		df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)


####4)TRAIN/TEST DATA####
def train_test_split(df, pred_vars, dep_var):
	'''
	Choose predictors and dependent variables, create train/test data splits

	Inputs: df, list of predicted variables, independent variable

	Returns: X_train, X_test, y_train, y_test
	'''
	X = df[pred_vars]
	y = df[dep_var]
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	return X_train, X_test, y_train, y_test


####TEMPORAL_VALIDATION_FUNCTIONS####
#Create temporal validation function in your pipeline that can create training and test sets over time. 
#You can choose the length of these splits based on analyzing the data. For example, the test sets could 
#be six months long and the training sets could be all the data before each test set.

#Guidelines for Analysis:
#training 1: 1/1/2012 to 4/30/2012 (leave 60 days to see result of training data) 
#test 1: 7/1/2012 to 10/31/2012

#training 2: 1/1/2012 to 10/31/2012
#test 2: 1/1/2013 to 4/30/2013

#training 3: 1/1/2012 to 4/30/2013
#test 3: 7/1/2013 to 10/31/2013

def temporal_splits(start_time, end_time, prediction_windows, outcome_days):
	'''
	Split temporal data into windows based on inputs

	Inputs: start_time, end_time. prediction_windows, outcome days

	Returns: splits - a list of start and end times

	code updated with permission: https://github.com/rayidghani/magicloops/blob/master/simpleloop.py, credit to Rayid Ghani
	'''
	splits = []
	start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
	end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
	actual_end_time_date = end_time_date - relativedelta(days=+(outcome_days+1))
	for prediction_window in prediction_windows:
		on_window = 1
		test_end_time = start_time_date
		while (actual_end_time_date > test_end_time):
			train_start_time = start_time_date
			train_end_time = train_start_time + on_window * relativedelta(months=+prediction_window) - relativedelta(days=+(outcome_days+2))
			test_start_time = train_start_time + on_window * relativedelta(months=+prediction_window)
			test_end_time = test_start_time + relativedelta(months=+prediction_window) - relativedelta(days=+(outcome_days+2))
			splits.append([train_start_time, train_end_time, test_start_time, test_end_time])
			on_window += 1
	return splits


def temporal_split_train_test_dfs(df, splits):
	'''
	Create train test splits on data, based on temporal splits

	Inputs: df, splits

	Returns: train test data in a list of dataframes, based on splits
	'''
	df_train_test_list = []
	for split in splits:
		train_start, train_end, test_start, test_end = split
		train_df = df[(df['date_posted'] >= train_start) & (df['date_posted'] <= train_end)].copy()
		test_df = df[(df['date_posted'] >= test_start) & (df['date_posted'] <= test_end)].copy()
		df_train_test_list.extend([train_df, test_df])
	return df_train_test_list


def clean_data(df_list, cat_cols=None, disc_cols=None):
	'''
	Clean and process data after train/test split. 
	Cleaning includes filling nulls, converting true/false to binaries, 
	converting columns to categorical, and continuous to discrete

	Inputs: 
	- df_list - list of test/control dataframes
	- cat_cols - list of columns to convert to categorical
	- disc_cols - list of columns to convert to discrete

	Returns: None, updates df in place for all cleaning

	'''
	for df in df_list:
		fill_null_auto(df)
		convert_tf_to_binary(df)
		convert_to_categorical(df, cat_cols)
		convert_to_disc(df, disc_cols)


####RUN PIPELINE####
###############################################
'''
Pipeline code updated with permission:
https://github.com/rayidghani/magicloops/blob/master/simpleloop.py, credit to Rayid Ghani
'''
def define_clfs_params():
	clfs = {
		'BG': BaggingClassifier(n_estimators=10),
		'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
		'LR': LogisticRegression(penalty='l1', C=1e5),
		'SVM': svm.LinearSVC(random_state=0, penalty='l1', dual=False),
		'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
		'DT': DecisionTreeClassifier(),
		'KNN': KNeighborsClassifier(n_neighbors=3),
		'NB': GaussianNB()}

	grid = {
		'BG': {'n_estimators': [10,100]}, 
		'RF': {'n_estimators': [1,10,100], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2']},
		'LR': { 'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10]},
		'GB': {'n_estimators': [1,10,100], 'learning_rate' : [0.05,0.1,0.5],'subsample' : [0.5,1.0], 'max_depth': [10,20,50]},
		'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10]},
		'SVM' :{'penalty':['l1','l2'], 'C' :[0.01,0.1,1]},
		'KNN' :{'n_neighbors': [1,5,10,25],'weights': ['uniform','distance']},
		'NB': {}
		}
	
	return clfs, grid


def models_to_run():
	models_to_run = ['RF', 'AB', 'DT', 'GB', 'SVM', 'KNN', 'LR', 'BG']
	return models_to_run
	

def joint_sort_descending(l1, l2):
	idx = np.argsort(l1)[::-1]
	return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
	cutoff_index = int(len(y_scores) * (k / 100.0))
	test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
	return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
	y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
	preds_at_k = generate_binary_at_k(y_scores, k)
	precision = precision_score(y_true, preds_at_k)
	return precision


def recall_at_k(y_true, y_scores, k):
	y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
	preds_at_k = generate_binary_at_k(y_scores, k)
	recall = recall_score(y_true, preds_at_k)
	return recall


def f1_at_k(y_true, y_scores, k):
	precision = precision_at_k(y_true, y_scores, k)
	recall = recall_at_k(y_true, y_scores, k)
	return 2 * (precision * recall)/(precision + recall)


def plot_precision_recall_n(y_true, y_prob, model_name):
	from sklearn.metrics import precision_recall_curve
	y_score = y_prob
	precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
	precision_curve = precision_curve[:-1]
	recall_curve = recall_curve[:-1]
	pct_above_per_thresh = []
	number_scored = len(y_score)
	for value in pr_thresholds:
		num_above_thresh = len(y_score[y_score>=value])
		pct_above_thresh = num_above_thresh / float(number_scored)
		pct_above_per_thresh.append(pct_above_thresh)
	pct_above_per_thresh = np.array(pct_above_per_thresh)
	
	plt.clf()
	fig, ax1 = plt.subplots()
	ax1.plot(pct_above_per_thresh, precision_curve, 'b')
	ax1.set_xlabel('percent of population')
	ax1.set_ylabel('precision', color='b')
	ax2 = ax1.twinx()
	ax2.plot(pct_above_per_thresh, recall_curve, 'r')
	ax2.set_ylabel('recall', color='r')
	ax1.set_ylim([0,1])
	ax1.set_ylim([0,1])
	ax2.set_xlim([0,1])
   
	name = model_name
	plt.title(name)
	plt.show()


#run all models on all data, including temporal splits
def clf_loop_all_data(models_to_run, clfs, grid, train_test_dfs, pred_vars, dep_var):
	"""Runs the loop using models_to_run, clfs, gridm and the data
	"""
	results_df = pd.DataFrame(columns=('time_period', 'model_type', 'clf', 'parameters','auc-roc',
										'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', 
										'r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_30', 'r_at_50', 
										'f1_at_1', 'f1_at_2', 'f1_at_5', 'f1_at_10', 'f1_at_20', 'f1_at_30', 'f1_at_50'))
	counter = 1

	for i, df in enumerate(train_test_dfs):
		if i % 2 == 0:
			train_df = train_test_dfs[i]
			test_df = train_test_dfs[i + 1]
			X_train = train_df[pred_vars]
			y_train = train_df[dep_var]
			X_test = test_df[pred_vars]
			y_test = test_df[dep_var]
			period = "train_test_data_period" + "_" + str(counter)
			print(period)
			counter += 1

			for index, clf in enumerate([clfs[x] for x in models_to_run]):
				print(models_to_run[index])
				parameter_values = grid[models_to_run[index]]
				for p in ParameterGrid(parameter_values):
					clf.set_params(**p)
					if 'SVM'in models_to_run[index]:
						y_pred_probs = clf.fit(X_train, y_train).decision_function(X_test)
					else:
						y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
					y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
					results_df.loc[len(results_df)] = [period, models_to_run[index], clf, p,
														roc_auc_score(y_test, y_pred_probs),
														precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
														precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
														precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
														precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
														precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
														precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
														precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
														recall_at_k(y_test_sorted, y_pred_probs_sorted, 1.0),
														recall_at_k(y_test_sorted, y_pred_probs_sorted, 2.0),
														recall_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
														recall_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
														recall_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
														recall_at_k(y_test_sorted, y_pred_probs_sorted, 30.0),
														recall_at_k(y_test_sorted, y_pred_probs_sorted, 50.0),
														f1_at_k(y_test_sorted, y_pred_probs_sorted, 1.0),
														f1_at_k(y_test_sorted, y_pred_probs_sorted, 2.0),
														f1_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
														f1_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
														f1_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
														f1_at_k(y_test_sorted, y_pred_probs_sorted, 30.0),
														f1_at_k(y_test_sorted, y_pred_probs_sorted, 50.0)]
					plot_precision_recall_n(y_test,y_pred_probs,clf)
	
	return results_df

####END####
###############################################


