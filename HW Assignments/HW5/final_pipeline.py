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
import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta


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


def fill_null_auto(df):
	'''
	Input: df

	Returns: None, updates df in place
	'''
	columns = df.columns.tolist()
	for column in columns:
		if df[column].isnull().any():
			df[column].fillna(df[column].mean(), inplace=True)


####4)GENERATE-FEATURES/PREDICTORS####
def convert_column_type(df, col, new_type):
	'''
	Convert a column to a new type 

	Inputs: df, column, new_type

	Returns: None, updates the column in df
	'''
	df[col] = df[col].astype(new_type)


def convert_to_disc(df, cols, bins, labels):
	'''
	Discretize continuous variables into discrete variables

	Inputs: df, cols, bins, labels

	Returns: None, updates the columns in df
	'''
	for col in cols:
		df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)


def convert_to_dummy(df, cols=None):
	'''
	Convert variables to dummy variables

	Inputs: df, col

	Returns: None, updates df
	'''
	df = pd.get_dummies(df, columns=cols)


def convert_to_datetime(df, cols):
	'''
	'''
	for col in cols:
		df[col] =  pd.to_datetime(df[col])


def convert_column_type(df, col, new_type):
	'''
	Convert a column to a new type 

	Inputs: df, column, new_type

	Returns: None, updates the column in df
	'''
	df[col] = df[col].astype(new_type)


	#don't forget to do this
	#fill null cols
	#p.fill_null_cols(df, ['students_reached'])


####TEMPORAL_VALIDATION_FUNCTIONS####
#Create temporal validation function in your pipeline that can create training and test sets over time. 
#You can choose the length of these splits based on analyzing the data. For example, the test sets could 
#be six months long and the training sets could be all the data before each test set.

#training 1: 1/1/2012 to 4/30/2012 (leave 60 days to see result of training data) you can define training as 1/1/2012 - 6/30/12 but you are only going to eval those that have posting dates 1/1/12-4/30/12 but the extra time is for those posted on like 4/29/12, 
#test 1: 7/1/2012 to 10/31/2012, you only eval those with posting dates 7/1/2012 to 10/31/2012 (so you have time to get the labels for those posted on 10/31/2012)

#training 2: 1/1/2012 to 10/31/2012, 
#test 2: 1/1/2013 to 4/30/2013 with 4/30 - 6/30 for evaluation/getting labels

#training 3: 1/1/2012 to 4/30/2013, 
#test 3: 7/1/2013 to 10/31/2013 with 10/31 to 12/31 for evaluation/getting labels


def temporal_validate(start_time, end_time, prediction_windows, outcome_days):
    '''
    Identifies times at which to split data into training and
    test sets over time.

    Input:
    start_time: start time of data
    end_time: last date of data, including labels and outcomes that we have
    prediction_windows: a list of how far out we want to predict, in months
    outcome_days: number of days needed to evaluate the outcome

    Returns:
    a list of training set start and end time, and testing set start and end time,
    for each temporally validated dataset

    Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
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


def temporal_split_train_test_dfs(df, dep_var, pred_vars, splits):
	'''
	Create train test splits for one group of train/test data (no X or y yet)

	Inputs: df, dep_var, pred_vars

	Returns: train test data in a list of dataframes
	'''
	df_train_test_list = []

	for split in splits:
		train_start, train_end, test_start, test_end = split
		train_df = df[(df['date_posted'] >= train_start) & (df['date_posted'] <= train_end)].copy()
		test_df = df[(df['date_posted'] >= test_start) & (df['date_posted'] <= test_end)].copy()
		df_train_test_list.extend([train_df, test_df])

	return df_train_test_list




def clean_data(df_list):
	'''
	'''
	for df in df_list:
		#fill null cols in place
		fill_null_auto(df)


def temporal_split(full_data, train_start, train_end, test_start, test_end, datetime_var, y_var):
	'''
	Splits data into temporally validated training set and test sets.
	Input:
	full_data: full dataframe
	train_start: start datetime for training set
	train_end: end datetime for training set
	test_start: start datetime for test set
	test_end: end datetime for test set
	datetime_var: the name of the temporal variable being split on
	y_var: the name of the dependent/output variable
	
	Returns:
	Training and testing sets for the independent and dependent variables.
	'''
	train_data = full_data[(full_data[datetime_var] >= train_start) & (full_data[datetime_var] <= train_end)]
	y_train = train_data[y_var]
	X_train = train_data.drop([y_var, datetime_var], axis = 1)

	test_data = full_data[(full_data[datetime_var] >= test_start) & (full_data[datetime_var] <= test_end)]
	y_test = test_data[y_var]
	X_test = test_data.drop([y_var, datetime_var], axis = 1)
	return X_train, X_test, y_train, y_test











def temporal_train_test(df, splits, datetime_var, y_var):
	'''
	'''
	for split in splits:
		X_train, X_test, y_train, y_test = temporal_split(df, split[0], split[1], split[2], split[3], datetime_var, y_var)








# def temp_val_train(df, date_col, date_val):
# 	'''
# 	Create temp val train df

# 	Inputs: df, date_col, date_val

# 	Returns: temp val train df
# 	'''
# 	return df[df[date_col] <= date_val]


# def temp_val_test(df, date_col, date_1, date_2):
# 	'''
# 	Create temp val test df

# 	Inputs: df, date_col, date_1, date_2

# 	Returns: temp val test df
# 	'''
# 	return df[df[date_col].between(date_1, date_2, inclusive=True)]











###############################################
'''
The following code used with permission:
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
    #plt.savefig(name)
    plt.show()
    

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df = pd.DataFrame(columns=('model_type', 'clf', 'parameters','auc-roc','p_at_5',
                                        'p_at_10', 'p_at_20', 'r_at_5', 'r_at_10', 'r_at_20', 'f1_at_5',
                                        'f1_at_10', 'f1_at_20'))
    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        print(models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            clf.set_params(**p)
            if 'SVM'in models_to_run[index]:
                y_pred_probs = clf.fit(X_train, y_train).decision_function(X_test)
            else:
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
            y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
            results_df.loc[len(results_df)] = [models_to_run[index], clf, p,
                                              roc_auc_score(y_test, y_pred_probs),
                                              precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                              precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                              precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                              recall_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                              recall_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                              recall_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                                              f1_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                              f1_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                              f1_at_k(y_test_sorted, y_pred_probs_sorted, 20.0)]
            plot_precision_recall_n(y_test,y_pred_probs,clf)
    return results_df


######################################
#old code


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





