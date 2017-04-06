#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from tester import test_classifier, dump_classifier_and_data
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import svm, grid_search, datasets
from sklearn import linear_model, datasets
from sklearn import tree
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

### ============================================================================
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# features_list is updated in Task 3 after performing the necessary cleaning on the data				

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

# Converting Python dictionary to Pandas DataFrame 
enron_data = pd.DataFrame.from_dict(data_dict, orient='index', dtype=None)

### ============================================================================
### Task 2: Remove outliers

# Replace the 'NaN' values with Numpy NaN
enron_data.replace(to_replace='NaN', value=np.nan, inplace=True)

# Changing the type of data from object(aka. String) to numeric and filling NaN values with zero
# Doing this will also change the NaN values in 'email_address' column to zero
enron_data = enron_data.apply(lambda x: pd.to_numeric(x, errors='ignore')).fillna(0)

# Dropping the row named 'TOTAL' as it does not describe any specific person in the dataset
enron_data = enron_data.drop(['TOTAL'])

# Dropping the 3 columns(aka.features) that either had zero observations or not enough to be useful for analysis
enron_data_2 = enron_data.drop(['restricted_stock_deferred','director_fees','loan_advances'], axis=1)

### ============================================================================
### Task 3: Create new feature(s)

# Adding two new columns to my existing dataframe
enron_data_2['fraction_from_poi'] = ""
enron_data_2['fraction_to_poi'] = ""

# Looping through each row to calculate the fraction for to and from emails
for i in range(len(enron_data_2['to_messages'])):
    if enron_data_2['to_messages'][i] != 0.0:
        v = float(enron_data_2['from_poi_to_this_person'][i]) / float(enron_data_2['to_messages'][i])
        enron_data_2.set_value(enron_data_2.index[i], 'fraction_from_poi', v)
        
    if enron_data_2['to_messages'][i] == 0.0 and enron_data_2['from_poi_to_this_person'][i] == 0.0:
        enron_data_2.set_value(enron_data_2.index[i], 'fraction_from_poi', 0.0)

    if enron_data_2['from_messages'][i] != 0.0:
        v = float(enron_data_2['from_this_person_to_poi'][i]) / float(enron_data_2['from_messages'][i])
        enron_data_2.set_value(enron_data_2.index[i], 'fraction_to_poi', v)
        
    if enron_data_2['from_messages'][i] == 0.0 and enron_data_2['from_this_person_to_poi'][i] == 0.0:
        enron_data_2.set_value(enron_data_2.index[i], 'fraction_to_poi', 0.0)

# Changing Pandas to dict to work with
enron_data2_dict = enron_data_2.to_dict(orient='index')

### Store to my_dataset for easy export below.
my_dataset = enron_data2_dict

# list of my selected features after cleaning the data from outliers/features I didn't need
features_list = ['poi','salary','to_messages', 'deferral_payments',
                'total_payments','exercised_stock_options','bonus','restricted_stock',
                'shared_receipt_with_poi','total_stock_value','expenses','from_messages',
                'other','from_this_person_to_poi','deferred_income','long_term_incentive',
                'from_poi_to_this_person', 'fraction_from_poi', 'fraction_to_poi']
                #'restricted_stock_deferred' -> removed because of zero observations,
				#'director_fees' -> removed because of zero observation,
				#'loan_advances' -> removed because of 1 POI out ot 3 observations,

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### ============================================================================
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Choosing Naive Bayes algorithm
### ============================================================================
nb_algorithm = GaussianNB()

# Scaling the feature values between [0,1] to remove any negative value
scaler = preprocessing.MinMaxScaler()

# Do feature selection using SelectKBest to have features with the highest scores
kbest = SelectKBest()

# Tuning parameters to be used in the algorithm
param_grid = {'kbest__k' : list(range(1,(len(features_list)-1)))}

# Spliting data randomly to train and test for cross validation
sss = StratifiedShuffleSplit(labels, 1000, random_state = 1)

# Putting it all together in a pipeline
pipe =  Pipeline(steps=[('scaler',scaler),('kbest', kbest), ("nb_algorithm", nb_algorithm)])

# Creating the classifier to fit
nbcclf = grid_search.GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

nbcclf.fit(features, labels)

# Choose the best estimator among different features that were selected
clf = nbcclf.best_estimator_

### Choosing Decision Tree algorithm
### ============================================================================
#dtree =  tree.DecisionTreeClassifier()

# Scaling the feature values between [0,1] to remove any negative value
#scaler = preprocessing.MinMaxScaler()

# Do feature selection using SelectKBest to have features with the highest scores
#kbest = SelectKBest()

# Tuning the algorithm parameters
#param_grid = {"dtree__min_samples_split": [2,4,6], # use 3 different set of minimum split
#              "dtree__min_samples_leaf":[2,4,6],
#              'kbest__k' : list(range(1,(len(features_list)-1)))}

# Spliting data randomly to train and test for cross validation
#sss = StratifiedShuffleSplit(labels, 1000, random_state = 1)

# Putting it all together in a pipeline
#pipe =  Pipeline(steps=[('scaler',scaler),('kbest', kbest), ("dtree", dtree)])

# Creating the classifier to fit
#nbcclf = grid_search.GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

#nbcclf.fit(features, labels)

# Choose the best estimator among different features that were selected
#clf = dtcclf.best_estimator_

#test_classifier(clf, my_dataset, features_list)

### Choosing Logistic Regression algorithm
### ============================================================================
# Choosing decision tree algorithm
#lgr =  linear_model.LogisticRegression()

# Scaling the feature values between [0,1] to remove any negative value
#scaler = preprocessing.MinMaxScaler()

# Do feature selection using SelectKBest to have features with the highest scores
#kbest = SelectKBest()

# Tuning the algorithm parameters
#param_grid = {'lgr__C': [0.1, 1, 10, 100, 1000], # parameter for logistic regression to change regularization
#              'kbest__k' : list(range(1,(len(features_list)-1)))}

# Spliting data randomly to train and test for cross validation
#sss = StratifiedShuffleSplit(labels, 1000, random_state = 1)

# Putting it all together in a pipeline
#pipe =  Pipeline(steps=[('scaler',scaler),('kbest', kbest), ("lgr", lgr)])

# Creating the classifier to fit
#lgrcclf = grid_search.GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

#lgrcclf.fit(features, labels)

# Choose the best estimator among different features that were selected
#clf = lgrcclf.best_estimator_

#test_classifier(clf, my_dataset, features_list)

### ============================================================================
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

print "Results from choosing Naive Bayes classifier"
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)