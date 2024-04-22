# -*- coding: utf-8 -*-
'''
Python script for defining, classifying, and building models to address brewery hotspots.
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# pull in city level data
city_df = pd.read_csv('../data/city_level.csv')

# check for null values
# city_df.isnull().sum()

'''
city                            0
city_brewery_count              0
state                           0
state_brewery_count             0
bar                             0
brewpub                         0
closed                          0
contract                        0
large                           0
location                        0
micro                           0
nano                            0
planning                        0
proprietor                      0
regional                        0
taproom                         0
college_town                    0
major_city                      0
state_national_park_count       0
national_park_vistors           0
ski_resort                      0
ski_resort_count                0
tech_city                       0
total population              723
% age 21+                     723
% age 65+                     723
% race - white                723
% race - black                723
% race - american native      723
% race  - asian               723
% race - pacific islander     723
% race - other                723
% race - hispanic/latino      723
% male                        723
2021 median income           2512
region                          0
brewery_concentration         723
per_capita_ranked             723
custom_ranked                   0
'''

# check region
# city_df['region'].value_counts()
'''
midwest      769
west         762
northeast    699
south        678
'''

'''
Notes
    
The eventuals:
We will want to remove the following columns in preparation for modeling:
    - city (unique identifiers)
    - state (51 more columns don't seem necessary, especially when there is
             state specific data)
    - 2021 median income (86.4% missing values)
    - brewery_concentration (a variable created from population and city brewery count)
    - per_capita_ranked (a variable crated from brewery_concentration)
    - encode region (4 categories, *probably* important)
    - city_brewery_count (basis for custom_ranked)

Creating hotspot criteria:
    - we used a method which ranked the city from 1-6 (6 being consider the max hotspot)
    - this was based on city_brewery_count (reason for dropping)
    - create ranking system based on concentration
    - this will need an immediate removal of the rows requiring census data
'''

# remove income column
model_df = city_df.drop(['2021 median income'], axis=1, inplace=True)

# remove rows without census data
model_df.dropna(subset=['total population'], axis=0, inplace=True)

# check nulls
model_df.isnull().sum()

'''
city                         0
city_brewery_count           0
state                        0
state_brewery_count          0
bar                          0
brewpub                      0
closed                       0
contract                     0
large                        0
location                     0
micro                        0
nano                         0
planning                     0
proprietor                   0
regional                     0
taproom                      0
college_town                 0
major_city                   0
state_national_park_count    0
national_park_vistors        0
ski_resort                   0
ski_resort_count             0
tech_city                    0
total population             0
% age 21+                    0
% age 65+                    0
% race - white               0
% race - black               0
% race - american native     0
% race  - asian              0
% race - pacific islander    0
% race - other               0
% race - hispanic/latino     0
% male                       0
region                       0
brewery_concentration        0
per_capita_ranked            0
custom_ranked                0
'''

# size of dataset
model_df.shape

# no longer need to create concentrations here

# drop brewery concentration (collinearity with ranking)
# drop city and state
# drop city_brewery_count
# drop per_capita_ranked
columns_to_drop = ['city', 'state', 'brewery_concentration', 'city_brewery_count', 'per_capita_ranked']
model_df.drop(columns_to_drop, axis=1, inplace=True)

# encode regions
model_df = pd.get_dummies(model_df, columns=['region'], drop_first=True)

# rename custom_ranking to ranking
model_df.rename({'custom_ranked':'ranked'}, inplace=True, axis=1)

# check datatypes - want all numerical
# model_df.info()

# may need to rename columns for interpretability and normality
# export head of column for use in website
# model_df_head = model_df.head(10)
# model_df_head.to_csv('../data/model_data.csv', index = False)

'''
Modeling
- perform default algorithms, report metrics
- apply scaling, perform default algorithms, report metrics
- run hypertuning on top 3 algorithms
'''

# import general sklearn libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# import specific sklearn libraries
# sklearn decision tree - tree.DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# sklearn logistic regression
from sklearn.linear_model import LogisticRegression
# sklearn knn
from sklearn.neighbors import KNeighborsClassifier 
# sklearn svm (could import svm and use svm.SVC, but we'll directly import SVC)
from sklearn.svm import SVC
# sklearn naive bayes
from sklearn.naive_bayes import GaussianNB
# sklearn linear discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# maybe random forest?

'''
1. Default Algorithms
'''
# result storing
result_columns = ['accuracy', 'precision', 'recall', 'f1', 'model']
round_1_results = pd.DataFrame(columns=result_columns)

# training and test sets
X = model_df.drop('ranked', axis=1)
y = model_df['ranked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

'''
Decision Tree Default
'''
# decision tree
# training
clf = DecisionTreeClassifier().fit(X_train, y_train)
# prediction
clf_y_pred = clf.predict(X_test)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Decision Tree Classification'}
clf_results = pd.DataFrame([clf_results])
round_1_results = pd.concat([round_1_results, clf_results], ignore_index=True)

'''
Logistic Regression Default
'''
# logistic regression
# training
# the model itself failed to converge, so deviating slightly from default to allow
# max_iter from default 100 to 10000
clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
# prediction
clf_y_pred = clf.predict(X_test)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Logistic Regression'}
clf_results = pd.DataFrame([clf_results])
round_1_results = pd.concat([round_1_results, clf_results], ignore_index=True)

'''
K-Nearest-Neighbors Default
'''
# KNN (note that KNN method requires arrays for fitting and training)
# training
clf = KNeighborsClassifier().fit(np.array(X_train), np.array(y_train))
# prediction (need to change X_test to X_test.values)
clf_y_pred = clf.predict(np.array(X_test))
# results
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'KNN'}
clf_results = pd.DataFrame([clf_results])
round_1_results = pd.concat([round_1_results, clf_results], ignore_index=True)

'''
Support Vector Machine - Classification Default
'''
# SVM
# training
clf = SVC().fit(X_train, y_train)
# prediction
clf_y_pred = clf.predict(X_test)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Support Vector Machine'}
clf_results = pd.DataFrame([clf_results])
round_1_results = pd.concat([round_1_results, clf_results], ignore_index=True)

'''
Naive Bayes Default
'''
# Naive Bayes
# training
clf = GaussianNB().fit(X_train, y_train)
# prediction
clf_y_pred = clf.predict(X_test)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Naive Bayes'}
clf_results = pd.DataFrame([clf_results])
round_1_results = pd.concat([round_1_results, clf_results], ignore_index=True)

'''
Linear Discriminant Analysis Default
'''
# Linear Discriminant Analysis
# training
clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
# prediction
clf_y_pred = clf.predict(X_test)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Linear Discriminant Analysis'}
clf_results = pd.DataFrame([clf_results])
round_1_results = pd.concat([round_1_results, clf_results], ignore_index=True)

'''
Results from Round 1:
    
   accuracy precision    recall        f1                         model
0  0.926829  0.928881  0.926829   0.92698  Decision Tree Classification
1     0.875   0.79324     0.875  0.831636           Logistic Regression
2  0.873476  0.841246  0.873476   0.84128                           KNN
3  0.878049  0.783841  0.878049   0.82793        Support Vector Machine
4  0.036585  0.873634  0.036585  0.016129                   Naive Bayes
5  0.896341  0.874851  0.896341  0.883871  Linear Discriminant Analysis
'''

# export round 1 results for website - commented out post script run
# round_1_results.to_csv('../data/hotspot_round1.csv', index = False)

'''
Round 2 - Apply Scaling
'''

# normalizing data
scaler = StandardScaler()
X_train_normal = scaler.fit_transform(X_train)
X_test_normal = scaler.fit_transform(X_test)

# result storing
round_2_results = pd.DataFrame(columns=result_columns)

'''
Decision Tree Default - Scaled Data
'''
# decision tree
# training
clf = DecisionTreeClassifier().fit(X_train_normal, y_train)
# prediction
clf_y_pred = clf.predict(X_test_normal)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Decision Tree Classification'}
clf_results = pd.DataFrame([clf_results])
round_2_results = pd.concat([round_2_results, clf_results], ignore_index=True)

'''
Logistic Regression Default - Scaled Data
'''
# logistic regression
# training
# the model itself failed to converge, so deviating slightly from default to allow
# max_iter from default 100 to 10000
clf = LogisticRegression(max_iter=10000).fit(X_train_normal, y_train)
# prediction
clf_y_pred = clf.predict(X_test_normal)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Logistic Regression'}
clf_results = pd.DataFrame([clf_results])
round_2_results = pd.concat([round_2_results, clf_results], ignore_index=True)

'''
K-Nearest-Neighbors Default - Scaled Data
'''
# KNN (note that KNN method requires arrays for fitting and training)
# training
clf = KNeighborsClassifier().fit(X_train_normal, y_train)
# prediction (need to change X_test to X_test.values)
clf_y_pred = clf.predict(X_test_normal)
# results
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'KNN'}
clf_results = pd.DataFrame([clf_results])
round_2_results = pd.concat([round_2_results, clf_results], ignore_index=True)

'''
Support Vector Machine - Classification Default - Scaled Data
'''
# SVM
# training
clf = SVC().fit(X_train_normal, y_train)
# prediction
clf_y_pred = clf.predict(X_test_normal)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Support Vector Machine'}
clf_results = pd.DataFrame([clf_results])
round_2_results = pd.concat([round_2_results, clf_results], ignore_index=True)

'''
Naive Bayes Default - Scaled Data
'''
# Naive Bayes
# training
clf = GaussianNB().fit(X_train_normal, y_train)
# prediction
clf_y_pred = clf.predict(X_test_normal)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Naive Bayes'}
clf_results = pd.DataFrame([clf_results])
round_2_results = pd.concat([round_2_results, clf_results], ignore_index=True)

'''
Linear Discriminant Analysis Default - Scaled Data
'''
# Linear Discriminant Analysis
# training
clf = LinearDiscriminantAnalysis().fit(X_train_normal, y_train)
# prediction
clf_y_pred = clf.predict(X_test_normal)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'accuracy': clf_accuracy,
               'precision': clf_precision,
               'recall': clf_recall,
               'f1': clf_f1,
               'model': 'Linear Discriminant Analysis'}
clf_results = pd.DataFrame([clf_results])
round_2_results = pd.concat([round_2_results, clf_results], ignore_index=True)


'''
Results from Round 2:
    
   accuracy precision    recall        f1                         model
0  0.908537  0.914935  0.908537  0.910845  Decision Tree Classification
1  0.928354   0.92785  0.928354   0.92684           Logistic Regression
2  0.884146  0.831365  0.884146  0.851703                           KNN
3  0.907012  0.860135  0.907012  0.876955        Support Vector Machine
4  0.022866  0.866248  0.022866   0.00685                   Naive Bayes
5   0.91311  0.891192   0.91311  0.898986  Linear Discriminant Analysis
'''

# export round 1 results for website - commented out post script run
# round_2_results.to_csv('../data/hotspot_round2.csv', index = False)

'''
Visualizing the Results
'''
# round 1
melted_1 = round_1_results.melt(id_vars = ['model'],
                                             value_vars = ['accuracy', 'precision', 'recall', 'f1'],
                                             var_name = 'metric')
melted_1 = melted_1.rename(columns={'model':'Model', 'metric': 'Metric'})
melted_1['Metric'] = melted_1['Metric'].str.capitalize()
sns.barplot(data = melted_1, x = 'value', y = 'Metric', hue = 'Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Default Models Non-Scaled Data')

# round 2
melted_2 = round_2_results.melt(id_vars = ['model'],
                                             value_vars = ['accuracy', 'precision', 'recall', 'f1'],
                                             var_name = 'metric')
melted_2 = melted_2.rename(columns={'model':'Model', 'metric': 'Metric'})
melted_2['Metric'] = melted_2['Metric'].str.capitalize()
sns.barplot(data = melted_2, x = 'value', y = 'Metric', hue = 'Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Default Models Scaled Data')

# decision tree from round 1 (non-scaled data), logistic regression from round 2 (scaled data)
best_clf_round_1 = round_1_results[round_1_results['model']=='Decision Tree Classification'].reset_index(drop=True)
best_clf_round_2 = round_2_results[round_2_results['model']=='Logistic Regression'].reset_index(drop=True)
top_models_round_1 = pd.concat([best_clf_round_1, best_clf_round_2], axis=0, ignore_index=True)
top_models_round_1['model'] = pd.Series(['Decision Tree - Non-Scaled Data', 'Logistic Regression - Scaled Data'])

top_models = top_models_round_1.melt(id_vars = ['model'],
                                             value_vars = ['accuracy', 'precision', 'recall', 'f1'],
                                             var_name = 'metric')
top_models = top_models.rename(columns={'model':'Model', 'metric': 'Metric'})
top_models['Metric'] = top_models['Metric'].str.capitalize()

sns.barplot(data = top_models, x = 'value', y = 'Metric', hue = 'Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Top Performing Default Models')


'''
Final Round: Hypertuning the Top Default Models
'''
# result storage
result_columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Model']
hypertuned_results = pd.DataFrame(columns=result_columns)

# decision tree parameters to run through
tree_parameters = {'criterion': ('gini', 'entropy', 'log_loss'),
              'splitter': ('best', 'random'),
              'max_depth': (None, 2, 4, 6, 8, 10),
              'max_features': (None, 'sqrt', 'log2'),
              'class_weight': (None, 'balanced')}
# create gridsearchcv object
tree_clf_hyper = GridSearchCV(DecisionTreeClassifier(), tree_parameters)
# train the model
tree_clf_hyper.fit(X_train, y_train)

# results
tree_clf_hyper_results = pd.DataFrame(tree_clf_hyper.cv_results_)
# export results to csv - commented out post script run
# tree_clf_hyper_results.to_csv('../data/hotspot_hyper_tree_results.csv', index = False)


tree_clf_hyper.best_params_
'''
{'class_weight': None,
 'criterion': 'entropy',
 'max_depth': 6,
 'max_features': None,
 'splitter': 'best'}
'''

# decision tree with best params
clf = DecisionTreeClassifier(**tree_clf_hyper.best_params_).fit(X_train, y_train)
# prediction
clf_y_pred = clf.predict(X_test)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'Accuracy': clf_accuracy,
               'Precision': clf_precision,
               'Recall': clf_recall,
               'F1': clf_f1,
               'Model': 'Hypertuned Decision Tree - Non-Scaled Data'}
clf_results = pd.DataFrame([clf_results])
hypertuned_results = pd.concat([hypertuned_results, clf_results], ignore_index=True)

# logistic regression
# decision tree parameters to run through
logistic_parameters = {'max_iter': [10000],
                       'penalty': (None, 'l2', 'l1', 'elasticnet'),
                       'class_weight': (None, 'balanced'),
                       'solver': ('lbfgs', 'saga'),
                       'multi_class': ('auto', 'multinomial')}
# create gridsearchcv object
logistic_clf_hyper = GridSearchCV(LogisticRegression(), logistic_parameters)
# train the model
logistic_clf_hyper.fit(X_train_normal, y_train)

# results
logistic_clf_hyper_results = pd.DataFrame(tree_clf_hyper.cv_results_)
# export results to csv - commented out post script run
# logistic_clf_hyper_results.to_csv('../data/hotspot_hyper_logistic_results.csv', index = False)

logistic_clf_hyper.best_params_
'''
{'class_weight': None,
 'max_iter': 10000,
 'multi_class': 'auto',
 'penalty': 'l1',
 'solver': 'saga'}
'''

# decision tree with best params
clf = LogisticRegression(**logistic_clf_hyper.best_params_).fit(X_train_normal, y_train)
# prediction
clf_y_pred = clf.predict(X_test_normal)
# results (will set average to weighted for multiclass vs binary)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
clf_precision = precision_score(y_test, clf_y_pred, average='weighted')
clf_recall = recall_score(y_test, clf_y_pred, average='weighted')
clf_f1 = f1_score(y_test, clf_y_pred, average='weighted')
clf_results = {'Accuracy': clf_accuracy,
               'Precision': clf_precision,
               'Recall': clf_recall,
               'F1': clf_f1,
               'Model': 'Hypertuned Logistic Regression - Scaled Data'}
clf_results = pd.DataFrame([clf_results])
hypertuned_results = pd.concat([hypertuned_results, clf_results], ignore_index=True)

# save results to csv - commented out post script run
# hypertuned_results.to_csv('../data/best_hyper_results.csv', index = False)

# best results visualization
melted_best = hypertuned_results.melt(id_vars = ['Model'],
                                             value_vars = ['Accuracy', 'Precision', 'Recall', 'F1'],
                                             var_name = 'Metric')

sns.barplot(data = melted_best, x = 'value', y = 'Metric', hue = 'Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Best Hypertuned Models')

'''
Saving the Model
'''
# let's export this file as a pkl file
with open('../models/hotspot_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
