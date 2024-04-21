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
major_city                      0
national_park_vistors           0
ski_resort                      0
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
    - encode region (4 categories, *probably* important)

Creating hotspot criteria:
    - based on concentration -> per capita
    - create ranking system based on concentration
    - this will need an immediate removal of the rows requiring census data
'''

# remove income column
model_df = city_df.drop(['2021 median income'], axis=1).copy()

# remove rows without census data
model_df.dropna(subset=['total population'], axis=0, inplace=True)

# check nulls
# model_df.isnull().sum()

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
major_city                   0
national_park_vistors        0
ski_resort                   0
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
'''

# create new column with brewery concentration per capita (per 1000)
model_df['brewery_concentration'] = 1000*(model_df['city_brewery_count'] / model_df['total population'])
# create ranks: 1 - 10
# labels as false and the + 1 return ints instead of category type data
model_df['ranked'] = pd.qcut(model_df['brewery_concentration'], q=10, labels=False) + 1

# drop brewery concentration (collinearity with ranking)
# drop city and state
model_df.drop(['city','state','brewery_concentration'], axis=1, inplace=True)

# REMEMBER: a misclassification could mean either too low or too high concentration

# encode regions
model_df = pd.get_dummies(model_df, columns=['region'], drop_first=True)

# check datatypes
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
Decent results from decision tree
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
not great performance and requires deviation from default to even produce results
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
not great performance and requires array workaround to fit and predict
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
not great performance
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
not great performance
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
0  0.896341  0.899739  0.896341  0.896276  Decision Tree Classification
1  0.282012  0.255597  0.282012  0.261475           Logistic Regression
2  0.518293  0.546575  0.518293  0.523997                           KNN
3  0.160061  0.093318  0.160061  0.093528        Support Vector Machine
4  0.155488  0.080463  0.155488  0.086722                   Naive Bayes
5   0.22561  0.229622   0.22561  0.225666  Linear Discriminant Analysis
'''

# export round 1 results for website
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
Abysmal performance compared to non-scaled data
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
slight improvement over non-scaled data
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
We don't need to use array versions with normalized data, but still not great
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
still not great performance
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
nope, just nope
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
not great
'''

'''
Results from Round 2:
    
   accuracy precision    recall        f1                         model
0   0.27439  0.280048   0.27439   0.22599  Decision Tree Classification
1  0.314024  0.316085  0.314024  0.295649           Logistic Regression
2  0.224085  0.225138  0.224085  0.217315                           KNN
3  0.219512   0.22855  0.219512  0.217283        Support Vector Machine
4  0.096037  0.125984  0.096037  0.031613                   Naive Bayes
5  0.224085  0.230842  0.224085  0.224684  Linear Discriminant Analysis
'''

# export round 1 results for website
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

# round 2
melted_2 = round_2_results.melt(id_vars = ['model'],
                                             value_vars = ['accuracy', 'precision', 'recall', 'f1'],
                                             var_name = 'metric')
melted_2 = melted_2.rename(columns={'model':'Model', 'metric': 'Metric'})
melted_2['Metric'] = melted_2['Metric'].str.capitalize()
sns.barplot(data = melted_2, x = 'value', y = 'Metric', hue = 'Model')

# just decision tree from round 1
decision_tree_1 = round_1_results[round_1_results['model']=='Decision Tree Classification']
decision_tree_1.drop(['model'], axis=1, inplace=True)
decision_tree_1.columns = ['Accuracy', 'Precision', 'Recall', 'F1']
main_color = sns.color_palette()[0]
sns.barplot(data = decision_tree_1, color = main_color, orient='h')
plt.xlim(0,1)
plt.title('Default Decision Tree Results')


'''
Final Round: Hypertuning the Decision Tree
'''

# parameters to run through
parameters = {'criterion': ('gini', 'entropy', 'log_loss'),
              'splitter': ('best', 'random'),
              'max_depth': (None, 2, 4, 6, 8, 10),
              'max_features': (None, 'sqrt', 'log2'),
              'class_weight': (None, 'balanced')}
# create gridsearchcv object
clf_hyper = GridSearchCV(DecisionTreeClassifier(), parameters)
# train the model
clf_hyper.fit(X_train, y_train)

# results
clf_hyper_results = pd.DataFrame(clf_hyper.cv_results_)
# export results to csv
# clf_hyper_results.to_csv('../data/hotspot_hypertuning.csv', index = False)

clf_hyper.best_params_
'''
{'class_weight': None,
 'criterion': 'entropy',
 'max_depth': None,
 'max_features': None,
 'splitter': 'best'}
'''

clf_hyper.scorer_

# decision tree with best params
clf = DecisionTreeClassifier(**clf_hyper.best_params_).fit(X_train, y_train)
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
               'Version': 'Hypertuned'}
hyper_tuned_df = pd.DataFrame([clf_results])

# compare results to default decision tree
decision_tree_compare = decision_tree_1.copy()
decision_tree_compare['Version'] = 'Default'
decision_tree_compare = pd.concat([decision_tree_compare, hyper_tuned_df], ignore_index=True)

# get ready for plotting
# round hypertune
melted_hyper = decision_tree_compare.melt(id_vars = ['Version'],
                                          value_vars = ['Accuracy', 'Precision', 'Recall', 'F1'],
                                          var_name = 'Metric')
sns.barplot(data = melted_hyper, x = 'value', y = 'Metric', hue = 'Version')
plt.title('Decision Tree Versions')

# zoom in
sns.barplot(data = melted_hyper, x = 'value', y = 'Metric', hue = 'Version')
plt.xlim([0.89, 0.91])
plt.title('Decision Tree Versions - Zoom')


'''
Applying the Model
'''
# before we proceed, let's export this file as a pkl file
with open('../model/hotspot_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# let's say we want to see how a city ranks among breweries according to our model
# grab random entries from our test set
random_test_features = X_test.sample(10)
# grab their corresponding hotspot rankings
random_test_rankings = y_test.loc[random_test_features.index]

# combine for website
random_testing = pd.concat([random_test_features, random_test_rankings], axis=1)

# send to data for website
# random_testing.to_csv('../data/hotspot_random_testing.csv', index = False)

# plug into our model
random_pred = clf.predict(random_test_features)

# reset indices
actual = random_test_rankings.reset_index(drop=True)
actual.columns = ['Actual']
predicted = pd.DataFrame(random_pred, columns=['Predicted']).reset_index(drop=True)

# let's compare manually
random_compare = pd.concat([actual, predicted], axis=1)
# random_testing.to_csv('../data/hotspot_random_comparison.csv', index = False)
