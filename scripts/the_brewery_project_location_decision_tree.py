# -*- coding: utf-8 -*-
"""the_brewery_project_location_decision_tree.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y0tWxyTQZjXK9PLx5FRazsse2q1KrXnT
"""

#Goal: to build a focused decision tree that is usable by the lay audience/ potential
#brewery owners.

#Load packages
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn import tree
import graphviz
import pickle
#Call Data:
city_df = pd.read_csv('../data/city_level.csv')

#Reduce Dataframe to required columns and transform
model_df = city_df.drop(["city", "state","city_brewery_count",
                                 "2021 median income","brewery_concentration",
                                 "per_capita_ranked","national_park_vistors"],axis = 1)

model_df = model_df[model_df["total population"]>0] #Remove null populations
model_df = pd.get_dummies(model_df, columns=['region'], drop_first=True)
model_df = model_df.astype(int)
model_df.rename({'custom_ranked':'ranked'}, inplace=True, axis=1)

model_df.info()

#Determine test and training sets
X  = model_df.drop(["ranked"],axis = 1)
y = model_df["ranked"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 71)

# decision tree parameters to run through
tree_parameters = {'criterion': ('gini', 'entropy', 'log_loss'),
              'splitter': ('best', 'random'),
              'max_depth': (3,4),
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
#{'class_weight': None,
# 'criterion': 'gini',
# 'max_depth': 5,
# 'max_features': None,
# 'splitter': 'best'}

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
               'Model': 'Decision Tree'}
clf_results = pd.DataFrame([clf_results])
clf_results

#x[2] refers to total count of brewpubs
#x[7] refers to total microbreweries
#x[20] refers to % of population age 21+
#x[23] refers to % of population identifying as black
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph

# let's export this file as a pkl file
with open('../models/location_decision_tree_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
