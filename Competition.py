## load libraries
#measure time code execution
from datetime import datetime
start = datetime.now()

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sweetviz as sv
# Preprocessing
from sklearn.preprocessing import LabelEncoder
# Modeling
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Model selection
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score

from sklearn import metrics
from sklearn.metrics import confusion_matrix

#pipelines
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, plot_importance, plot_tree

from sklearn.metrics import (precision_recall_curve, confusion_matrix, accuracy_score, 
                                       roc_curve, auc, f1_score, roc_auc_score, precision_score, 
                                       recall_score)
import seaborn as sns
from skopt import BayesSearchCV
from sklearn.naive_bayes import GaussianNB

import sys
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

#################################
#Import dataset
#################################

dataset = pd.read_csv("hotels_train.csv", sep = ",")
dataset.info()

#################################
# One-hot encoding
#################################

# First: we create two data sets for numeric and non-numeric data
numerical = dataset.select_dtypes(exclude=['object'])
categorical = dataset.select_dtypes(include=['object'])

# Second: One-hot encode the non-numeric columns

onehot = pd.get_dummies(categorical)

# third: Third: Union the one-hot encoded columns to the numeric ones

df = pd.concat([numerical, onehot], axis=1)


X = df.loc[ : , df.columns != 'is_canceled'] # all columns except target variable
y = df[['is_canceled']] # target variable


#################################
# Data Splitting 
#################################

X_train, test_X, y_train, test_y = train_test_split(X, y, test_size=.3, random_state=123)
X_eval, X_test, y_eval, y_test = train_test_split(test_X, test_y, test_size=.5, random_state=123)

#################################
# Undersampling/Oversampling Train set
#################################
round(y_train['is_canceled'].value_counts()*100/len(y_train['is_canceled']), 2)

'''
0    57.77
1    42.23
'''
# Pretty balanced dataset.

#################################
# Pipeline definition
#################################

# Imputer to fulfill empty values
imp_constant = SimpleImputer(strategy='constant', fill_value='unknown')
imp_mean = SimpleImputer(strategy='mean')
# Scaling numerical data
scaler = StandardScaler()
# Columns works
num_cols = make_column_selector(dtype_include='number')
cat_cols = make_column_selector(dtype_exclude='number')
#Making one preprocessor
preprocessor = make_column_transformer(
    (make_pipeline(imp_mean, scaler), num_cols),
    (make_pipeline(imp_constant), cat_cols),
    remainder='passthrough')


#################################
# Pipeline application
#################################

for v in dataset.isnull().sum():
    print(v)
    
# Models definition,list
models = [LogisticRegression(max_iter=1000), 
          SVC(), 
          RandomForestClassifier(), 
          GaussianNB(), 
          XGBClassifier(use_label_encoder=False, eval_metric="logloss")]
# we are going to train the following models:
model_names = ['logistic_regression', 'support_vector_classifier', 'random_forest', 'naive_bayess', 'xgb']


y_train=np.ravel(y_train)

# Loop to train and estimate the models from the models deifnition,list
results = pd.DataFrame()

for model, name in zip(models, model_names):
    print(f'Evaluating: {name}')
    pipe = make_pipeline(preprocessor, model)
    
    results.loc[name, 'train'] = cross_val_score(pipe, X_train, y_train, cv=3, scoring='roc_auc').mean()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_eval)
    
    results.loc[name, 'test_accuracy'] = accuracy_score(y_eval, y_pred)
    results.loc[name, 'test_roc_auc'] = roc_auc_score(y_eval, y_pred)
    results.loc[name, 'test_f1'] = f1_score(y_eval, y_pred)
    results.loc[name, 'test_precision'] = precision_score(y_eval, y_pred)
    results.loc[name, 'test_recall'] = recall_score(y_eval, y_pred)
    
# best models results    
print(results)

#################################
# Pipeline results
#################################

# Plot to show models results

plt.figure(figsize=(7,7))
sns.heatmap(results, annot=True, square=True, cmap="viridis", cbar=False,
            fmt='0.3f', annot_kws={'size':14}, linewidths=1)
plt.title('Particular models score, default parameters\n', fontdict={'fontsize':18})
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5], ['Cross\nValidation', 'Accuracy', 'ROC\nAUC', 'F1', 'Precision', 'Recall'], fontsize=12)
plt.yticks([0.5,1.5,2.5,3.5,4.5], ['Logistic\nRegression', 'Support Vector\nClassifier', 'Random\nForest', 'Naive\nBayess', 'XGBoost'],
          fontsize=12)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12,7))
plot = results.T.iloc[1:].plot(ax=ax, kind='barh', width=.82, cmap='Set3', edgecolor='k')

for p in plot.patches:
    plot.annotate(
        text='{:.3}'.format(p.get_width()), 
        xy=(p.get_width(), p.get_y() + p.get_height()/2),
        ha='center', 
        va='center', 
        xytext=(-20, 0), 
        textcoords='offset points', 
        color='k',
        fontsize=10)
plt.title('Estimation of the particular models with default parameters', fontdict={'fontsize':16})
plt.yticks([0,1,2,3,4], ['Accuracy', 'ROC AUC', 'F1', 'Precision', 'Recall'], rotation=0)
plt.xlim(left=.3, right=1.05)
plt.xlabel('Score', fontsize=15)
plt.tight_layout()
plt.show()

# In terms of accuracy Random Forest and XGBoost both allocate the best results when trained with default hyperparameters. 
# Therefore we will tune both and compare them.

#################################
# Hyperparameter tuning (XGBOoost)
#################################

# Example of a grid search for a XGBoost 

# Model hyperparameters
grid_params = {
    'learning_rate': [0.01, 0.05, .1, .3, .6],
    'max_depth': np.arange(2, 12, 1),
    'min_child_weight' : [1, 3, 5, 7],
    'reg_alpha': np.arange(0,10,2),
    'reg_lambda': np.arange(0,10,1),
    'gamma': [0.5, 1, 1.5, 2, 5],
    'n_estimators': np.arange(100, 600, 100),
    'random_state': [42]
}

# Parameters, which are allowing us to stop learning, in case of overfitting after 15 rounds (value not getting better)
fit_params = {
    'early_stopping_rounds' : 15, 
    'eval_set' : [(X_eval, y_eval)],
    'verbose': 0
}

# Bayesian optimisation
opt = BayesSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    search_spaces=grid_params,
    fit_params=fit_params,
    n_iter=5,
    cv=10,
    random_state=123
)

opt = opt.fit(X_train, y_train)
# best hyperparameters
opt.best_params_
# train model with best hyperparameters
xgb = XGBClassifier(use_label_encoder=False,  **opt.best_params_)
xgb.fit(X_train, 
        y_train, 
        early_stopping_rounds=15, 
        eval_set=[(X_eval, y_eval)], 
        eval_metric='logloss',
        verbose=0)

#################################
# Model prediction
#################################
# XGBoost predictions
y_pred_XGB = xgb.predict(X_test)

#################################
# Model performance
#################################
# XGBoost confussion matrix
plt.figure(figsize=(4,2))
sns.heatmap(data=confusion_matrix(y_test, y_pred_XGB), annot=True, cbar=False, fmt="d", linewidths=2, 
            annot_kws={'fontsize':17}, xticklabels=['Positive', 'Negative'], 
            yticklabels=['Positive', 'Negative'], cmap="cividis")
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, labeltop=True)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


print(f'Accuracy: {accuracy_score(y_test, y_pred_XGB)}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_XGB)}')
'''
Accuracy: 0.8788888888888889
ROC AUC: 0.8690335722974291
'''

# XGBoost variable importance
xgb.get_booster().feature_names = list(preprocessor.transformers_[0][2])

f_imp = pd.DataFrame(zip(xgb.get_booster().feature_names, xgb.feature_importances_), 
                     columns=['feature', 'importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(9,6))
plot = sns.barplot(data=f_imp.head(10), y='feature', x='importance', palette='summer_r', edgecolor='k')
for p in plot.patches[0:1]:
    plot.annotate(
        
        text='{:.3f}'.format(p.get_width()), 
        xy=(p.get_width(), p.get_y() + p.get_height()/2),
        ha='center', 
        va='center', 
        xytext=(-22, 0), 
        textcoords='offset points', 
        color='k',
        fontsize=13)
for p in plot.patches[1:]:
    plot.annotate(
        text='{:.3f}'.format(p.get_width()), 
        xy=(p.get_width(), p.get_y() + p.get_height()/2),
        ha='center', 
        va='center', 
        xytext=(24, 0), 
        textcoords='offset points', 
        color='k',
        fontsize=13)
plt.title('10 most important features of XGBoost', fontdict={'fontsize':16})

plt.xlabel('Variable Importance', fontsize=15)
plt.ylabel('Variables', fontsize=15)
plt.tight_layout()
plt.show()

#################################
# Hyperparameter tuning (Random Forest)
#################################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 12, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Model hyperparameters
grid_params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(grid_params)


# Example of a grid search for a Random Forest

# Bayesian optimisation
opt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state = 42),
    search_spaces=grid_params,
    n_iter=5,
    cv=10,
    random_state=123
)

opt = opt.fit(X_train, y_train)
# best hyperparameters
opt.best_params_
# train model with best hyperparameters
rf = RandomForestClassifier(**opt.best_params_)
rf.fit(X_train, y_train)


#################################
# Model prediction
#################################
# Random Forest predictions
y_pred_rf = rf.predict(X_test)

#################################
# Model performance
#################################
# Random Forest Confussion matrix
plt.figure(figsize=(4,2))
sns.heatmap(data=confusion_matrix(y_test, y_pred_rf), annot=True, cbar=False, fmt="d", linewidths=2, 
            annot_kws={'fontsize':17}, xticklabels=['Positive', 'Negative'], 
            yticklabels=['Positive', 'Negative'], cmap="cividis")
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, labeltop=True)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_rf)}')

'''
Accuracy: 0.8458888888888889
ROC AUC: 0.8242037924228137
'''

end = datetime.now()
execution_time = end - start
print(f'Execution Time: {execution_time}')

#################################
# Model testing
#################################

test = pd.read_csv("hotels_test.csv", sep = ",")

# First: we create two data sets for numeric and non-numeric data
numerical = test.select_dtypes(exclude=['object'])
categorical = test.select_dtypes(include=['object'])

# Second: One-hot encode the non-numeric columns

onehot = pd.get_dummies(categorical)

# third: Third: Union the one-hot encoded columns to the numeric ones
X_test_set = pd.concat([numerical, onehot], axis=1)

for i in [i for i in X_train.columns if i not in X_test_set.columns]:
    X_test_set[i] = 0

# XGBoost prediction
pred_XGB = xgb.predict(X_test_set)

# We stay with the XGBoost model
pd.DataFrame(pred_XGB).to_csv('predictions.csv', index = False)
