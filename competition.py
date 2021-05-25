## load libraries
#measure time code execution
from datetime import datetime
start = datetime.now()

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import LabelEncoder

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


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import (precision_recall_curve, confusion_matrix, accuracy_score, 
                                       roc_curve, auc, f1_score, roc_auc_score, precision_score, 
                                       recall_score)
import sys
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

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

# Our one-hot encoded test set doesn't have as many variables as the training set. The make them zeros

for i in [i for i in X_train.columns if i not in X_test_set.columns]:
    X_test_set[i] = 0

X_test_set.info()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

def Bagging_func(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, final_test=X_test_set):

    tree = DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=None,
                                  random_state=1)

    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=500, 
                            max_samples=1.0, 
                            max_features=1.0, 
                            bootstrap=True, 
                            bootstrap_features=False, 
                            n_jobs=1, 
                            random_state=1)

    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)

    from sklearn.metrics import confusion_matrix

    y_pred = bag.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.tight_layout()
    #plt.savefig('images/06_09.png', dpi=300)
    plt.show()

    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, pos_label=0))
    print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, pos_label=0))
    
    test_predicted = bag.predict(final_test)
    return(test_predicted)


predictions = Bagging_func()

pd.DataFrame(predictions).to_csv('predictions.csv', index = False)

end = datetime.now()
start - end
