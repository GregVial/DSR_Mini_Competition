#svm

import pandas as pd
import numpy as np

input = pd.read_csv("data/sanitized.csv")
input.set_index("Grant.Application.ID",drop=True,inplace=True)
input.index.name=None
input.isnull().any()

# COPIED FROM NOTEBOOK:
from sklearn.model_selection import train_test_split, GridSearchCV

input.fillna(value=0,inplace=True)
X = input.ix[:,1:]
y = input.ix[:,0]
X.shape
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=131078)
X.shape

# COPY FROM BELOW INTO NOTEBOOK:
from sklearn import svm
from sklearn.metrics import roc_auc_score

# two different sklearn svm models:
s_vec = svm.SVC(C=1, kernel='linear')
# s_vec = svm.NuSVC(nu=0.5, kernel = 'linear')

from sklearn.model_selection import GridSearchCV
# 'linear', 'poly', 'rbf', 'sigmoid'

# NOT DOING CROSS VALIDATION
print('Starting SVM Grid Search')
svm_gs = GridSearchCV(s_vec, param_grid={'C':[0.01, 0.1, 1, 10],'kernel':['linear', 'rbf']})
svm_gs.fit(X_train, y_train)

s_vec = svm_gs.best_estimator_
y_pred_svm = s_vec.predict(X_test)
svm_accuracy = np.mean(y_pred_svm == y_test)
svm_roc = roc_auc_score(y_test, y_pred_svm)

print('SVM Accuracy is ' + str(svm_accuracy))
print('SVM ROC Area Under Curve is ' + str(svm_roc))

from sklearn.metrics import confusion_matrix
svm_conmat = np.array(confusion_matrix(y_test, y_pred))
svm_confusion = pd.DataFrame(svm_conmat, index=['not_granted', 'granted'],columns=['predicted_not_granted', 'predicted_granted'])

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# need to standardize
scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

log_r = LogisticRegression(penalty='l2', C=1.0, solver = 'sag', n_jobs=-1)
log_r.fit(X_train, y_train)
print('Starting LR Grid Search')
lr_gs = GridSearchCV(log_r, param_grid={'C':[0.01, 0.1, 1, 10]})
lr_gs.fit(X_train_std, y_train)
log_r = lr_gs.best_estimator_

y_pred_lr = log_r.predict(X_test_std) #
lr_accuracy = np.mean(y_pred_lr == y_test)
lr_roc = roc_auc_score(y_test, y_pred_lr)

print('LR Accuracy is ' + str(lr_accuracy))
print('LR ROC Area Under Curve is ' + str(lr_roc))
