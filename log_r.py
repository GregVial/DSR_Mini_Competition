#svm

import pandas as pd
import numpy as np

# input = pd.read_csv("data/sanitized.csv")
# input.set_index("Grant.Application.ID",drop=True,inplace=True)
# input.index.name=None
# input.isnull().any()
#
# # COPIED FROM NOTEBOOK:
# from sklearn.model_selection import train_test_split, GridSearchCV
#
# input.fillna(value=0,inplace=True)
# X = input.ix[:,1:]
# y = input.ix[:,0]
# X.shape
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=131078)
# X.shape

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_train.drop('Grant.Application.ID', axis=1, inplace=True)
y_train = y_train.values.reshape(-1)

y_test = pd.read_csv("data/y_test.csv")
test_index = y_test.loc[:,'Grant.Application.ID']
y_test.drop('Grant.Application.ID', axis=1, inplace=True)
y_test = y_test.values.reshape(-1)

# COPY FROM BELOW INTO NOTEBOOK:

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
# 'linear', 'poly', 'rbf', 'sigmoid'

if __name__ == "__main__":

    # LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    log_r = LogisticRegression(penalty='l2', C=1.0, solver = 'lbfgs', n_jobs=-1, max_iter=1000, verbose=1)
    # log_r.fit(X_train_std, y_train)

    C = [0.1072]# optimized
    print('Starting LR Grid Search')
    lr_gs = GridSearchCV(log_r, param_grid={'C':C})
    lr_gs.fit(X_train_std, y_train)
    log_r = lr_gs.best_estimator_
    log_r_params = lr_gs.best_params_
    print('Best parameters were '+ str(log_r_params))

    y_pred_lr = log_r.predict(X_test_std)
    y_pred_lr_probs = log_r.predict_proba(X_test_std)

    y_pred_lr_probs = pd.DataFrame(y_pred_lr_probs, index=test_index)

    probs_target = y_pred_lr_probs.loc[:,1]

    lr_accuracy = np.mean(y_pred_lr == y_test)
    lr_roc = roc_auc_score(y_test, y_pred_lr)

    print('LR Accuracy is ' + str(lr_accuracy))
    print('LR Binary ROC Area Under Curve is ' + str(lr_roc))

    lr_roc_prob = roc_auc_score(y_test, probs_target)
    print('LR Prob ROC Area Under Curve is ' + str(lr_roc_prob))

    y_pred_lr_probs.to_csv('result/result log.csv')
