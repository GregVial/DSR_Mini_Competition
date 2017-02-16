import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

raw = pd.read_csv('data/unimelb_training.csv')
#
# sponsor_code = raw.loc[:,'Sponsor.Code']
# grant_cat = raw.loc[:,'Grant.Category.Code']
# value_band = raw.loc[:,'Contract.Value.Band...see.note.A']

df = raw.ix[:,2:5]

df.dropna(axis=0, subset=["Sponsor.Code","Grant.Category.Code"],inplace=True)

na_dummy_bool=True
df = pd.concat([df,pd.get_dummies(df["Sponsor.Code"],prefix="Sponsor", dummy_na=na_dummy_bool)],axis=1)
df = pd.concat([df,pd.get_dummies(df['Grant.Category.Code'],prefix="Grant", dummy_na=na_dummy_bool)],axis=1)

df.drop("Sponsor.Code",axis=1,inplace=True)
df.drop('Grant.Category.Code',axis=1,inplace=True)
#df.drop('Contract.Value.Band...see.note.A',axis=1,inplace=True)


# df = pd.concat([data,pd.get_dummies(data['Contract.Value.Band...see.note.A'],prefix="Sponsor", dummy_na=na_dummy_bool)],axis=1)
# df.drop("Contract.Value.Band...see.note.A",axis=1,inplace=True)

X_train,X_test,y_train,y_test = train_test_split(df.ix[:,1:].astype(str),df.ix[:,0].astype(str))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)

#if __name__ =="__main__":
#    gbc.fit(X_train,y_train)

gbc.fit(X_train,y_train)

y_pred = gbc.predict(X_test)
print(np.mean(y_test==y_pred))