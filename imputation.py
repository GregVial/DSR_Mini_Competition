import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

raw = pd.read_csv('data/unimelb_training.csv')
#
# sponsor_code = raw.loc[:,'Sponsor.Code']
# grant_cat = raw.loc[:,'Grant.Category.Code']
# value_band = raw.loc[:,'Contract.Value.Band...see.note.A']

df = raw.ix[:,2:5]

df.dropna(axis=0, inplace=True)
na_dummy_bool=True
df = pd.concat([df,pd.get_dummies(df["Sponsor.Code"],prefix="Sponsor", dummy_na=na_dummy_bool)],axis=1)
df.drop("Sponsor.Code",axis=1,inplace=True)

df = pd.concat([df,pd.get_dummies(df['Grant.Category.Code'],prefix="Sponsor", dummy_na=na_dummy_bool)],axis=1)
df.drop('Grant.Category.Code',axis=1,inplace=True)

# df = pd.concat([data,pd.get_dummies(data['Contract.Value.Band...see.note.A'],prefix="Sponsor", dummy_na=na_dummy_bool)],axis=1)
# df.drop("Contract.Value.Band...see.note.A",axis=1,inplace=True)

X_train = df.ix[:,1:].astype(str)
y_train = df.ix[:,0].astype(str)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)
params={
    "n_estimators":[500],
    "min_samples_split":[5],
    "min_samples_leaf":[4]
}
if __name__ == "__main__":
    gbc.fit(X_train,y_train)
