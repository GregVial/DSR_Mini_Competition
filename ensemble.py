import pandas as pd
import numpy as np

log_r = pd.read_csv("result/result log.csv")
log_r.columns = ['Grant.Application.ID','log R False', 'log R True']
# RF = pd.read_csv("data/result RF.csv")
# RF_boosted = pd.read_csv("data/result RF boost.csv")

y_test = pd.read_csv("data/y_test.csv")
y_test.set_index(keys = ['Grant.Application.ID'], drop=True)

data = [y_test, log_r] # log_r, RF, RF_boosted,

data = [df.set_index(keys = ['Grant.Application.ID'], drop=True) for df in data]
all_pred = pd.concat(data, axis=1)

# ensemble

all_pred['log R pred'] = all_pred['log R True'] > 0.5
