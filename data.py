import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

raw = pd.read_csv('data/unimelb_training.csv')

sponsor_code = raw.loc[:, 'Sponsor.Code']

def summary(series):
    num_miss = series.isnull().sum()
    not_miss = series.notnull().sum()
    count = len(series)
    pct_missing = num_miss / count
    return pct_missing

pct_missing = summary(sponsor_code)

encoder = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
encoder.fit(['>10 to 15', '>5 to 10', '>=0 to 5', 'Less than 0', 'more than 15',
       'nan'])

people = pd.DataFrame()
time_at_uni = pd.DataFrame()
for col in raw.columns:
    col_list = list(raw.columns) # list so I can use index later

    if col[:10] == 'Person.ID.':
        col_start = col_list.index(col)
        col_end = col_start + 15 # hard coded
        person = raw.iloc[:,col_start:col_end]

        # including our ID and status
        person.loc[:,'Grant.Application.ID'] = raw.loc[:,'Grant.Application.ID']
        person.loc[:,'Grant.Status'] = raw.loc[:,'Grant.Status']
        
        col_names = list(person.columns)
        col_names = [col.strip('0123456789') for col in col_names] # removes all numbers
        person.columns = col_names
        print(len(person))

        # dealing with number of years at uni
        original = person.loc[:,'No..of.Years.in.Uni.at.Time.of.Grant.'].astype(str)
        transformed = encoder.transform(original)
        transformed = pd.DataFrame(transformed, columns=encoder.classes_)

        person.drop('No..of.Years.in.Uni.at.Time.of.Grant.', axis=1, inplace=True)




        # alternative method using averages for each cat
        # average_ages = {'Less than 0':0, '>=0 to 5':2.5, '>5 to 10':7.5, '>10 to 15':12.5, 'more than 15':20, 'nan':0 }
        # for index, row in transformed.iterrows():
        #     for key, value in average_ages.items():
        #         row[key] = float(value) * row[key]
        #
        #     time_at_uni.loc[index,'Time at Uni'] = row.sum()

        person = pd.concat([person, transformed], axis=1)

        people = people.append(person, ignore_index=True)

# summaries for team
num_ppl = len(people) / len(person) # should be 15!
people.index = people.loc[:,'Grant.Application.ID']

num_ppl = people.groupby(['Grant.Application.ID'])['Person.ID.'].count() # number of people with ids

num_birth_years = people.groupby(['Grant.Application.ID'])['Year.of.Birth.'].count()
total_birth_year = people.groupby(['Grant.Application.ID'])['Year.of.Birth.'].sum()
avg_birth_year = total_birth_year / num_birth_years

num_successful = people.groupby(['Grant.Application.ID'])['Number.of.Successful.Grant.'].sum()
num_unsuccessful = people.groupby(['Grant.Application.ID'])['Number.of.Unsuccessful.Grant.'].sum()

# number of phds - TODO could be improved
num_phd  = people.groupby(['Grant.Application.ID'])['With.PHD.'].count() # assuming only value ever entered is yes # include

people_summary = pd.DataFrame()
ppl_groupby = people.groupby(['Person.ID.'])
# using groupbys
people_summary.loc[:,'Num applications'] = ppl_groupby['Grant.Application.ID'].count()
people_summary.loc[:,'Num grant status'] = ppl_groupby['Grant.Status'].sum()
people_summary.loc[:,'Num successful'] = ppl_groupby['Number.of.Successful.Grant.'].sum()
people_summary.loc[:,'Num unsuccessful'] = ppl_groupby['Number.of.Unsuccessful.Grant.'].sum()

# using dataframe to create more statistics
people_summary.loc[:,'Ratio succ/unscc'] = people_summary.loc[:,'Num successful'] / (people_summary.loc[:,'Num unsuccessful'] + people_summary.loc[:,'Num successful'])
people_summary.loc[:,'Ratio target/samples'] = people_summary.loc[:,'Num grant status'] / (people_summary.loc[:,'Num grant status'] + people_summary.loc[:,'Num applications'])
people_summary.sort(columns=['Ratio succ/unscc'], ascending=False, inplace=True)

#
visual = pd.DataFrame()
visual.loc[:,'Ratio succ/unscc'] = people_summary.loc[:,'Ratio succ/unscc']
visual.loc[:,'Ratio target/samples'] = people_summary.loc[:,'Ratio target/samples']

import matplotlib.pyplot as plt  #sets up plotting under plt
import seaborn as sns

visual.sort(columns=['Ratio target/samples'], ascending=False, inplace=True)

plot_data = visual.iloc[:1000,:]

single = sns.lmplot('Ratio succ/unscc','Ratio target/samples',data=plot_data, fit_reg=False)

grid = sns.PairGrid(visual)
grid.map_upper(plt.scatter)
grid.map_lower(plt.scatter)
