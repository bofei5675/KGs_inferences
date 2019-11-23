import pandas as pd
import numpy as np
import os
df = pd.read_csv('./data/tanimoto_info_PCA50.csv')
data_path = '/scratch/bz1030/relationPrediction/data/deepddi'
print(df.head())
print(df.shape)
entity_ids = pd.read_csv(data_path + '/entity2id.txt', sep='\t', header=None)
entity_ids.columns = ['drug_id','id']
col_names = df.columns.tolist()
col_names[0] = 'drug_id'

df.columns = col_names

df = pd.merge(df, entity_ids, left_on='drug_id', right_on='drug_id', how='inner')
print(df.head())
selected_cols = ['id'] + ['PC_{}'.format(i) for i in range(1,51)]
df = df[selected_cols]
print(df.head())
df.sort_values('id', inplace=True, ascending=True)
print(df.head())
df.set_index('id', inplace=True)
df.to_csv('entity2vec.txt', sep='\t', index=False, header=None)
