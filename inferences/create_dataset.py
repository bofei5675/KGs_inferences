import os
import pandas as pd

dataset_name = './drugbank_100'

if not os.path.exists(dataset_name):
    os.makedirs(dataset_name)

df = pd.read_csv('./drugbank_data_subset100.txt', sep='\t', header=None)
num_samples = df.shape[0]

entities = df.iloc[:, 0].tolist() + df.iloc[:, 2].tolist()
entities = set(entities)
entities = pd.DataFrame(entities)
entities.to_csv(dataset_name + '/entities.txt', sep='\t',index=False, header=None)

relations = df.iloc[:, 1].tolist()
relations = set(relations)
relations = pd.DataFrame(relations)
relations.to_csv(dataset_name  + '/relations.txt', sep='\t',index=False, header=None)


train_size = int(0.7 * num_samples)
val_size = int(0.1 * num_samples)
test_size = int(0.2 * num_samples)

train_data = df.sample(n=train_size, replace=False)
df = df.loc[~df.index.isin(train_data)]
val_data = df.sample(n=val_size, replace=False)
df = df.loc[~df.index.isin(val_data)]
test_data = df.copy()

train_data.to_csv(dataset_name + '/' + 'train.txt', sep='\t', index=False, header=None)
val_data.to_csv(dataset_name + '/' + 'valid.txt', sep='\t', index=False, header=None)
test_data.to_csv(dataset_name + '/'  + 'test.txt', sep='\t', index=False, header=None)


