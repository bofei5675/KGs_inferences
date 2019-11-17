import os
import pandas as pd

dataset_name = './drugbank_1000'

if not os.path.exists(dataset_name):
    os.makedirs(dataset_name)

df = pd.read_csv('./drugbank_data_subset1000.txt', sep='\t', header=None)
num_samples = df.shape[0]

entities = df.iloc[:, 0].tolist() + df.iloc[:, 2].tolist()
entities = set(entities)
entities = pd.DataFrame(entities)
entities.reset_index(inplace=True)
entities = entities.iloc[:, [1, 0]]
entities.to_csv(dataset_name + '/entity2id.txt', sep='\t',index=False, header=None)

relations = df.iloc[:, 1].tolist()
relations = set(relations)
relations = pd.DataFrame(relations)
relations.reset_index(inplace=True)
relations = relations.iloc[:, [1, 0]]
relations.to_csv(dataset_name + '/relation2id.txt', sep='\t', index=False, header=None)

train_size = int(0.7 * num_samples)
val_size = int(0.1 * num_samples)
test_size = int(0.2 * num_samples)
relations_set = set(df.iloc[:, 1].tolist())
count = 0
while True: # make sure train data has all relations
    train_data = df.sample(n=train_size, replace=False)
    total_r_in_train = len(set(train_data.iloc[:, 1].tolist()))
    if total_r_in_train == len(relations_set):
        break
    else:
        count += 1
        print(count ,'Sampling failed')

df = df.loc[~df.index.isin(train_data)]
val_data = df.sample(n=val_size, replace=False)
df = df.loc[~df.index.isin(val_data)]
test_data = df.copy()

print(f'Train has {len(set(train_data.iloc[:,1].tolist()))}')
print(f'Original dataset has {len(relations_set)}')

train_data.to_csv(dataset_name + '/' + 'train.txt', sep='\t', index=False, header=None)
val_data.to_csv(dataset_name + '/' + 'valid.txt', sep='\t', index=False, header=None)
test_data.to_csv(dataset_name + '/' + 'test.txt', sep='\t', index=False, header=None)