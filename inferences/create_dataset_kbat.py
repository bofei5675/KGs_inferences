import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-enum", "--entities-num",
                      default=100, type=int, help="number of entities")
parser.add_argument("--dir",
                      default='../data/', type=str, help="directory")

# entities_num = 100
# dataset_name = './drugbank_{}'.format(entities_num)

def main():
    global args
    args = parser.parse_args()
    entities_num = args.entities_num
    dataset_name = os.path.join(args.dir, 'subsample_{}'.format(entities_num))
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    df = pd.read_csv('./drugbank_data_subset{}.txt'.format(entities_num), sep='\t', header=None)
    num_samples = df.shape[0]

    entities = df.iloc[:, 0].tolist() + df.iloc[:, 2].tolist()
    entities = set(entities)
    entities = pd.DataFrame(entities)
    entities.reset_index(inplace=True)
    # entities.to_csv(dataset_name + '/id2entity.txt', sep='\t', index=False, header=None)
    entities = entities.iloc[:, [1, 0]]
    entities.to_csv(dataset_name + '/entity2id.txt', sep='\t',index=False, header=None)

    relations = df.iloc[:, 1].tolist()
    relations = set(relations)
    relations = pd.DataFrame(relations)
    relations.reset_index(inplace=True)
    # relations.to_csv(dataset_name + '/idrelation.txt', sep='\t', index=False, header=None)
    relations = relations.iloc[:, [1, 0]]
    relations.to_csv(dataset_name + '/relation2id.txt', sep='\t', index=False, header=None)

    train_size = int(0.7 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = int(0.2 * num_samples)
    print('Data size', train_size, val_size, test_size)
    relations_set = set(df.iloc[:, 1].tolist())
    count = 0
    # get all relation at least once
    drop_set = []
    train_data = []
    for r in relations_set:
        drug_interactions = df.loc[df.iloc[:, 1] == r]
        drug_interactions = drug_interactions.iloc[0]
        drop_set.append(drug_interactions.name)
        train_data.append(drug_interactions.tolist())

    df =  df.loc[~df.index.isin(drop_set)]
    print('After processed', len(train_data))
    train_size = train_size - len(train_data)
    print('Train size left', train_size)
    train_data = pd.DataFrame(train_data)
    train_data = train_data.append(df.sample(n=train_size, replace=False))
    total_r_in_train = len(set(train_data.iloc[:,1].tolist()))
    print('Sampling condition:', len(relations_set), total_r_in_train)

    df = df.loc[~df.index.isin(train_data.index)]
    val_data = df.sample(n=val_size, replace=False)
    df = df.loc[~df.index.isin(val_data.index)]
    test_data = df.copy()
    print(train_data.shape, val_data.shape, test_data.shape)
    print(f'Train has {len(set(train_data.iloc[:,1].tolist()))}')
    print(f'Original dataset has {len(relations_set)}')
    print('Check overlap', train_data.index.isin(test_data.index).sum())
    train_data.to_csv(dataset_name + '/' + 'train.txt', sep='\t', index=False, header=None)
    val_data.to_csv(dataset_name + '/' + 'valid.txt', sep='\t', index=False, header=None)
    test_data.to_csv(dataset_name + '/' + 'test.txt', sep='\t', index=False, header=None)

if __name__=='__main__':
    main()
