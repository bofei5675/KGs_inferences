import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-enum", "--entities-num",
                      default=100, type=int, help="number of entities")
parser.add_argument("--dir",
                      default='../data/binary/', type=str, help="directory")

def main():
    global args
    args = parser.parse_args()
    entities_num = args.entities_num
    dataset_name = os.path.join(args.dir, 'sample_{}'.format(entities_num))

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    d = {}
    with open(args.dir + 'node_list.txt', 'r') as f1, \
            open(dataset_name + '/' + 'entity2id.txt', 'w') as f2:
        next(f1)
        for line in f1:
            (key, val) = line.split()
            d[key] = val
            f2.write(val + '\t' + key + '\n')

    with open(args.dir + 'DrugBank_DDI.edgelist', 'r') as f1, \
            open(dataset_name + '/'  + 'data_full.txt', 'w') as f2:
        for line in f1:
            drug1, drug2 = line.split()
            relation = '_ true _.'
            f2.write(d[drug1] +'\t'+ relation + '\t' + d[drug2] + '\n')

    with open(dataset_name + '/'  + 'relation2id.txt', 'w') as f:
        f.write('_ true _.' + '\t' + '0\n')

    df = pd.read_csv(args.dir + 'data_full.txt', sep='\t', header=None)
    num_samples = entities_num  # df.shape[0]

    train_size = int(0.7 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = int(0.2 * num_samples)

    train_data = df.sample(n=train_size, replace=False)
    df = df.loc[~df.index.isin(train_data)]
    val_data = df.sample(n=val_size, replace=False)
    df = df.loc[~df.index.isin(val_data)]
    test_data = df.sample(n=test_size, replace=False)

    train_data.to_csv(dataset_name + '/' + 'train.txt', sep='\t', index=False, header=None)
    val_data.to_csv(dataset_name + '/' + 'valid.txt', sep='\t', index=False, header=None)
    test_data.to_csv(dataset_name + '/' + 'test.txt', sep='\t', index=False, header=None)


if __name__=='__main__':
    main()
