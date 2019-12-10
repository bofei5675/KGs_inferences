import pandas as pd
import numpy as np
import os
import torch
import sys
sys.path.append('../')
from models import SpKBGATModified, SpKBGATConvOnly

def main():
    df = pd.read_csv('./data/tanimoto_info_PCA50.csv')
    data_path = '/scratch/bz1030/relationPrediction/data/deepddi'
    print(df.head())
    print(df.shape)
    entity_ids = pd.read_csv(data_path + '/entity2id.txt', sep='\t', header=None)
    relation_ids = pd.read_csv(data_path + '/relation2id.txt', sep='\t', header=None)
    entity_ids.columns = ['drug_id', 'id']
    relation_ids.columns = ['relation_id', 'id']
    col_names = df.columns.tolist()
    col_names[0] = 'drug_id'

    df.columns = col_names

    df = pd.merge(df, entity_ids, left_on='drug_id', right_on='drug_id', how='inner')
    print(df.head())
    selected_cols = ['drug_id'] + ['PC_{}'.format(i) for i in range(1,51)]
    df = df[selected_cols]
    df.set_index('drug_id', inplace=True)
    df.to_csv('entity2vec.txt', sep='\t', header=None)
    # load embedding from torch
    CUDA = torch.cuda.is_available()
    load_model = '/scratch/bz1030/relationPrediction/checkpoints/deepddi_tanh/gat/trained_299.pth'
    if CUDA:
        gat = torch.load(load_model)
    else:
        gat = torch.load(load_model, map_location='cpu')
    learned_ent_emb = gat['module.entity_embeddings'].data.numpy()
    learned_relation_emb = gat['module.relation_embeddings'].data.numpy()
    print(learned_ent_emb.shape)

    learned_ent_emb  = pd.DataFrame(learned_ent_emb)
    learned_ent_emb['drug_id'] = entity_ids['drug_id']
    learned_relation_emb = pd.DataFrame(learned_relation_emb)
    learned_relation_emb['relation_id'] = relation_ids['relation_id']
    learned_ent_emb.to_csv('entity2vec_gat.csv', header=None, sep='\t', index=False)
    learned_relation_emb.to_csv('relation2vec_gat.csv', header=None, sep='\t', index=False)


if __name__ == '__main__':
    main()
