import pandas as pd
import os
import numpy as np
'''
Generate analysis data for this paper
'''

def main():
    chem_meta_data = pd.read_csv('/scratch/bz1030/relationPrediction/inferences/data/drugbank_emb_labels.txt', sep='\t',
                                 header=None)

    emb = pd.read_csv('/scratch/bz1030/relationPrediction/checkpoints/deepddi_tanh/entity_emb.txt', sep=' ',
                      header=None)
    print(chem_meta_data.head())
    print(emb.head())
    data = pd.merge(emb, chem_meta_data, left_on=0, right_on=0, how='inner')
    print(data.head())
    data.set_index(0, inplace=True)
    data.to_csv('./data/analysis_data_chem_class.csv', header=None)


if __name__ == '__main__':
    main()