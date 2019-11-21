import pandas as pd
import time

df = pd.read_csv('./drugbank_data_subset100.txt', sep='\t', header=None)

df.head()

df = df.iloc[:, [0, 2]]

print(df.head())

for idx, row in df.iterrows():
    row = row.tolist()
    drug1, drug2 = row[0], row[1]

    tuple1 = df.loc[(df.iloc[:, 0] == drug1) & (df.iloc[:, 1] == drug2)]
    tuple2 = df.loc[(df.iloc[:, 1] == drug1) & (df.iloc[:, 0] == drug2)]
    if len(tuple1) > 1 or len(tuple2) > 1:
        print(drug1, drug2, len(tuple1), len(tuple2))
