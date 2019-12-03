import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
G = nx.Graph()
deepddi = pd.read_csv('/scratch/bz1030/relationPrediction/data/drugbank_100/valid.txt', header=None, sep='\t')
relation2id = pd.read_csv('/scratch/bz1030/relationPrediction/data/drugbank_100/relation2id.txt', sep='\t',header=None)
drop_rate = 0.5
for idx, row in deepddi.iterrows():
    #if np.random.uniform(0, 1) < drop_rate:
    #    continue
    color = relation2id.loc[relation2id.iloc[:, 0] == row.iloc[1], 1]
    node1, node2 = (row.iloc[0], row.iloc[2])
    G.add_edge(node1, node2, color=color)

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
nx.draw_random(G, node_size=20, width=0.2, edge_color=colors)
plt.savefig("simple_path.png", dpi=300) # save as png
#plt.show() # display