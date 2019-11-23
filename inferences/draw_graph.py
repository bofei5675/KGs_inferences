import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

G = nx.Graph()

deepddi = pd.read_csv('/scratch/bz1030/relationPrediction/data/drugbank_100/valid.txt', header=None, sep='\t')

for idx, row in deepddi.iterrows():
    node1, node2 = (row.iloc[0], row.iloc[2])
    G.add_edge(node1, node2)

nx.draw_random(G, node_size=5, width=1)
plt.savefig("simple_path.png", dpi=300) # save as png
#plt.show() # display