INITIAL_FILENAME = "ml-nlp-ai-ann-chatbot-ia-dataviz-dist2.tsv"
METHODS = ['FastText - Random Walk', 'Node2Vec', 'Spectral Clustering']
TOPK = 5
SPECTRAL_CLUSTERING_FILENAME = 'spectral.pkl'
FAST_MEAN_FILENAME = 'fast_dict.pkl'
NODE2VEC_FILENAME = 'node2vec.model'
DF_NODE_FILENAME = 'df_node.csv'
DF_EDGE_FILENAME = 'df_edge.csv'

import os
DATA_PATH = os.path.join(os.getcwd(),'data')
GENERATED_DATA_PATH = os.path.join(DATA_PATH,'generated')
OUTPUT_EXPLOITATION_PATH = os.path.join(os.path.join(os.getcwd(),'exploitation'),'output')

import numpy as np
import random
random.seed(1)
np.random.seed(1)