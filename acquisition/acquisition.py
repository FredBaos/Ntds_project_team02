# Imports
import networkx as nx
import pandas as pd
import numpy as np
import random
import wikipedia
import time
import pickle
import re
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from operator import itemgetter
from config import *
from helpers.predict import *
from acquisition_helpers import *

# Creation of dataframes containing nodes and edges information

# Loading data
edge_list = pd.read_csv(os.path.join(DATA_PATH,INITIAL_FILENAME), sep = "\t")

# First, we create the node dataframe by filling the 'url' and 'keywords' columns.

df_node = pd.DataFrame(columns=['name','url', 'keywords'])
df_node['name'] = pd.unique(edge_list.source.append(edge_list.target))
    
# takes a long time to run, load a pickle by default
is_saved = True

if not is_saved:
    urls = []
    for name in df_node.name:
        urls.append(get_url(name))
    with open(os.path.join(DATA_PATH,'urls.pickle'), 'wb') as handle:
        pickle.dump(urls, handle)
        
else:
    with open(os.path.join(DATA_PATH,'urls.pickle'), 'rb') as handle:
        urls = pickle.load(handle)
        
df_node['url'] = urls

# Then, we create lists of keywords for each page by doing TF-IDF on pages summaries.

# takes a long time to run, summaries are saved in a pickle by default
is_saved = True

if not is_saved:
    counter = 0
    docs = []
    
    for name in df_node['name']:
        print(counter)
        counter += 1
        docs.append(pre_process(get_summary(name)))

    with open(os.path.join(DATA_PATH,'summaries.pickle'), 'wb') as handle:
        pickle.dump(docs, handle)
        
else:
    with open(os.path.join(DATA_PATH,'summaries.pickle'), 'rb') as handle:
        docs = pickle.load(handle)
        
stopwords = get_stop_words(os.path.join(DATA_PATH,"stopwords.txt"))
cv = CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector = cv.fit_transform(docs)

# TF-IDF transformer
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

# adding keywords to df_node
df_node['keywords'] = [list(get_keywords(docs[idx], tfidf_transformer, cv).keys()) for idx in range(len(df_node))]

df_node.to_csv(os.path.join(GENERATED_DATA_PATH,DF_NODE_FILENAME))

# Then, we create the edge dataframe, containing links between nodes.
df_edge = pd.DataFrame(columns=['source','target'])
df_edge['source'] = [df_node[df_node.name==name].index.values[0] for name in edge_list.source]
df_edge['target'] = [df_node[df_node.name==name].index.values[0] for name in edge_list.target]

df_edge.to_csv(os.path.join(GENERATED_DATA_PATH,DF_EDGE_FILENAME))