import igraph as ig
import chart_studio.plotly as py
import pandas as pd
import webbrowser
import json
import urllib.request
import plotly.graph_objs as go
import pickle
from ipywidgets import widgets
from scipy.special import softmax
from config import *

def create_text(keys_ls,df_node):
    """Create HTML formatted text for the answer of the query."""
    if len(keys_ls) == 0:
        base_text = "\n \n No pages found"
    else:
        base_text = "\n \n The most prominent pages are: \n"
        for key in keys_ls:
            filtered = df_node[df_node.index == key][['name','url']].values[0]
            url = filtered[1]
            name = filtered[0]
            url_text = "* [{}]({}) \n".format(name,url)
            base_text += url_text
    return base_text

def compute_color(preds):
    """Compute color for the nodes. Return a dict where the key is node index and the value is a float for the colorscale."""
    all_scores = list(preds.values())
    color_values = softmax(all_scores)
    color_output = {}
    diff = max(color_values)
    for k,v in zip(preds.keys(),color_values):
        color_output[k] = v/diff
    return color_output

def create_plot_items(df_node,df_edge,labels,color_node,texts_to_show,color_edge):
    N = len(df_node)
    
    # Extract list
    urls = df_node['url'].tolist()
    Edges = df_edge[['source','target']].values.tolist()

    # Create graph
    G = ig.Graph(Edges, directed=False)
    layt = G.layout('kk', dim=3)

    # Coordinates for the nodes and edges
    Xn = [layt[k][0] for k in range(N)]
    Yn = [layt[k][1] for k in range(N)]
    Zn = [layt[k][2] for k in range(N)]
    Xe = []
    Ye = []
    Ze = []
    for e in Edges:
        Xe += [layt[e[0]][0],layt[e[1]][0], None]
        Ye += [layt[e[0]][1],layt[e[1]][1], None]
        Ze += [layt[e[0]][2],layt[e[1]][2], None]

    # Plot the edges
    trace1 = go.Scatter3d(x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   line=dict(color=color_edge, width=1),
                   hoverinfo='none'
                   )
    # Plot the nodes
    trace2 = go.Scatter3d(x=Xn,
                   y=Yn,
                   z=Zn,
                   mode='markers+text',
                   name='actors',
                   marker=dict(symbol='circle',
                                 size=6,
                                 color=color_node,
                                 colorscale='Rainbow',
                                 line=dict(color='rgb(50,50,50)', width=0.5),
                                   colorbar=dict(
                                        title=""
                                    ),
                                 ),
                   text=texts_to_show,
                   textposition="top center",
                   textfont = {'size':60},
                   hovertext=labels,
                   hoverinfo='text',   
                   customdata=urls
           )

    # Set up the axis
    axis = dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )

    # Custom layout
    layout = go.Layout(
             title="",
             width=900,
             height=900,
             showlegend=False,
             scene=dict(
                 xaxis=dict(axis),
                 yaxis=dict(axis),
                 zaxis=dict(axis),
            ),
         margin=dict(
            t=100
        ),
        hovermode='closest',
      )

    # Create Figure
    g = go.FigureWidget(data=[trace1, trace2],layout=layout)
    
    return g

def create_and_save_plot():
    df_node = pd.read_csv(os.path.join(GENERATED_DATA_PATH,DF_NODE_FILENAME)).drop('Unnamed: 0',axis=1)
    df_edge = pd.read_csv(os.path.join(GENERATED_DATA_PATH,DF_EDGE_FILENAME)).drop('Unnamed: 0',axis=1) 

    labels = df_node['name'].tolist()
    color_node = ['rgba(200,200,200,0.05)' for i in range(3*len(labels))]
    texts_to_show = [None for i in range(len(labels))]
    size_node = [15 for i in range(len(labels))]
    color_edge = ['rgba(128,128,128,0.7)' for _ in range(3*len(df_edge))]

    g = create_plot_items(df_node,df_edge,labels,color_node,texts_to_show,color_edge)
    with open(os.path.join(GENERATED_DATA_PATH,'graph.pkl'),'wb') as f:
        pickle.dump(g,f)


def load_graph():
    with open(os.path.join(GENERATED_DATA_PATH,'graph.pkl'),'rb') as f:
        g = pickle.load(f)
    
    # Title of the visualization
    title = widgets.HTML(
        value="<h3> Wikipedia Recommender System </h3>",
    )

    # Some help text
    annotations = widgets.HTML(
        value="<h4> By clicking on a node, you will be directed on the corresponding web page. </h4>",
    )
    
    # Text Box for the query
    textbox_query = widgets.Text(
        value='',
        placeholder='Type something',
        description='Query:',
        disabled=False
    )

    # Select Method
    selector = widgets.Select(
        options=METHODS,
        value='Node2Vec',
        description='Method:',
        disabled=False
    )

    query_answer = widgets.HTML(value='')
    
    return title, annotations, textbox_query, selector, query_answer, g
