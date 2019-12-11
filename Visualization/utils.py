import igraph as ig
import pandas as pd
import webbrowser
import json
import urllib.request
import chart_studio.plotly as py
import plotly.graph_objs as go
from ipywidgets import widgets
from scipy.special import softmax
import pickle
import sys
sys.path.append("..")
from config import *



# Create HTML formatted text for the answer of the query
def create_text(keys_ls,df_node):
    if len(keys_ls) == 0:
        base_text = "No pages found"
    else:
        base_text = "The most prominent pages are :<ul>"
        for key in keys_ls:
            filtered = df_node[df_node.index == key][['name','url']].values[0]
            url = filtered[1]
            name = filtered[0]
            url_text = "<li><a href=" + url + """ target="_blank"> """ + name + "</a></li>"
            base_text += url_text
        base_text += "</ul>"
    return base_text

# Compute color for the nodes
# Return a dict where the key is node index and the value is a float for the colorscale
def compute_color(preds):
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


def load_graph(filepath):
    with open(filepath,'rb') as f:
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
