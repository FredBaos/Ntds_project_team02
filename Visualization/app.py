import dash
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import bs4 as bs

import webbrowser
import json
import urllib.request
from utils import *
import sys
sys.path.append("..")
from predict import *
from config import *

import os
parent_dir = os.path.dirname(os.getcwd())


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server


#Load data for the graph
df_node = pd.read_csv(os.path.join(parent_dir,DF_NODE_FILENAME)).drop('Unnamed: 0',axis=1)
df_edge = pd.read_csv(os.path.join(parent_dir,DF_EDGE_FILENAME)).drop('Unnamed: 0',axis=1)

# Create placeholders for graph information
labels = df_node['name'].tolist()
color_node = ['rgba(200,200,200,0.05)' for i in range(3*len(labels))]
color_node_original = color_node.copy()
texts_to_show = [None for i in range(len(labels))]
texts_to_show_original = texts_to_show.copy()
size_node = [15 for i in range(len(labels))]
size_node_original = size_node.copy()
color_edge = ['rgba(128,128,128,0.7)' for _ in range(3*len(df_edge))]
color_edge_original = color_edge.copy()

_, _, _, _, _, g = load_graph('graph,pkl')
FIGURE = {
                'data': [g.data[0],g.data[1]],
                'layout': g.layout
         }
query_bot = load_models(parent_dir)



app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Wikipedia Recommender System",
                                    className="uppercase title",
                                ),
                                html.Span(
                                    "By clicking on a node, you will be directed on the corresponding web page."
                                ),
                                html.Br(),
                                html.Span(" ", className="uppercase bold"),
                                html.Span(
                                    "."
                                ),
                            ]
                        )
                    ],
                    className="app__header",
                ),
                html.Div(
                    [
                        dcc.Input(
                            id='query',
                            placeholder='Enter a query...',
                            type='text',
                            value=''
                        )  
                    ],
                    className="app__input",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="method_radio",
                                    options=[{"label": method, "value": method} for method in METHODS],
                                    labelClassName="radio__labels",
                                    inputClassName="radio__input",
                                    value="Node2Vec",
                                    className="radio__group",
                                ),
                                html.Div(
                                    id='answer'
                                ),

                                dcc.Graph(
                                    id="graph",
                                    figure=FIGURE,
                                ),
                            ],
                            className="two-thirds column",
                        ),
                    ],
                    className="container card app__content bg-white",
                ),
                html.Div(
                    [
                        html.Span(
                                    "By Team 2"
                                ),
                    ],
                    className="container bg-white p-0",
                ),
            ],
            className="app__container",
        ),

        html.Div(id='hidden-div', style={'display':'none'}),
    ]
)

with g.batch_update():
    g.data[1].marker.color = color_node_original.copy()
    g.data[1].marker.size = size_node_original.copy()
    g.data[1].text = texts_to_show_original.copy()
    g.data[0].line.color = color_edge_original.copy()
    g.data[0].line.width = 10


# Open url when clicking on node
def update_point(trace, points, selector):
    if len(points.point_inds) != 0:
        url = g.data[1].customdata[points.point_inds[0]]
        webbrowser.open_new_tab(url)
   
@app.callback(
    Output('hidden-div', 'children'),
    [Input('graph', 'clickData')]
)
def display_click_data(clickData):
    if clickData:
        if len(clickData) != 0:
            pt = clickData['points'][0]
            try:
                url = pt['customdata']
                webbrowser.open_new_tab(url)
            except:
                return None
    return None

def find_index_edges(nodes_ls):
    edges_idx = []
    for source in nodes_ls:
        for target in nodes_ls:
            filtered = df_edge[((df_edge.source == source) & (df_edge.target == target) ) | ( (df_edge.source == target) & (df_edge.target == source) )]
            if len(filtered)!=0:
                edges_idx.append(filtered.index.values[0])
                
    return edges_idx

# From https://github.com/plotly/dash/issues/118
def convert_html_to_dash(el,style = None):
    CST_PERMITIDOS =  {'div','span','a','hr','br','p','b','i','u','s','h1','h2','h3','h4','h5','h6','ol','ul','li',
                        'em','strong','cite','tt','pre','small','big','center','blockquote','address','font','img',
                        'table','tr','td','caption','th','textarea','option'}
    def __extract_style(el):
        if not el.attrs.get("style"):
            return None
        return {k.strip():v.strip() for k,v in [x.split(": ") for x in el.attrs["style"].split(";")]}

    if type(el) is str:
        return convert_html_to_dash(bs.BeautifulSoup(el,'html.parser'))
    if type(el) == bs.element.NavigableString:
        return str(el)
    else:
        name = el.name
        style = __extract_style(el) if style is None else style
        contents = [convert_html_to_dash(x) for x in el.contents]
        if name.title().lower() not in CST_PERMITIDOS:        
            return contents[0] if len(contents)==1 else html.Div(contents)
        return getattr(html,name.title())(contents,style = style)

@app.callback(
    [Output("answer", "children"),Output('graph', 'figure')],
    [Input("query", "n_submit"),Input("query", "n_blur"),Input("method_radio", "value")],
    [State("query","value")]
)
# On a new query, compute the predictions and color the nodes accordingly
def make_query(ns,nb,current_selector,query):
    texts_to_show = texts_to_show_original.copy()
    color_node = color_node_original.copy()
    size_node = size_node_original.copy()
    color_edge = color_edge_original.copy()
    html_text = ""
    
    if query != "":
        preds = query_bot.make_prediction(query, current_selector)
        if preds == None:
            html_text = create_text([],df_node)
        else:
            dict_colors = compute_color(preds)
            html_text = create_text(list(dict_colors.keys()),df_node)

            for k,v in dict_colors.items():
                size_node[k] = 30
                color_node[k] = v
                texts_to_show[k] = labels[k]
            
            edges_idx = find_index_edges(list(preds.keys()))
            for idx in edges_idx:
                color_edge[3*idx] = 'rgb(255,0,0,1)'
            

        with g.batch_update():
            g.data[1].marker.size = size_node
            g.data[1].marker.color = color_node
            g.data[1].text = texts_to_show
            g.data[0].line.color = color_edge

    fig = {
            'data': [g.data[0],g.data[1]],
            'layout': g.layout
            }

    return convert_html_to_dash(html_text), fig

if __name__ == "__main__":
    app.run_server(debug=True)