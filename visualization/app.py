# Inspired from https://plot.ly/python/v3/3d-network-graph/

import sys
import dash
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
import webbrowser
import urllib.request
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from utils import *
from helpers.predict import *
from config import *

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

# Load data for the graph
df_node = pd.read_csv(os.path.join(GENERATED_DATA_PATH,DF_NODE_FILENAME)).drop('Unnamed: 0',axis=1)
df_edge = pd.read_csv(os.path.join(GENERATED_DATA_PATH,DF_EDGE_FILENAME)).drop('Unnamed: 0',axis=1)

# Create placeholders for graph information
labels = df_node['name'].tolist()
color_node = ['rgba(255,255,255,0.05)' for i in range(3*len(labels))]
color_node_original = color_node.copy()
texts_to_show = [None for i in range(len(labels))]
texts_to_show_original = texts_to_show.copy()
size_node = [15 for i in range(len(labels))]
size_node_original = size_node.copy()
color_edge = ['rgba(128,128,128,0.7)' for _ in range(3*len(df_edge))]
color_edge_original = color_edge.copy()

# Load graph 
g = load_graph()
# Create figure
FIGURE = {
                'data': [g.data[0],g.data[1]],
                'layout': g.layout
         }
# Load query models
query_bot = load_models()

# HTML layout
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
                                    "By clicking on a node, 'Chosen node' link will redirect you to the corresponding web page."
                                ),
                                html.Br(),
                                html.Span("The page title of nodes that best fit the query as well as the neighbours are shown."),
                                html.Br(),
                                html.Span(
                                    "Red edges mean that the pages are in the 'See also' section on Wikipedia website."
                                ),
                                html.Br(),
                                html.Span(
                                    "The color represents the cosine similarity score."
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
                                
                                dcc.Markdown(
                                    id='answer',
                                ),

                                dcc.Markdown(id="url",
                                className='url_center',),

                                dcc.Graph(
                                    id="graph",
                                    figure=FIGURE,
                                ),
                            ],

                        ),
                    ],
                    className="container card app__content bg-white",
                ),
                html.Div(
                    [
                        html.Br(),
                        html.P(
                                    "By Team 2 - Network Tour of Data Science - EPFL"
                                ),
                        html.A('Github code',
                            href='https://github.com/FredBaos/Ntds_project_team02'
                        ),
                    ],
                    className="container p-0",
                ),
            ],
            className="app__container",
        ),

        html.Div(id='hidden-div', style={'display':'none'}),
    ]
)

# Initialization of graph information
with g.batch_update():
    g.data[1].marker.color = color_node_original.copy()
    g.data[1].marker.size = size_node_original.copy()
    g.data[1].text = texts_to_show_original.copy()
    g.data[0].line.color = color_edge_original.copy()
    g.data[0].line.width = 10
    g.data[1].marker.colorscale = 'Magma'
    g.data[0].opacity = 0.5
    g.data[1].textfont['size'] = 40

# Open url when clicking on a node
@app.callback(
    Output('url', 'children'),
    [Input('graph', 'clickData')]
)
def display_click_data(clickData):
    if clickData:
        if len(clickData) != 0:
            pt = clickData['points'][0]
            try:
                url = pt['customdata']
                name = df_node[df_node.url==url].name.values[0]
                webbrowser.open_new_tab(url)
                return "Chosen node: [{}]({})".format(name,url)
            except:
                return None
    return None

# On a new query, compute the predictions, color the nodes and edges accordingly.
# Show title of relevant nodes (answers + neighbours)
@app.callback(
    [Output("answer", "children"),Output('graph', 'figure')],
    [Input("query", "n_submit"),Input("query", "n_blur"),Input("method_radio", "value")],
    [State("query","value")]
)
def make_query(ns,nb,current_selector,query):
    # Initialize the information back to zero normal state.
    texts_to_show = texts_to_show_original.copy()
    color_node = color_node_original.copy()
    size_node = size_node_original.copy()
    color_edge = color_edge_original.copy()
    markdown_text = ""
    
    if query != "":
        preds = query_bot.make_prediction(query, current_selector)
        if preds == None:
            # Show no pages found
            markdown_text = create_text([],df_node)
        else:
            dict_colors = compute_color(preds)
            markdown_text = create_text(list(dict_colors.keys()),df_node)

            # Change chosen nodes size and color
            # Display title page of chosen nodes and their neighbours
            for k,v in dict_colors.items():
                size_node[k] = 30
                color_node[k] = v
                texts_to_show[k] = labels[k]
                for i in df_edge[df_edge.source == k]['target'].values:
                    texts_to_show[i] = labels[i]
            
            # Color in red edges between chosen nodes
            edges_idx = find_index_edges(list(preds.keys()),df_edge)
            for idx in edges_idx:
                color_edge[3*idx] = 'rgb(255,0,0,1)'
            
        # Update the graph
        with g.batch_update():
            g.data[1].marker.size = size_node
            g.data[1].marker.color = color_node
            g.data[1].text = texts_to_show
            g.data[0].line.color = color_edge

    fig = {
            'data': [g.data[0],g.data[1]],
            'layout': g.layout
            }

    return markdown_text, fig

# Run server on port 80 need to be opened for full access
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(sys.argv[1]), debug=True)