import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
import sys
import pandas as pd
import os

dataset = pd.read_csv("Stream/stream_file.csv")
dataset['date'] = pd.to_datetime(dataset['date'])

print(dataset.head())
X = [] #PRIMARY
Y = []

X2 = [] #PRIMARY ANOM
Y2 = []

X3 = [] #X
Y3 = []

X4 = [] #X ANOM
Y4 = []

X5 = [] #Y
Y5 = []

X6 = [] #Y ANOM
Y6 = []


app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}


app = dash.Dash(__name__)
app.layout = html.Div(
    children = [
        html.Div(
            children = [
                dcc.Graph(id='live-graph',
                style={'width': '70%', 'height': '900px', 'display': 'inline-block'},
                animate=True,
                ),
                dcc.Interval(
                    id='graph-update',
                    interval=1*1000, #this is in milliseconds
                    n_intervals = 0
                    ),
                html.Div(
                style={'width': '30%', 'height': '50%', 'margin-bottom':500, 'display': 'inline-block'},
                children = [
                html.Div(id='live-update-text',
                style={'width':'75%', 'margin-bottom':500, 'textAlign': 'center',  'font-family': 'Courier New'}),
                html.Div(id='live-update-text-2',
                style={'width':'75%', 'height': '900px', 'margin-top':500, 'textAlign': 'center', 'verticalAlign':'top', 'font-family': 'Courier New'}),
                ])
                ]
        )
    ]
)

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])

def update_graph_scatter(n):
    l = len(X)
    for i in range(24):
        X.append(dataset['date'].iloc[l+i])
        Y.append(dataset['z'].iloc[l+i])

        if dataset['anomaly'].iloc[(len(X)+i)] == True:
            X2.append(dataset['date'].iloc[l])
            Y2.append(dataset['z'].iloc[l])

        X3.append(dataset['date'].iloc[(l+i)])
        Y3.append(dataset['x_x'].iloc[(l+i)])

        if dataset['anomaly_x'].iloc[(len(X)+i)] == True:
            X4.append(dataset['date'].iloc[(len(X)+i)])
            Y4.append(dataset['x_x'].iloc[(len(X)+i)])

        X5.append(dataset['date'].iloc[l+i])
        Y5.append(dataset['y_x'].iloc[l+i])

        if dataset['anomaly_y'].iloc[(len(X)+i)] == True:
            X6.append(dataset['date'].iloc[(len(X)+i)])
            Y6.append(dataset['y_x'].iloc[(len(X)+i)])

    fig = plotly.subplots.make_subplots(rows=2,
                          cols=1,
                          print_grid=True,
                          horizontal_spacing=0.18,
                         )

    data_1 = go.Scatter(
            x=list(X),
            y=list(Y),
            name='Primary Axis',
            mode= 'lines',
            yaxis = 'y1'
            )


    data_2 = go.Scatter( #PRIMARY ANOM
            x=list(X2),
            y=list(Y2),
            name='Anomaly',
            mode= 'markers',
            marker=dict(
            color='Red',
            size=5,
        ),
        )


    data_3 = go.Scatter(
            x=list(X3), #X
            y=list(Y3),
            name='Vibration - X Axis',
            mode= 'lines',
            )

    data_4 = go.Scatter(
            x=list(X4), #X Anom
            y=list(Y4),
            name='Anomaly',
            mode= 'markers',
            marker=dict(
            color='Red',
            size=5,
        ),
        )

    data_5 = go.Scatter(
            x=list(X5), #Y
            y=list(Y5),
            name='Vibration - Y Axis',
            mode= 'lines',
        )

    data_6 = go.Scatter(
                x=list(X6),
                y=list(Y6), #Y Anom
                name='Anomaly',
                mode= 'markers',
                marker=dict(
                color='Red',
                size=5,
            ),
            )


    fig['layout'].update(
        title='Elevator 1 - Operation Status',
        paper_bgcolor="LightSteelBlue",
        xaxis=dict(range=['2018-07-09T12:05:16.701', '2018-07-09T12:07:37.900']
        ))


    fig.append_trace(data_1, 1, 1)
    fig.append_trace(data_2, 1, 1)
    fig.append_trace(data_3, 2, 1)
    fig.append_trace(data_4, 2, 1)
    fig.append_trace(data_5, 2, 1)
    fig.append_trace(data_6, 2, 1)



    fig['layout']['yaxis1'].update(range= [-15, 15], #left yaxis'y1
                         showgrid=True,
                         title= 'Acceleration (g)',
                         )

    fig['layout']['yaxis2'].update(range= [-15, 15], #left yaxis'y1
                          showgrid=True,
                          title= 'Acceleration (g)',
                          )

    fig['layout']['xaxis2'].update(range=['2018-07-09T12:05:16.701', '2018-07-09T12:07:37.900']),


    fig.update_layout(legend_orientation="h")

    return fig

@app.callback(Output('live-update-text', 'children'),
              [Input('graph-update', 'n_intervals')])


def update_metrics(n):
    style = {'fontSize': '45px', 'padding-right': '30px'}
    style2 = {'fontSize': '25px', 'padding-left': '30px', 'color':'red' }
    style3 = {'fontSize': '25px', 'padding-left': '30px'}
    #if abs(new_val - old_val) >= 0.2:

    if dataset['trip'].iloc[len(Y)] == 'up':
        status = 'Ascending...'
    elif dataset['trip'].iloc[len(Y)] == 'down':
        status = 'Descending... '
    else:
        status = 'Elevator Waiting...'



    return [
            html.Span(status, style=style),
                ]

@app.callback(Output('live-update-text-2', 'children'),
              [Input('graph-update', 'n_intervals')])


def update_metrics(n):

    return "."


if __name__ == '__main__':
    app.run_server(debug=False)
