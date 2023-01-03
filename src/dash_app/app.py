# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from datetime import date, datetime, time, timedelta

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
now = datetime.now()
tommorrow = now + timedelta(days=1)
weekday_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def get_prediction_for_today(now):
    if now.hour < 10:
        return 'This is yesterdays pred for today lunch'
    elif now.hour < 13:
        return 'This is the 10 am predictions for today lunch'
    else:
        return 'Lunch has passed, take a look at the prediction for tomorrow'

def get_prediction_for_tomorrow(now):
    if now.hour < 10:
        return 'This is yesterdays pred for today lunch'
    elif now.hour < 13:
        return 'This is the 10 am predictions for today lunch'
    else:
        return 'Lunch has passed, take a look at the prediction for tomorrow'

def format_date(datetime_object):
    return f'{weekday_list[datetime_object.weekday()]}, {datetime_object.day}.{datetime_object.month}.{datetime_object.year}'

app.layout = html.Div(children=[
    html.H1(children=f'Predictions for today, {format_date(now)}'),
    html.Div(children=get_prediction_for_today(now=now)),

    html.H1(children=f'Predictions for tomorrow, {format_date(tommorrow)}'),
    html.Div(children=get_prediction_for_today(now=now)),
    html.H1(children=f'Predictions for future days'),
    html.Div(children='show prediction for future days'),

    html.H1(children=f'Error analysis'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
