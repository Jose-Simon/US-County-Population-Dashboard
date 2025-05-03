# ----------------------------------------------------------------------------
# Start of the program
# ----------------------------------------------------------------------------

print("==== Program Start ====")

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc, html, Output, Input, State, Dash, dash_table
import json
import time

start_time_program = time.time()

# ----------------------------------------------------------------------------
# Load the Full 2000-2024 Dataset
# ----------------------------------------------------------------------------

# File paths
csv_path = 'county_population_2000_2024_long.csv'
geojson_path = 'counties.geojson'

# 1. Load CSV with FIPS preserved as 5-digit string
df = pd.read_csv(csv_path, dtype={'FIPS': str})
df['FIPS'] = df['FIPS'].str.zfill(5)

# 2. Immediately drop rows where any of FIPS, State, County, Year, Population is missing
df = df.dropna(subset=['FIPS', 'State', 'County', 'Year', 'Population'])

# 3. Load GeoJSON
with open(geojson_path, 'r') as f:
    counties_geojson = json.load(f)
	
print("CSV and GeoJSON loaded successfully. Dataframe shape:", df.shape)

# 4. Build the state and county options
state_options = [{'label': state, 'value': state} for state in sorted(df['State'].dropna().astype(str).unique())]
county_options = [
    {'label': f"{row['County']}, {row['State']}", 'value': row['FIPS']}
    for _, row in df[['FIPS', 'County', 'State']].drop_duplicates().iterrows()
]

# 5. Define Population Groups
population_groups = [
    {"label": "500K+", "value": "1"},
    {"label": "100K-500K", "value": "2"},
    {"label": "50K-100K", "value": "3"},
    {"label": "30K-50K", "value": "4"},
    {"label": "20K-30K", "value": "5"},
    {"label": "10K-20K", "value": "6"},
    {"label": "5K-10K", "value": "7"},
    {"label": "<5K", "value": "8"}
]

# ----------------------------------------------------------------------------
# Initialize Dash App
# ----------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

app.layout = html.Div(style={'font-family': 'Helvetica, Arial, sans-serif', 'padding': '10px'}, children=[

    html.H1(id='dashboard-title', style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div(style={
        'backgroundColor': '#003366',
        'padding': '5px',
        'display': 'flex',
        'flex-wrap': 'wrap',
        'justify-content': 'space-around',
        'color': 'white',
        'gap': '10px',
        'margin-bottom': '10px'
    }, children=[
        html.Div([
            html.Label("Start Year", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='start-year-dropdown',
                options=[{'label': str(year), 'value': year} for year in range(2000, 2025)],
                value=2000,
                clearable=False,
                style={'backgroundColor': 'white', 'color': 'black', 'fontSize': '16px'}
            )
        ], style={'width': '10%', 'minWidth': '150px'}),

        html.Div([
            html.Label("End Year", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='end-year-dropdown',
                options=[{'label': str(year), 'value': year} for year in range(2000, 2025)],
                value=2024,
                clearable=False,
                style={'backgroundColor': 'white', 'color': 'black', 'fontSize': '16px'}
            )
        ], style={'width': '10%', 'minWidth': '150px'}),

        html.Div([
            html.Label("Metric", style={'font-weight': 'bold'}),
            dcc.RadioItems(
                id='metric-radio',
                options=[
                    {'label': 'Absolute Change', 'value': 'numeric_diff'},
                    {'label': 'Percentage Change', 'value': 'percent_diff'}
                ],
                value='percent_diff',
                labelStyle={'display': 'block', 'margin-top': '5px'}
            )
        ], style={'width': '10%', 'minWidth': '150px'}),

        html.Div([
            html.Label("State Filter", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='state-filter-dropdown',
                options=state_options,
                multi=True,
                placeholder="Select states...",
                style={'backgroundColor': 'white', 'color': 'black', 'fontSize': '16px'}
            )
        ], style={'width': '20%', 'minWidth': '200px'}),

        html.Div([
            html.Label("County Filter", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='county-filter-dropdown',
                options=county_options,
                multi=True,
                placeholder="Select counties...",
                style={'backgroundColor': 'white', 'color': 'black', 'fontSize': '16px'}
            )
        ], style={'width': '20%', 'minWidth': '200px'}),
		
		html.Div([
            html.Label("Population Group", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='population-group-dropdown',
                options=population_groups,
                multi=True,
                placeholder="Select Group...",
                clearable=True,
                style={'backgroundColor': 'white', 'color': 'black', 'fontSize': '16px'}
            )
        ], style={'width': '20%', 'minWidth': '180px'})
    ]),

    html.Div(id="summary-banner", style={
        'backgroundColor': '#003366',
        'padding': '5px',
        'display': 'flex',
        'justify-content': 'space-around',
        'color': 'white',
        'flex-wrap': 'wrap',
        'margin-bottom': '5px'
    }),

    dcc.Graph(id="choropleth-map", style={'height': '800px', 'width': '100%'}),

    html.Div(style={'display': 'flex', 'justify-content': 'space-around', 'margin-top': '20px', 'flex-wrap': 'wrap'}, children=[
        html.Div([
            html.H4("Growing Counties"),
            dash_table.DataTable(id='top10-table', style_table={'font-family': 'Helvetica, Arial, sans-serif', 'height': '350px', 'overflowY': 'auto'},
			style_cell={'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '14px', 'textAlign': 'right'},
			style_header={'fontFamily': 'Helvetica, Arial, sans-serif', 'fontWeight': 'bold', 'backgroundColor': '#003366', 'color': 'white'},
            style_cell_conditional=[
                {'if': {'column_id': 'county_state'}, 'textAlign': 'left'},
            ])
        ], style={'width': '45%'}),

        html.Div([
            html.H4("Declining Counties"),
            dash_table.DataTable(id='bottom10-table', style_table={'font-family': 'Helvetica, Arial, sans-serif', 'height': '350px', 'overflowY': 'auto'},
			style_cell={'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '14px', 'textAlign': 'right'},
			style_header={'fontFamily': 'Helvetica, Arial, sans-serif', 'fontWeight': 'bold', 'backgroundColor': '#003366', 'color': 'white'},
            style_cell_conditional=[
                {'if': {'column_id': 'county_state'}, 'textAlign': 'left'},
            ])
        ], style={'width': '45%'}),
    ])
])

# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------

@app.callback(
    Output('dashboard-title', 'children'),
    Input('start-year-dropdown', 'value'),
    Input('end-year-dropdown', 'value')
)
def update_title(start_year, end_year):
    return f"Population Change by US Counties ({start_year}-{end_year})"

@app.callback(
    Output('summary-banner', 'children'),
    Output('choropleth-map', 'figure'),
    Output('top10-table', 'data'),
    Output('top10-table', 'columns'),
    Output('bottom10-table', 'data'),
    Output('bottom10-table', 'columns'),
    Input('start-year-dropdown', 'value'),
    Input('end-year-dropdown', 'value'),
    Input('metric-radio', 'value'),
    Input('state-filter-dropdown', 'value'),
    Input('county-filter-dropdown', 'value'),
    Input('population-group-dropdown', 'value')
)
def update_dashboard(start_year, end_year, metric_type, selected_states, selected_counties, selected_group):
    dff = df.copy()
    if selected_states:
        dff = dff[dff['State'].isin(selected_states)]
    if selected_counties:
        dff = dff[dff['FIPS'].isin(selected_counties)]
    pop_start = dff[dff['Year'] == start_year][['FIPS', 'Population']]
    pop_end = dff[dff['Year'] == end_year][['FIPS', 'Population']]
    merged = pd.merge(pop_start, pop_end, on='FIPS', suffixes=('_start', '_end'))
    merged = merged.merge(dff[['FIPS', 'State', 'County']].drop_duplicates(), on='FIPS')

    bins = [-1, 4999, 9999, 19999, 29999, 49999, 99999, 499999, float('inf')]
    labels = ['8', '7', '6', '5', '4', '3', '2', '1']
    merged['PopGroup'] = pd.cut(merged['Population_start'], bins=bins, labels=labels)
    if selected_group:
        merged = merged[merged['PopGroup'].isin(selected_group)]

    merged['numeric_diff'] = merged['Population_end'] - merged['Population_start']
    merged['percent_diff'] = (merged['numeric_diff'] / merged['Population_start']) * 100
    merged['county_state'] = merged['County'] + ", " + merged['State']
    merged['start_year'] = start_year
    merged['end_year'] = end_year
    merged['total_county'] = len(merged)
    merged['Population_start_rank'] = merged['Population_start'].rank(ascending=False, method='min').astype(int)
    merged['Population_end_rank'] = merged['Population_end'].rank(ascending=False, method='min').astype(int)
    merged['numeric_diff_rank'] = merged['numeric_diff'].rank(ascending=False, method='min').astype(int)
    merged['percent_diff_rank'] = merged['percent_diff'].rank(ascending=False, method='min').astype(int)

    total_start_pop = merged['Population_start'].sum()
    total_end_pop = merged['Population_end'].sum()
    percent_change_total = (total_end_pop - total_start_pop) / total_start_pop * 100
    county_count = len(merged)

    if percent_change_total >= 0:
        arrow = "\u25B2"
        color = "green"
    else:
        arrow = "\u25BC"
        color = "red"

    summary = [
        html.Div([
            html.H4(f"{start_year} Population", style={'text-align': 'center'}),
            html.H2(f"{total_start_pop:,}", style={'text-align': 'center'})
        ]),
        html.Div([
            html.H4(f"{end_year} Population", style={'text-align': 'center'}),
            html.H2(f"{total_end_pop:,}", style={'text-align': 'center'})
        ]),
        html.Div([
            html.H4("Population Change", style={'text-align': 'center'}),
            html.H2([
                f"{percent_change_total:.2f}% ",
                html.Span(arrow, style={'color': color, 'font-size': '30px'})
            ], style={'text-align': 'center'})
        ]),
        html.Div([
            html.H4("Counties Displayed", style={'text-align': 'center'}),
            html.H2(f"{county_count:,}", style={'text-align': 'center'})
        ])
    ]

    hovertemplate = (
    "<b>%{customdata[0]}</b><br><br>"
    "<span style='font-size: 12px;'>"
    "%{customdata[10]} Population: %{customdata[1]:,} (Rank %{customdata[5]} of %{customdata[9]})<br>"
    "%{customdata[11]} Population: %{customdata[2]:,} (Rank %{customdata[6]} of %{customdata[9]})<br>"
    "Change: %{customdata[3]:,} (Rank %{customdata[7]} of %{customdata[9]})<br>"
    "Change %: %{customdata[4]:.2f}% (Rank %{customdata[8]} of %{customdata[9]})"
    "</span><extra></extra>"
)

    vmin = merged[metric_type].quantile(0.1)
    vmax = merged[metric_type].quantile(0.9)
    
    fig = px.choropleth(
        merged,
        geojson=counties_geojson,
        locations='FIPS',
        color=metric_type,
        featureidkey="properties.GEOID",
        color_continuous_scale='RdYlBu',
        range_color=[vmin, vmax],
        scope="usa",
        labels={metric_type: 'Change'}
    )
	
    fig.update_traces(
        customdata=merged[['county_state', 'Population_start', 'Population_end', 'numeric_diff', 'percent_diff',
                      'Population_start_rank', 'Population_end_rank', 'numeric_diff_rank','percent_diff_rank',
                      'total_county', 'start_year', 'end_year']],
        hovertemplate=hovertemplate
    )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    growing = merged[merged[metric_type] > 0].copy()
    declining = merged[merged[metric_type] < 0].copy()
    top10 = growing.nlargest(min(10, len(growing)), metric_type)
    bottom10 = declining.nsmallest(min(10, len(declining)), metric_type)
    top10.insert(0, '', range(1, len(top10) + 1))
    bottom10.insert(0, '', range(1, len(bottom10) + 1))

    columns = [
        {"name": "", "id": ""},
        {"name": "County", "id": "county_state"},
        {"name": f"{start_year} Population", "id": "Population_start"},
        {"name": f"{end_year} Population", "id": "Population_end"},
        {"name": "Change", "id": "numeric_diff"},
        {"name": "Change %", "id": "percent_diff"}
    ]

    for col in ['Population_start', 'Population_end', 'numeric_diff']:
        top10[col] = top10[col].apply(lambda x: f"{int(x):,}")
        bottom10[col] = bottom10[col].apply(lambda x: f"{int(x):,}")

    top10['percent_diff'] = top10['percent_diff'].apply(lambda x: f"{x:.2f}%")
    bottom10['percent_diff'] = bottom10['percent_diff'].apply(lambda x: f"{x:.2f}%")

    return summary, fig, top10.to_dict('records'), columns, bottom10.to_dict('records'), columns

# ----------------------------------------------------------------------------
# Run the app
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run()

# ----------------------------------------------------------------------------
# End of the program
# ----------------------------------------------------------------------------
print("\n==== Program Completed ====")
end_time_program = time.time()
print(f"Total time taken: {end_time_program - start_time_program:.2f} seconds")
