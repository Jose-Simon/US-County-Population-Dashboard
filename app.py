# ----------------------------------------------------------------------------
# Start of the program
# ----------------------------------------------------------------------------

print("==== Program Start ====")

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import ctx, dcc, html, Output, Input, State, Dash, dash_table
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

county_to_fips = df[['FIPS', 'County', 'State']].drop_duplicates()
county_to_fips['county_state'] = county_to_fips['County'] + ", " + county_to_fips['State']
county_state_to_fips_map = dict(zip(county_to_fips['county_state'], county_to_fips['FIPS']))

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

app.layout = html.Div(style={'padding': '10px'}, children=[

    html.H1(id='dashboard-title', className="header-bar"),

    html.Div(className='filter-bar', children=[
        html.Div(className='filter-item', children=[
            html.Label("Change Type"),
            dcc.RadioItems(
                id='metric-radio',
                options=[
                    {'label': 'Absolute', 'value': 'numeric_diff'},
                    {'label': 'Percentage', 'value': 'percent_diff'},
                    {'label': 'Percentage (normalized)', 'value': 'percent_diff_normalized'}
                ],
                value='percent_diff',
                labelStyle={'display': 'inline-block', 'margin-right': '5px'}
            )
        ]),

        html.Div(className='filter-item', children=[
            html.Label("Start Year"),
            dcc.Dropdown(
                id='start-year-dropdown',
                options=[{'label': str(year), 'value': year} for year in range(2000, 2025)],
                value=2000,
                clearable=False
            )
        ]),

        html.Div(className='filter-item', children=[
            html.Label("End Year"),
            dcc.Dropdown(
                id='end-year-dropdown',
                options=[{'label': str(year), 'value': year} for year in range(2000, 2025)],
                value=2024,
                clearable=False
            )
        ]),

        html.Div(className='filter-item', children=[
            html.Label("State Filter"),
            dcc.Dropdown(
                id='state-filter-dropdown',
                options=state_options,
                multi=True,
                placeholder="Select states..."
            )
        ]),

        html.Div(className='filter-item', children=[
            html.Label("County Filter"),
            dcc.Dropdown(
                id='county-filter-dropdown',
                options=county_options,
                multi=True,
                placeholder="Select counties..."
            )
        ]),
		
		html.Div(className='filter-item', children=[
            html.Label("Population Group"),
            dcc.Dropdown(
                id='population-group-dropdown',
                options=population_groups,
                multi=True,
                placeholder="Select population group...",
                clearable=True
            )
        ])
    ]),

    html.Div(id="summary-banner", className="summary-container"),

    html.Div(style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'flex-start',
        'marginTop': '20px',
        'marginBottom': '20px'
    }, children=[
        html.Div([
            dcc.Graph(
                id="choropleth-map",
                style={'height': '800px', 'width': '100%'}
            )
        ], className='card choropleth-wrapper', style={'width': '74%'}),
        html.Div(id='county-detail-pane', className='card county-detail-pane', style={
            'width': '24%',
            'maxHeight': '100%',
            'overflowY': 'auto'
        }),
        dcc.Store(id='filtered-data')
    ]),

    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(className='card', style={'width': '48%'}, children=[
            html.H4("Top Growing Counties"),
            dash_table.DataTable(id='topcnt-table', fixed_rows={'headers': True},
                style_table={'height': '350px', 'overflowY': 'auto'},
                style_cell={'fontFamily': 'Roboto, Arial, Helvetica, sans-serif', 'fontSize': '14px', 'textAlign': 'right'},
                style_header={'fontFamily': 'Roboto, Arial, Helvetica, sans-serif', 'fontWeight': 'bold', 'backgroundColor': '#003366', 'color': 'white'},
                style_cell_conditional=[
                    {'if': {'column_id': 'county_state'}, 'textAlign': 'left', 'width': '300px', 'maxWidth': '400px'},
                ])
        ]),

        html.Div(className='card', style={'width': '48%'}, children=[
            html.H4("Top Declining Counties"),
            dash_table.DataTable(id='bottomcnt-table', fixed_rows={'headers': True},
                style_table={'height': '350px', 'overflowY': 'auto'},
                style_cell={'fontFamily': 'Roboto, Arial, Helvetica, sans-serif', 'fontSize': '14px', 'textAlign': 'right'},
                style_header={'fontFamily': 'Roboto, Arial, Helvetica, sans-serif', 'fontWeight': 'bold', 'backgroundColor': '#003366', 'color': 'white'},
                style_cell_conditional=[
                    {'if': {'column_id': 'county_state'}, 'textAlign': 'left', 'width': '300px', 'maxWidth': '400px'},
                ])
        ])
    ]),

    html.Footer(
        html.Div([
            html.Span(["Designed by ",
                html.Span("Jose Simon", style={'fontWeight': 'bold'}),
                " | Built with Python, Dash, and Plotly | Data Sources: Census.gov intercensal data files for "
            ]),
            html.A("2000-2010", href="https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-2000-2010-counties.html", target="_blank"),
            html.Span(", "),
            html.A("2010-2020", href="https://www.census.gov/data/tables/time-series/demo/popest/intercensal-2010-2020-counties.html", target="_blank"),
            html.Span(", "),
            html.A("2020-2024", href="https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html", target="_blank"),
            html.Span(" and Simon Frost "),
            html.A("Counties GeoJSON file", href="https://gist.github.com/sdwfrost/d1c73f91dd9d175998ed166eb216994a", target="_blank")
        ]),
        className='footer'
    )
])

# ----------------------------------------------------------------------------
# Callback for Dashboard title
# ----------------------------------------------------------------------------

@app.callback(
    Output('dashboard-title', 'children'),
    Input('start-year-dropdown', 'value'),
    Input('end-year-dropdown', 'value')
)
def update_title(start_year, end_year):
    return f"Population Change by US Counties ({start_year}-{end_year})"

# ----------------------------------------------------------------------------
# Callback for Choropleth map
# ----------------------------------------------------------------------------

@app.callback(
    Output('summary-banner', 'children'),
    Output('choropleth-map', 'figure'),
    Output('topcnt-table', 'data'),
    Output('topcnt-table', 'columns'),
    Output('bottomcnt-table', 'data'),
    Output('bottomcnt-table', 'columns'),
    Output('filtered-data', 'data'),
    Input('start-year-dropdown', 'value'),
    Input('end-year-dropdown', 'value'),
    Input('metric-radio', 'value'),
    Input('state-filter-dropdown', 'value'),
    Input('county-filter-dropdown', 'value'),
    Input('population-group-dropdown', 'value'),
    Input('choropleth-map', 'clickData'),
    Input('topcnt-table', 'active_cell'),
    Input('bottomcnt-table', 'active_cell'),
    State('topcnt-table', 'data'),
    State('bottomcnt-table', 'data')
)
def update_dashboard(start_year, end_year, metric_type, selected_states, selected_counties, selected_group, map_click, top_cell, bottom_cell, top_data, bottom_data):
    if metric_type == 'percent_diff_normalized':
        percent_diff_normalized = True
        metric_type = 'percent_diff'
    else:
        percent_diff_normalized = False
    
    dff = df.copy()

    selected_fips = None
    ctx = dash.callback_context

    if ctx.triggered_id == 'choropleth-map' and map_click:
        selected_fips = map_click['points'][0]['location']
    elif ctx.triggered_id == 'topcnt-table' and top_cell and top_data:
        label = top_data[top_cell['row']]['county_state']
        selected_fips = county_state_to_fips_map.get(label)
    elif ctx.triggered_id == 'bottomcnt-table' and bottom_cell and bottom_data:
        label = bottom_data[bottom_cell['row']]['county_state']
        selected_fips = county_state_to_fips_map.get(label)

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
    merged['selected'] = merged['FIPS'] == selected_fips

    merged['numeric_diff'] = merged['Population_end'] - merged['Population_start']
    merged['percent_diff'] = (merged['numeric_diff'] / merged['Population_start']) * 100
    merged['numeric_diff_fmt'] = merged['numeric_diff'].apply(lambda x: f"{x:+,}" if pd.notnull(x) else "")
    merged['percent_diff_fmt'] = merged['percent_diff'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "")
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
    pop_change = total_end_pop - total_start_pop
    percent_change_total = (total_end_pop - total_start_pop) / total_start_pop * 100
    if percent_change_total >= 0:
        arrow = "\u25B2"
        color = "green"
    else:
        arrow = "\u25BC"
        color = "red"

    # County stats
    county_count = len(merged)
    increasing_count = (merged[metric_type] > 0).sum()
    decreasing_count = (merged[metric_type] <= 0).sum()

    # State stats
    state_summary = merged.groupby('State').agg({
        'Population_start': 'sum',
        'Population_end': 'sum'
    }).reset_index()

    state_summary['is_increasing'] = state_summary['Population_end'] > state_summary['Population_start']

    states_count = len(state_summary)
    states_increasing_count = state_summary['is_increasing'].sum()
    states_decreasing_count = states_count - states_increasing_count
    
    # Summary banner calculation
    summary = [
        html.Div([
            html.H4(f"{start_year} Population"),
            html.H2(f"{total_start_pop:,}")
        ], className="summary-card"),

        html.Div([
            html.H4(f"{end_year} Population"),
            html.H2(f"{total_end_pop:,}")
        ], className="summary-card"),

        html.Div([
            html.H4("Population Change"),
            html.H2([
                f"{pop_change:,} ({percent_change_total:.2f}%",
                html.Span(arrow, className="change-arrow"),
                ")"
            ])
        ], className="summary-card"),

        html.Div([
            html.H4("States Displayed"),
            html.H2([
                f"{states_count:,} (",
                f"{states_increasing_count:,}",
                html.Span("▲", className="change-up"),
                "   ",
                f"{states_decreasing_count:,}",
                html.Span("▼", className="change-down"),
                ")"
            ])
        ], className="summary-card"),

        html.Div([
            html.H4("Counties Displayed"),
            html.H2([
                f"{county_count:,} (",
                f"{increasing_count:,}",
                html.Span("▲", className="change-up"),
                "   ",
                f"{decreasing_count:,}",
                html.Span("▼", className="change-down"),
                ")"
            ])
        ], className="summary-card")
    ]

    hovertemplate = (
        "<b>%{customdata[0]}</b><br><br>"
        "<span style='font-size: 12px;'>"
        "%{customdata[10]} Population: <b>%{customdata[1]:,}</b> <span style='font-size: 10px; color: #cccccc;'>(Rank <b>%{customdata[5]}</b> of %{customdata[9]})</span><br>"
        "%{customdata[11]} Population: <b>%{customdata[2]:,}</b> <span style='font-size: 10px; color: #cccccc;'>(Rank <b>%{customdata[6]}</b>)</span><br>"
        "Change: <b>%{customdata[3]}</b> <span style='font-size: 10px; color: #cccccc;'>(Rank <b>%{customdata[7]}</b>)</span><br>"
        "Change %: <b>%{customdata[4]}</b> <span style='font-size: 10px; color: #cccccc;'>(Rank <b>%{customdata[8]}</b>)</span>"
        "</span><extra></extra>"
    )

    if percent_diff_normalized == True:
        # Use percent_diff values, but center color scale around overall percent change
        center = percent_change_total
        q = merged['percent_diff'].quantile([0.1, 0.9])
        q_max = max(abs(q[0.1] - center), abs(q[0.9] - center))
        color_field = 'percent_diff'
        range_min, range_max = center - q_max, center + q_max

    elif metric_type == 'percent_diff':
        q = merged['percent_diff'].quantile([0.1, 0.9])
        q_max = max(abs(q[0.1]), abs(q[0.9]))
        color_field = 'percent_diff'
        range_min, range_max = -q_max, q_max

    else:  # numeric_diff
        q = merged['numeric_diff'].quantile([0.1, 0.9])
        q_max = max(abs(q[0.1]), abs(q[0.9]))
        color_field = 'numeric_diff'
        range_min, range_max = -q_max, q_max

    fig = px.choropleth(
        merged,
        geojson=counties_geojson,
        locations='FIPS',
        color=metric_type,
        featureidkey="properties.GEOID",
        color_continuous_scale='RdBu', #RdYlBu
        range_color=[range_min, range_max],
        scope="usa",
        labels={metric_type: 'Change'}
    )
	
    fig.update_traces(
        marker_line_width=merged['selected'].apply(lambda x: 10 if x else 0.3),
        marker_line_color=merged['selected'].apply(lambda x: '#66FF00' if x else 'gray'),
        colorbar=dict(
            title=dict(font=dict(family="Roboto, Arial, Helvetica, sans-serif", color="#333333", size=14)),
            tickfont=dict(family="Roboto, Arial, Helvetica, sans-serif", color="#333333", size=12)
        ),
        customdata=merged[['county_state', 'Population_start', 'Population_end', 'numeric_diff_fmt', 'percent_diff_fmt',
                      'Population_start_rank', 'Population_end_rank', 'numeric_diff_rank','percent_diff_rank',
                      'total_county', 'start_year', 'end_year']],
        hovertemplate=hovertemplate
    )

    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(
            title=dict(text='Change', font=dict(family="Roboto, Arial, Helvetica, sans-serif", size=14, color="#333333")),
            tickfont=dict(family="Roboto, Arial, Helvetica, sans-serif", size=12, color="#333333")
        )
    )

    growing = merged[merged[metric_type] > 0].copy()
    declining = merged[merged[metric_type] < 0].copy()
    topcnt = growing.nlargest(min(100, len(growing)), metric_type)
    bottomcnt = declining.nsmallest(min(100, len(declining)), metric_type)
    topcnt.insert(0, '', range(1, len(topcnt) + 1))
    bottomcnt.insert(0, '', range(1, len(bottomcnt) + 1))

    columns = [
        {"name": "", "id": ""},
        {"name": "County", "id": "county_state"},
        {"name": f"{start_year} Population", "id": "Population_start"},
        {"name": f"{end_year} Population", "id": "Population_end"},
        {"name": "Change", "id": "numeric_diff"},
        {"name": "Change %", "id": "percent_diff"}
    ]

    for col in ['Population_start', 'Population_end', 'numeric_diff']:
        topcnt[col] = topcnt[col].apply(lambda x: f"{int(x):,}")
        bottomcnt[col] = bottomcnt[col].apply(lambda x: f"{int(x):,}")

    topcnt['percent_diff'] = topcnt['percent_diff'].apply(lambda x: f"{x:.2f}%")
    bottomcnt['percent_diff'] = bottomcnt['percent_diff'].apply(lambda x: f"{x:.2f}%")

    return summary, fig, topcnt.to_dict('records'), columns, bottomcnt.to_dict('records'), columns, merged.to_dict('records')

# ----------------------------------------------------------------------------
# Callback for County detail
# ----------------------------------------------------------------------------

@app.callback(
    Output('county-detail-pane', 'children'),
    Input('choropleth-map', 'clickData'),
    Input('topcnt-table', 'active_cell'),
    Input('bottomcnt-table', 'active_cell'),
    Input('start-year-dropdown', 'value'),
    Input('end-year-dropdown', 'value'),
    Input('filtered-data', 'data'),
    State('topcnt-table', 'data'),
    State('bottomcnt-table', 'data')
)
def update_county_detail(map_click, top_cell, bottom_cell, start_year, end_year, filtered_data, top_data, bottom_data):
    fips = None
    label = None
    triggered = ctx.triggered_id

    if triggered == 'choropleth-map' and map_click:
        fips = map_click['points'][0]['location']
    elif triggered == 'topcnt-table' and top_cell and top_data:
        label = top_data[top_cell['row']]['county_state']
    elif triggered == 'bottomcnt-table' and bottom_cell and bottom_data:
        label = bottom_data[bottom_cell['row']]['county_state']

    print("Clicked label from table:", label)
    if label:
        # Extract FIPS from label if needed using mapping
        fips = county_state_to_fips_map.get(label)

    if not fips:
       return "Click on a county in the map or table to view details."

    dff = df[df['FIPS'] == fips].sort_values(by='Year')	
    dff = dff[(dff['Year'] >= start_year) & (dff['Year'] <= end_year)].copy()

    if dff.empty:
       return "No data available for selected county."

    county_name = f"{dff.iloc[0]['County']}, {dff.iloc[0]['State']}"
    dff['county_state'] = county_name

    start_pop = dff.iloc[0]['Population']
    # dff['Change %'] = 100 * (dff['Population'] - start_pop) / start_pop
    dff['YoY'] = dff['Population'].diff().astype(float)
    dff['YoY %'] = dff['Population'].pct_change().astype(float) * 100
    dff['YoY_fmt'] = dff['YoY'].apply(lambda x: f"{int(x):+,}" if pd.notnull(x) else "")
    dff['YoY_pct_fmt'] = dff['YoY %'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "")
    dff['Year Label'] = dff['Year'].astype(str)
    dff['Year Label Short'] = "'" + dff['Year'].astype(str).str[-2:]

    hovertemplate=(
        "<b>%{customdata[0]}</b><br><br>"
        "%{customdata[1]} Population: <b>%{customdata[2]:,}</b><br>"
        "Change from prior year: <b>%{customdata[3]}</b><br>"
        "Change % from prior year: <b>%{customdata[4]}</b><extra></extra>"
    )

    max_val = dff['YoY %'].max()
    min_val = dff['YoY %'].min()
    y_range = 1.1 * max(abs(max_val), abs(min_val))
	
    filtered_df = pd.DataFrame(filtered_data)
    total = len(filtered_df)

    rank_start = filtered_df.set_index('FIPS')['Population_start'].rank(ascending=False, method='min').astype(int).get(fips)
    rank_end = filtered_df.set_index('FIPS')['Population_end'].rank(ascending=False, method='min').astype(int).get(fips)
    
    df_change = pd.DataFrame(filtered_data).set_index('FIPS')
    df_change['numeric_diff'] = df_change['Population_end'] - df_change['Population_start']
    df_change['percent_diff'] = (df_change['numeric_diff'] / df_change['Population_start']) * 100

    rank_diff = df_change['numeric_diff'].rank(ascending=False, method='min').astype(int).get(fips, None)
    rank_pct = df_change['percent_diff'].rank(ascending=False, method='min').astype(int).get(fips, None)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=dff['Year Label Short'],
        y=dff['YoY %'],
        marker_color=['#003366' if x >= 0 else '#CC3300' for x in dff['YoY %']]
    ))
    bar_fig.update_layout(
        title=dict(
            text="<b>Population % Change from Prior Year<b>",
            font=dict(color="#333333", family="Roboto, Arial, Helvetica, sans-serif")
        ),
        font=dict(family="Roboto, Arial, Helvetica, sans-serif", color="#333333"),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(
            title=dict(
                text='% Change',
                font=dict(color="#333333", family="Roboto, Arial, Helvetica, sans-serif")
            ),
            tickfont=dict(color="#333333", family="Roboto, Arial, Helvetica, sans-serif", size=10),
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            range=[-y_range, y_range]
        ),
        xaxis=dict(
            title=dict(
                text='Year',
                font=dict(color="#333333", family="Roboto, Arial, Helvetica, sans-serif")
            ),
            tickfont=dict(color="#333333", family="Roboto, Arial, Helvetica, sans-serif", size=10),
            tickangle=0
        ),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        hoverlabel=dict(
            bgcolor="rgba(50, 50, 50, 0.9)",  # same as choropleth map
            font=dict(color="white"),
            bordercolor="rgba(200, 200, 200, 0.5)"
        )
    )

    latest_row = dff.iloc[-1]
    pop_latest = latest_row['Population']
    change_pct = 100 * (pop_latest - start_pop) / start_pop
    change_raw = pop_latest - start_pop

    bar_fig.update_traces(
        customdata=dff[['county_state', 'Year Label', 'Population', 'YoY_fmt', 'YoY_pct_fmt']],
        hovertemplate=hovertemplate
    )

    # DEBUG
    # debug_text = html.Pre([
    #     f"FIPS: {fips}\n",
    #     f"map_click: {map_click}\n",
    #    f"top_cell: {top_cell}\n",
    #    f"bottom_cell: {bottom_cell}\n",
    #    f"start_year: {start_year}, end_year: {end_year}\n",
    #    f"filtered_data length: {len(filtered_data) if filtered_data else 0}\n",
    #])

    return [
        html.Div([
            html.H4(county_name, style={'fontWeight': 'bold', 'marginBottom': '10px'}),

            html.Div([
                html.Span(f"{start_year} Population: ", style={'fontSize': '16px'}),
                html.B(f"{start_pop:,}", style={'fontSize': '16px'}),
                html.Span([
                    " (Rank ",
                    html.B(f"{rank_start}", style={'fontWeight': 'bold'}),
                    f" of {total})"
                ], style={'fontSize': '14px', 'color': '#555555'})
            ]),

            html.Div([
                html.Span(f"{end_year} Population: ", style={'fontSize': '16px'}),
                html.B(f"{pop_latest:,}", style={'fontSize': '16px'}),
                html.Span([
                    " (Rank ",
                    html.B(f"{rank_end}", style={'fontWeight': 'bold'}),
                    ")"
                ], style={'fontSize': '14px', 'color': '#555555'})
            ]),

            html.Div([
                html.Span("Change: ", style={'fontSize': '16px'}),
                html.B(f"{change_raw:+,}", style={'fontSize': '16px'}),
                html.Span([
                    " (Rank ",
                    html.B(f"{rank_diff}", style={'fontWeight': 'bold'}),
                    ")"
                ], style={'fontSize': '14px', 'color': '#555555'})
            ]),

            html.Div([
                html.Span("Change %: ", style={'fontSize': '16px'}),
                html.B(f"{change_pct:+.2f}%", style={'fontSize': '16px'}),
                html.Span([
                    " (Rank ",
                    html.B(f"{rank_pct}", style={'fontWeight': 'bold'}),
                    ")"
                ], style={'fontSize': '14px', 'color': '#555555'})
            ]),

            #html.Div([
            #    html.H4("Debug Info (Render)", style={'color': 'red'}),
            #    debug_text
            #]),

            html.Div(style={'marginBottom': '15px'})
        ]),
        dcc.Graph(figure=bar_fig, config={'displayModeBar': False})
    ]

# ----------------------------------------------------------------------------
# Run the app
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run

# ----------------------------------------------------------------------------
# End of the program
# ----------------------------------------------------------------------------
print("\n==== Program Completed ====")
end_time_program = time.time()
print(f"Total time taken: {end_time_program - start_time_program:.2f} seconds")