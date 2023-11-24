import plotly.graph_objs as go
import streamlit as st

def delayed_vs_non_delayed_flights_chart(df):
    # Group by airline and calculate the number of delayed and non-delayed flights
    airline_stats = df.assign(Delayed=df['ArrDelay'] > 0) \
    .groupby('UniqueCarrier')['Delayed'] \
    .value_counts() \
    .unstack(fill_value=0) \
    .rename(columns={True: 'Delayed', False: 'Non-Delayed'})


    # Calculate percentages
    airline_stats['Delayed-Percentage'] = (airline_stats['Delayed'] / (airline_stats['Delayed'] + airline_stats['Non-Delayed'])) * 100
    airline_stats['Non-Delayed-Percentage'] = (airline_stats['Non-Delayed'] / (airline_stats['Delayed'] + airline_stats['Non-Delayed'])) * 100

    # Create a grouped bar chart using plotly.graph_objs
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=airline_stats.index,
        y=airline_stats['Non-Delayed'],
        text=airline_stats['Non-Delayed-Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        name='Non-Delayed Flights',
        offsetgroup=1,
        marker=dict(color='limegreen')
    ))

    fig.add_trace(go.Bar(
        x=airline_stats.index,
        y=airline_stats['Delayed'],
        text=airline_stats['Delayed-Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        name='Delayed Flights',
        offsetgroup=2,
        marker=dict(color='orangered')
    ))

    fig.update_layout(
        title='Delayed and Non-Delayed Flights per Airline',
        xaxis=dict(title='Airline'),
        yaxis=dict(title='Number of Flights'),
        barmode='group',
        legend=dict(x=0.25, y=1.1, orientation='h')
    )

    return fig
