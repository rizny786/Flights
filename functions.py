import plotly.graph_objs as go
import streamlit as st

def by_airline(df):
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

def pie_chart_delayed_vs_non_delayed(df):
    # Calculate the number of delayed and non-delayed flights
    delayed_flights = df[df['ArrDelay'] > 0].shape[0]
    non_delayed_flights = df[df['ArrDelay'] <= 0].shape[0]

    # Calculate percentages
    total_flights = delayed_flights + non_delayed_flights
    delayed_percentage = (delayed_flights / total_flights) * 100
    non_delayed_percentage = (non_delayed_flights / total_flights) * 100

    # Create pie chart using plotly.graph_objs
    fig = go.Figure(data=[go.Pie(labels=['Delayed', 'Non-Delayed'],
                                 values=[delayed_percentage, non_delayed_percentage],
                                 hole=0.3)])
    
    fig.update_layout(title='Delayed vs Non-Delayed Flights')
    #                   annotations=[dict(text=f'Delayed: {delayed_percentage:.2f}%', showarrow=False),
    #                                dict(text=f'Non-Delayed: {non_delayed_percentage:.2f}%', showarrow=False)])

    return fig

def by_origin(df):
    # Group by Origin and calculate the number of delayed and non-delayed flights
    origin_stats = df.assign(Delayed=df['ArrDelay'] > 0) \
        .groupby('Origin')['Delayed'] \
        .value_counts() \
        .unstack(fill_value=0) \
        .rename(columns={True: 'Delayed', False: 'Non-Delayed'})

    # Calculate percentages
    origin_stats['Delayed-Percentage'] = (origin_stats['Delayed'] / (origin_stats['Delayed'] + origin_stats['Non-Delayed'])) * 100
    origin_stats['Non-Delayed-Percentage'] = (origin_stats['Non-Delayed'] / (origin_stats['Delayed'] + origin_stats['Non-Delayed'])) * 100

    # Create a grouped bar chart using plotly.graph_objs
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=origin_stats.index,
        y=origin_stats['Non-Delayed'],
        text=origin_stats['Non-Delayed-Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        name='Non-Delayed Flights',
        offsetgroup=1,
        marker=dict(color='limegreen')
    ))

    fig.add_trace(go.Bar(
        x=origin_stats.index,
        y=origin_stats['Delayed'],
        text=origin_stats['Delayed-Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        name='Delayed Flights',
        offsetgroup=2,
        marker=dict(color='orangered')
    ))

    fig.update_layout(
        title='Delayed and Non-Delayed Flights Grouped by Origin',
        xaxis=dict(title='Origin'),
        yaxis=dict(title='Number of Flights'),
        barmode='group',
        legend=dict(x=0.25, y=1.1, orientation='h')
    )

    return fig


def by_dest(df):
    # Group by Dest and calculate the number of delayed and non-delayed flights
    dest_stats = df.assign(Delayed=df['ArrDelay'] > 0) \
        .groupby('Dest')['Delayed'] \
        .value_counts() \
        .unstack(fill_value=0) \
        .rename(columns={True: 'Delayed', False: 'Non-Delayed'})

    # Calculate percentages
    dest_stats['Delayed-Percentage'] = (dest_stats['Delayed'] / (dest_stats['Delayed'] + dest_stats['Non-Delayed'])) * 100
    dest_stats['Non-Delayed-Percentage'] = (dest_stats['Non-Delayed'] / (dest_stats['Delayed'] + dest_stats['Non-Delayed'])) * 100

    # Create a grouped bar chart using plotly.graph_objs
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dest_stats.index,
        y=dest_stats['Non-Delayed'],
        text=dest_stats['Non-Delayed-Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        name='Non-Delayed Flights',
        offsetgroup=1,
        marker=dict(color='limegreen')
    ))

    fig.add_trace(go.Bar(
        x=dest_stats.index,
        y=dest_stats['Delayed'],
        text=dest_stats['Delayed-Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        name='Delayed Flights',
        offsetgroup=2,
        marker=dict(color='orangered')
    ))

    fig.update_layout(
        title='Delayed and Non-Delayed Flights Grouped by Destination',
        xaxis=dict(title='Destination'),
        yaxis=dict(title='Number of Flights'),
        barmode='group',
        legend=dict(x=0.25, y=1.1, orientation='h')
    )

    return fig
