import plotly.graph_objs as go
import pandas as pd

def by_airline(df):
    # Group by airline and calculate the number of delayed and non-delayed flights
    # Group by airline and calculate the number of delayed and non-delayed flights
    delayed_flights = df[df['ArrDelay'] > 0].groupby('UniqueCarrier').size().rename('Delayed')
    non_delayed_flights = df[df['ArrDelay'] <= 0].groupby('UniqueCarrier').size().rename('Non-Delayed')

    # Combine delayed and non-delayed counts into a single DataFrame
    airline_stats = pd.concat([delayed_flights, non_delayed_flights], axis=1, sort=False).fillna(0)


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

    # Calculate percentages if total_flights is not zero
    total_flights = delayed_flights + non_delayed_flights
    delayed_percentage = (delayed_flights / total_flights) * 100 if total_flights != 0 else 0
    non_delayed_percentage = (non_delayed_flights / total_flights) * 100 if total_flights != 0 else 0

    
    # Create pie chart using plotly.graph_objs
    fig = go.Figure(data=[go.Pie(labels=['Delayed', 'Non-Delayed'],
                                 values=[delayed_percentage, non_delayed_percentage],
                                 hole=0.3)])
    
    fig.update_layout(title='Delayed vs Non-Delayed Flights')

    return fig


def by_origin(df):
    # Group by Origin and calculate the number of delayed and non-delayed flights
    delayed_flights = df[df['ArrDelay'] > 0].groupby('Origin').size().rename('Delayed')
    non_delayed_flights = df[df['ArrDelay'] <= 0].groupby('Origin').size().rename('Non-Delayed')

    # Combine delayed and non-delayed counts into a single DataFrame
    origin_stats = pd.concat([delayed_flights, non_delayed_flights], axis=1, sort=False).fillna(0)

    # Calculate percentages
    total_flights = origin_stats['Delayed'] + origin_stats['Non-Delayed']
    origin_stats['Delayed-Percentage'] = (origin_stats['Delayed'] / total_flights) * 100
    origin_stats['Non-Delayed-Percentage'] = (origin_stats['Non-Delayed'] / total_flights) * 100

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
    delayed_flights = df[df['ArrDelay'] > 0].groupby('Dest').size().rename('Delayed')
    non_delayed_flights = df[df['ArrDelay'] <= 0].groupby('Dest').size().rename('Non-Delayed')

    # Combine delayed and non-delayed counts into a single DataFrame
    dest_stats = pd.concat([delayed_flights, non_delayed_flights], axis=1, sort=False).fillna(0)

    # Calculate percentages
    total_flights = dest_stats['Delayed'] + dest_stats['Non-Delayed']
    dest_stats['Delayed-Percentage'] = (dest_stats['Delayed'] / total_flights) * 100
    dest_stats['Non-Delayed-Percentage'] = (dest_stats['Non-Delayed'] / total_flights) * 100

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

def flight_trend_by_airline(dataframe):
    # Group data by 'UniqueCarrier' and 'Month' and count the number of flights
    flight_counts = dataframe.groupby(['UniqueCarrier', 'Month']).size().reset_index(name='FlightCount')

    # Create traces for all airlines
    traces = []
    airlines = flight_counts['UniqueCarrier'].unique()
    for airline in airlines:
        airline_data = flight_counts[flight_counts['UniqueCarrier'] == airline]
        trace = go.Scatter(
            x=airline_data['Month'],
            y=airline_data['FlightCount'],
            mode='lines+markers',
            name=f'Flight Trend for {airline}',
            marker=dict()
        )
        traces.append(trace)

    # Create layout for the chart
    layout = go.Layout(
        title='Flight Trend by Airline',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Number of Flights')
    )

    # Create figure with data and layout
    fig = go.Figure(data=traces, layout=layout)

    return fig


def plot_feature_importance(combined_feature_importance):
    data = []
    models = combined_feature_importance['Model'].unique()

    for model in models:
        model_data = combined_feature_importance[combined_feature_importance['Model'] == model]
        data.append(go.Bar(x=model_data['Feature'], y=model_data['Importance'], name=model))

    layout = go.Layout(
        title='Feature Importance Comparison',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Importance'),
        barmode='group',
        xaxis_tickangle=-45  # Rotate x-axis labels for better readability
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_model_accuracy(accuracies):
    models = ['Decision Tree', 'XGBoost']
    colors = ['blue', 'green']
    
    data = [go.Bar(x=models, y=accuracies, marker=dict(color=colors))]
    
    layout = go.Layout(
        title='Model Accuracy Comparison',
        xaxis=dict(title='Models'),
        yaxis=dict(title='Accuracy'),
        yaxis_range=[0, 1]  # Set the y-axis range to match accuracy values (0-1)
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig