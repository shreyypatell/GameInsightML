import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import plotly.utils

COLORS = ['#23395B', '#D97B66', '#6C8A5A', '#F4E9D8', '#444444', '#8AA6C1', '#E3A98F']


def fig_to_json(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Poppins, sans-serif', color='#444444'),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))


def genre_distribution(df):
    counts = df['Genre'].value_counts().reset_index()
    counts.columns = ['Genre', 'Count']
    fig = px.bar(counts, x='Genre', y='Count', color='Genre', color_discrete_sequence=COLORS)
    return fig_to_json(fig)


def platform_distribution(df):
    counts = df['Platform'].value_counts().head(15).reset_index()
    counts.columns = ['Platform', 'Count']
    fig = px.bar(counts, x='Platform', y='Count', color_discrete_sequence=[COLORS[1]])
    return fig_to_json(fig)


def publisher_distribution(df):
    counts = df['Publisher'].value_counts().head(15).reset_index()
    counts.columns = ['Publisher', 'Count']
    fig = px.bar(counts, x='Count', y='Publisher', orientation='h', color_discrete_sequence=[COLORS[2]])
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    return fig_to_json(fig)


def yearwise_releases(df):
    counts = df.groupby('Year').size().reset_index(name='Count')
    fig = px.line(counts, x='Year', y='Count', markers=True, color_discrete_sequence=[COLORS[0]])
    return fig_to_json(fig)


def global_sales_distribution(df, target):
    fig = px.histogram(df, x=target, nbins=40, color_discrete_sequence=[COLORS[1]])
    return fig_to_json(fig)


def top_selling_games(df, target, name_col='Name'):
    if name_col not in df.columns:
        return None
    top = df.nlargest(10, target)[[name_col, target]]
    fig = px.bar(top, x=target, y=name_col, orientation='h', color_discrete_sequence=[COLORS[0]])
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    return fig_to_json(fig)


def top_publishers_by_sales(df, target):
    grouped = df.groupby('Publisher')[target].sum().nlargest(10).reset_index()
    fig = px.bar(grouped, x=target, y='Publisher', orientation='h', color_discrete_sequence=[COLORS[2]])
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    return fig_to_json(fig)


def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0, '#F4E9D8'], [0.5, '#D97B66'], [1, '#23395B']],
        zmin=-1, zmax=1
    ))
    return fig_to_json(fig)


def boxplot_by_genre(df, target):
    fig = px.box(df, x='Genre', y=target, color='Genre', color_discrete_sequence=COLORS)
    return fig_to_json(fig)


def histogram_numeric(df, column):
    fig = px.histogram(df, x=column, nbins=30, color_discrete_sequence=[COLORS[0]])
    return fig_to_json(fig)


def pairplot_subset(df, target):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    subset_cols = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
    fig = px.scatter_matrix(df, dimensions=subset_cols, color_discrete_sequence=[COLORS[0]])
    return fig_to_json(fig)


def dataset_overview(df):
    overview = {
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
        'missing': df.isnull().sum().to_dict(),
        'describe': json.loads(df.describe(include='all').fillna('').to_json())
    }
    return overview
