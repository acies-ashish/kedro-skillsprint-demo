import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def eda_report(data: pd.DataFrame):
    describe_raw = data.describe().to_dict()
    def summarize_stats(d):
        return {k: {stat: d[k][stat] for stat in ['mean', 'std', 'min', 'max'] if stat in d[k]} for k in d}

    return {
        "shape": data.shape,
        "columns": data.columns.tolist(),
        "dtypes": data.dtypes.apply(str).to_dict(),
        "nulls": data.isnull().sum().to_dict(),
        "target_counts": data['ordered'].value_counts(normalize=True).to_dict(),
        "describe_summary": summarize_stats(describe_raw),
        "feature_means": data.drop(columns=['UserID', 'ordered']).mean().to_dict(),
    }

def plot_target_distribution(data: pd.DataFrame):
    fig = px.histogram(data, x='ordered', title='Distribution of Target (ordered)')
    fig.update_layout(bargap=0.2)
    return fig

def plot_feature_correlations(data: pd.DataFrame):
    corr = data.drop(columns=['UserID']).corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title='Correlation')
    ))
    fig.update_layout(title='Feature Correlation Heatmap', height=800)
    return fig

def plot_feature_vs_target(data: pd.DataFrame):
    means = data.drop(columns=['UserID']).groupby('ordered').mean().T
    means['0'] = means.get(0, pd.Series([0]*len(means), index=means.index))
    means['1'] = means.get(1, pd.Series([0]*len(means), index=means.index))

    fig = go.Figure()
    fig.add_trace(go.Bar(name='No Purchase (0)', x=means.index, y=means['0']))
    fig.add_trace(go.Bar(name='Purchase (1)', x=means.index, y=means['1']))

    fig.update_layout(
        title='Mean of Features Grouped by Target (ordered)',
        xaxis_title='Features',
        yaxis_title='Mean Value',
        barmode='group',
        height=600
    )
    return fig
