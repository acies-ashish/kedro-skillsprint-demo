from kedro.pipeline import Pipeline, node
from .nodes import eda_report, plot_target_distribution, plot_feature_correlations, plot_feature_vs_target

def create_pipeline(**kwargs):
    return Pipeline([
        node(func=eda_report, inputs="cleaned_data", outputs="eda_report", name="eda_report_node"),
        node(func=plot_target_distribution, inputs="cleaned_data", outputs="target_dist_plot", name="target_dist_plot_node"),
        node(func=plot_feature_correlations, inputs="cleaned_data", outputs="feature_corr_plot", name="feature_corr_plot_node"),
        node(func=plot_feature_vs_target, inputs="cleaned_data", outputs="feature_vs_target_plot", name="feature_vs_target_plot_node"),
    ])
