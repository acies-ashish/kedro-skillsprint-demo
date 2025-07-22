from kedro.pipeline import Pipeline, node
from .nodes import train_model, evaluate_model, feature_importance_plot

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_model,
            inputs=dict(
                features_train="features_train",
                input_features="params:input_features",
                target="params:target"
            ),
            outputs="model",
            name="train_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=dict(
                model="model",
                features_test="features_test",
                input_features="params:input_features",
                target="params:target"
            ),
            outputs="evaluation",
            name="evaluate_model_node"
        ),
        node(
            func=feature_importance_plot,
            inputs=dict(
                model="model",
                feature_names="params:input_features"
            ),
            outputs="feature_importance_plot",
            name="feature_importance_plot_node"
        ),
    ])
