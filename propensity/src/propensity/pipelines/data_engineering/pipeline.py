from kedro.pipeline import Pipeline, node

from .nodes import clean_data, split_dataset

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=clean_data,
            inputs="raw_data",
            outputs="cleaned_data",
            name="clean_data_node"
        ),
        node(
            func=split_dataset,
            inputs=dict(
                data="cleaned_data",
                split_ratio="params:split_ratio",
                random_state="params:random_state",
                input_features="params:input_features",
                target="params:target"
            ),
            outputs=["features_train", "features_test"],
            name="split_data_node"
        ),
    ])
