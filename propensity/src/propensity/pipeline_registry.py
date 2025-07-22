from typing import Dict
from kedro.pipeline import Pipeline

from .pipelines.data_engineering import pipeline as data_engineering_pipeline
from .pipelines.eda import pipeline as eda_pipeline
from .pipelines.modeling import pipeline as modeling_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """
    Register the project's pipelines.

    Returns:
        A mapping from pipeline names to Pipeline objects.
    """
    data_eng = data_engineering_pipeline.create_pipeline()
    eda = eda_pipeline.create_pipeline()
    modeling = modeling_pipeline.create_pipeline()

    return {
        "__default__": data_eng + eda + modeling,  # Runs all steps in sequence
        "data_engineering": data_eng,
        "eda": eda,
        "modeling": modeling,
    }
