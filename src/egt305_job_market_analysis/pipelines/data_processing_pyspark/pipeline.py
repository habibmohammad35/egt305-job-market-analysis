from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    load_shuttles_to_csv,
    preprocess_companies,
    preprocess_reviews,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        ]
    )
