from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_nn_torch,
    train_linear_regression_std,
    train_random_forest_std,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_nn_torch,
            inputs="employee_dataset_features",
            outputs=[
                "nn_model",
                "nn_metrics",
                "nn_predictions",
                "nn_history",
                "nn_metadata",
            ],
            name="train_nn_torch_node_nonspark",
            tags=["nonspark"]
        ),
        node(
            func=train_linear_regression_std,
            inputs="employee_dataset_features",
            outputs=[
                "lr_model",
                "lr_metrics",
                "lr_predictions",
                "lr_history",
                "lr_metadata",
            ],
            name="train_linear_regression_node_nonspark",
        ),
        node(
            func=train_random_forest_std,
            inputs="employee_dataset_features",
            outputs=[
                "rf_model",
                "rf_metrics",
                "rf_predictions",
                "rf_history",
                "rf_metadata",
            ],
            name="train_random_forest_node_nonspark",
        ),
    ])
