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
                "nn_X_train",
                "nn_X_test",
                "nn_y_train",
                "nn_y_test",
                "nn_job_train",
                "nn_job_test",
            ],
            name="train_nn_torch_node",
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
                "lr_splits",
            ],
            name="train_linear_regression_node",
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
                "rf_splits",
            ],
            name="train_random_forest_node",
        ),
    ])
