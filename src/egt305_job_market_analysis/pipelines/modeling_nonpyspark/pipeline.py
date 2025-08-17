from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_and_encode_model_data, train_linear_torch, train_nn_torch

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_and_encode_model_data,
            inputs="employee_salary_clean",   
            outputs=["X_train", "X_test", "y_train", "y_test", "job_train", "job_test"],
            name="split_encode_node"
        ),
        node(
            func=train_linear_torch,
            inputs=["X_train", "y_train", "X_test", "y_test", "job_test"],
            outputs=["linear_model", "linear_metrics", "linear_predictions"],
            name="train_linear_torch_node"
        ),
        node(
            func=train_nn_torch,
            inputs=["X_train", "y_train", "X_test", "y_test", "job_test"],
            outputs=["nn_model", "nn_metrics", "nn_predictions"],
            name="train_nn_torch_node"
        )
    ])
