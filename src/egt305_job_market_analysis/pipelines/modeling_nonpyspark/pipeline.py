from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_and_encode_model_data, train_linear_model, train_randomforest_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # --- Step 1: Split & Encode ---
        node(
            func=split_and_encode_model_data,
            inputs="employee_salary_clean",   # input dataset from catalog
            outputs=dict(
                X_train="X_train",
                X_test="X_test",
                y_train="y_train",
                y_test="y_test",
                job_train="job_train",
                job_test="job_test",
                model_input_feature_names="model_input_feature_names"
            ),
            name="split_encode_data_node"
        ),

        # --- Step 2a: Train Linear Regression ---
        node(
            func=train_linear_model,
            inputs=[
                "X_train", "X_test", "y_train", "y_test", "job_test",
                "model_input_feature_names",
                "params:skip_linear"
            ],
            outputs=["linear_model", "linear_metrics", "linear_predictions"],
            name="train_linear_node"
        ),

        # --- Step 2b: Train Random Forest Regressor ---
        node(
            func=train_randomforest_model,
            inputs=[
                "X_train", "X_test", "y_train", "y_test", "job_test",
                "model_input_feature_names",
                "params:rf_n_estimators",
                "params:rf_max_depth",
                "params:skip_rf"
            ],
            outputs=["randomforest_model", "randomforest_metrics", "randomforest_predictions"],
            name="train_randomforest_node"
        )
    ])
