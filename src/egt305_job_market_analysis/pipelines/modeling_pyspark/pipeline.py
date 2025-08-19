from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (train_nn_torch_spark)


def create_pipeline(**kwargs) -> Pipeline:
   
    return pipeline(
        [
            node(
                func=train_nn_torch_spark,
                inputs=dict(
                    data="employee_features_spark",
                    cat_cols="params:modeling_pyspark.cat_cols",
                    test_size="params:modeling_pyspark.test_size",
                    random_state="params:modeling_pyspark.random_state",
                    batch_size="params:modeling_pyspark.batch_size",
                    lr="params:modeling_pyspark.lr",
                    epochs="params:modeling_pyspark.epochs",
                    patience="params:modeling_pyspark.patience",
                ),
                outputs=[
                    "nn_model_spark",       # PyTorch state_dict (pickle)
                    "nn_metrics_spark",     # JSON dict
                    "nn_predictions_spark", # pandas DataFrame (CSV)
                    "nn_history_spark",     # JSON dict
                    "nn_metadata_spark",    # JSON dict (includes shard/test paths)
                ],
                name="train_nn_torch_cluster_node",
                tags=["spark"]
            ),
        ]
    )
