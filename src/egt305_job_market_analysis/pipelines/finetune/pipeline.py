from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    finetune_nn_torch_nonspark_grid,
    finetune_nn_torch_spark_grid,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Fine-tune NONSPARK model
        node(
            func=finetune_nn_torch_nonspark_grid,
            inputs=[
                "employee_dataset_features",        # your pandas dataset
                "nn_model",                 # base model state dict
                "nn_metadata",              # base metadata
                "params:modeling_nonspark.param_grid",  # param grid dict
            ],
            outputs=[
                "nn_model_nonspark_finetuned",
                "nn_metrics_nonspark_finetuned",
                "nn_predictions_nonspark_finetuned",
                "nn_history_nonspark_finetuned",
                "nn_metadata_nonspark_finetuned",
            ],
            name="finetune_nonspark_node",
            tags=["nonspark"]
        ),

        # Fine-tune SPARK model
        node(
            func=finetune_nn_torch_spark_grid,
            inputs=[
                "employee_features_spark",           # your Spark dataset
                "nn_model_spark",                    # base model state dict
                "nn_metadata_spark",                 # base metadata
                "params:modeling_pyspark.param_grid",
            ],
            outputs=[
                "nn_model_spark_finetuned",
                "nn_metrics_spark_finetuned",
                "nn_predictions_spark_finetuned",
                "nn_history_spark_finetuned",
                "nn_metadata_spark_finetuned",
            ],
            name="finetune_spark_node",
            tags=["spark"]
        ),
    ])
