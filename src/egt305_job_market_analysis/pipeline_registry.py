"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from egt305_job_market_analysis.pipelines import data_processing_nonpyspark, data_processing_pyspark, modeling_nonpyspark, modeling_pyspark, finetune


def register_pipelines() -> dict[str, Pipeline]:
    """Manually register the project's pipelines."""

    # Individual pipelines
    nonpyspark_data_pipeline = data_processing_nonpyspark.create_pipeline()
    pyspark_data_pipeline = data_processing_pyspark.create_pipeline()
    modeling_nonspark_pipeline = modeling_nonpyspark.create_pipeline()
    modeling_spark_pipeline = modeling_pyspark.create_pipeline()
    finetuning = finetune.create_pipeline()

    # You can add more later (feature_engineering, modelling, etc.)
    return {
        # Run both by default
        "__default__": nonpyspark_data_pipeline + pyspark_data_pipeline + modeling_spark_pipeline + modeling_nonspark_pipeline + finetuning,

        # Named entries so to run each separately
        "nonspark_full": nonpyspark_data_pipeline + modeling_nonspark_pipeline, 
        "nonspark_modeling": modeling_nonspark_pipeline,
        "data_processing_nonpyspark": nonpyspark_data_pipeline,
        "spark_full": pyspark_data_pipeline + modeling_spark_pipeline, 
        "data_processing_pyspark": pyspark_data_pipeline,
        "spark_modeling": modeling_spark_pipeline,
        "finetune":finetuning
    }
