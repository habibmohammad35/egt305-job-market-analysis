"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from egt305_job_market_analysis.pipelines import data_processing_nonpyspark, data_processing_pyspark, modeling_nonpyspark


def register_pipelines() -> dict[str, Pipeline]:
    """Manually register the project's pipelines."""

    # Individual pipelines
    nonpyspark_pipeline = data_processing_nonpyspark.create_pipeline()
    pyspark_pipeline = data_processing_pyspark.create_pipeline()
    modeling_nonspark_pipeline = modeling_nonpyspark.create_pipeline()

    # You can add more later (feature_engineering, modelling, etc.)
    return {
        # Run both by default
        "__default__": nonpyspark_pipeline + pyspark_pipeline,

        # Named entries so to run each separately
        "nonspark": nonpyspark_pipeline + modeling_nonspark_pipeline, 
        "nonspark_modeling": modeling_nonspark_pipeline,
        "data_processing_pyspark": pyspark_pipeline,
    }
