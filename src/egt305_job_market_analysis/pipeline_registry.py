"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from egt305_job_market_analysis.pipelines import data_processing_nonpyspark, data_processing_pyspark 


def register_pipelines() -> dict[str, Pipeline]:
    """Manually register the project's pipelines."""

    # Individual pipelines
    nonpyspark_pipeline = data_processing_nonpyspark.create_pipeline()
    pyspark_pipeline = data_processing_pyspark.create_pipeline()

    # You can add more later (feature_engineering, modelling, etc.)
    return {
        # Run both by default
        "__default__": nonpyspark_pipeline + pyspark_pipeline,

        # Named entries so to run each separately
        "data_processing_nonpyspark": nonpyspark_pipeline,
        "data_processing_pyspark": pyspark_pipeline,
    }
