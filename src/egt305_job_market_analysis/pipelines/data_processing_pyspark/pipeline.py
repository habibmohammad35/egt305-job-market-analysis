from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_and_merge_employee_salary_spark,
    pre_split_feature_engineering_spark
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_and_merge_employee_salary_spark,
                inputs=["employee_dataset_raw_spark", "employee_salary_raw_spark"],
                outputs="employee_salary_clean_spark",
                name="clean_merge_employee_salary_spark_node",
            ),
            node(
                func=pre_split_feature_engineering_spark,
                inputs="employee_salary_clean_spark",    
                outputs="employee_features_spark", 
                name="pre_split_feature_engineering_node_spark",
            )
        ]
    )
