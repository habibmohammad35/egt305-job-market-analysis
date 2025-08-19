from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_and_merge_employee_salary,
    pre_split_feature_engineering
)



def create_pipeline(**kwargs) -> Pipeline:
    "Create a Kedro pipeline for cleaning and merging employee salary data."
    return pipeline(
        [
            node(
                func=clean_and_merge_employee_salary,
                inputs=["employee_dataset_raw", "employee_salaries_raw"],
                outputs="employee_salary_clean",
                name="clean_merge_employee_salary_node_nonspark",
                tags=["nonspark"]
            ),
            node(
                func=pre_split_feature_engineering,
                inputs="employee_salary_clean",
                outputs="employee_dataset_features",
                name="pre_split_feature_engineering_node_nonspark",
                tags=["nonspark"]
            ),
            
        ]
    )
