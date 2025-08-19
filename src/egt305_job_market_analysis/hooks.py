from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")

import time
import psutil
from kedro.framework.hooks import hook_impl
class TimingHook:
    def __init__(self):
        self.node_start = {}
        self.node_stats = {"spark": [], "nonspark": [], "other": []}

    @hook_impl
    def before_node_run(self, node, inputs):
        # just store start time
        self.node_start[node.name] = time.time()

    @hook_impl
    def after_node_run(self, node, outputs):
        start_time = self.node_start[node.name]
        elapsed = time.time() - start_time

        # CPU usage since last call
        cpu_used = psutil.cpu_percent(interval=None)

        # decide category from tags
        if "spark" in node.tags:
            category = "spark"
        elif "nonspark" in node.tags:
            category = "nonspark"
        else:
            category = "other"

        self.node_stats[category].append((node.name, elapsed, cpu_used))

        print(f"[{category.upper()} | {node.name}] "
              f"{elapsed:.2f}s | CPU {cpu_used:.2f}%")

    @hook_impl
    def after_pipeline_run(self, run_params, run_result):
        print("\n=== Run Summary by Category ===")
        for cat, stats in self.node_stats.items():
            if not stats:
                continue
            total_time = sum(t for _, t, _ in stats)
            avg_cpu = sum(c for _, _, c in stats) / len(stats)
            print(f"{cat.upper():8} → {len(stats)} nodes | "
                  f"Total: {total_time:.2f}s | Avg CPU: {avg_cpu:.2f}%")
