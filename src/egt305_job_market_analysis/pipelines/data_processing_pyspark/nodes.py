import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

logger = logging.getLogger(__name__)

def _rename_columns(df: DataFrame, mapping: dict) -> DataFrame:
    for old, new in mapping.items():
        if old in df.columns and old != new:
            df = df.withColumnRenamed(old, new)
    return df

def _extract_digits_long(colname: str):
    """Extract numeric part of an ID string (JOBxxxx, COMPxxxx, etc.) as LongType."""
    return F.regexp_extract(F.col(colname).cast("string"), r"(\d+)", 1).cast(T.LongType())

def clean_and_merge_employee_salary_spark(
    df_employee: DataFrame, df_salary: DataFrame
) -> DataFrame:

    # --------------------- EMPLOYEE CLEANING ---------------------
    e = df_employee
    e = _rename_columns(
        e,
        {
            "jobId": "job_id",
            "companyId": "company_id",
            "jobRole": "job_role",
            "education": "education",
            "major": "major",
            "Industry": "industry",
            "yearsExperience": "years_experience",
            "distanceFromCBD": "distance_from_cbd",
        },
    )

    if "company_id" in e.columns:
        e = e.withColumn("company_id", _extract_digits_long("company_id"))

    if "job_id" in e.columns:
        e = e.withColumn("job_id", _extract_digits_long("job_id"))

    for c in ["years_experience", "distance_from_cbd"]:
        if c in e.columns:
            e = e.withColumn(c, F.col(c).cast(T.DoubleType()))

    na_tokens = ["NA", "na", "NaN", "nan", "<NA>"]
    for c in ["education", "major"]:
        if c in e.columns:
            e = e.withColumn(
                c,
                F.when(F.col(c).isin(na_tokens), None).otherwise(F.col(c).cast(T.StringType())),
            )
    fill_map = {c: "NONE" for c in ["education", "major"] if c in e.columns}
    if fill_map:
        e = e.fillna(fill_map)

    for c in ["job_role", "industry"]:
        if c in e.columns:
            e = e.filter(F.col(c).isNotNull())

    for c in ["job_id", "company_id"]:
        if c in e.columns:
            e = e.filter(F.col(c).isNotNull())

    # --------------------- SALARY CLEANING -----------------------
    s = df_salary
    s = _rename_columns(s, {"jobId": "job_id", "salaryInThousands": "salary_k"})
    s = s.withColumn("job_id", _extract_digits_long("job_id"))
    s = s.withColumn("salary_k", F.col("salary_k").cast(T.LongType()))
    s = s.dropna(subset=["job_id", "salary_k"])

    # --------------------- MERGE & DEDUP -------------------------
    m = e.join(s, on="job_id", how="inner")
    logger.info("After merge: %d rows", m.count())

    dup_full = m.count() - m.dropDuplicates().count()
    dup_id = m.count() - m.dropDuplicates(["job_id"]).count()
    if dup_full or dup_id:
        logger.warning("Duplicates found: full=%d, by job_id=%d. Dropping.", dup_full, dup_id)
        m = m.dropDuplicates().dropDuplicates(["job_id"])
    logger.info("After deduplication: %d rows", m.count())

    # --------------------- OUTLIERS -------------------------------
    if "job_role" in m.columns:
        m = m.filter(F.col("job_role") != "PRESIDENT")

    if "distance_from_cbd" in m.columns:
        m = m.filter(F.col("distance_from_cbd") <= 100)

    if "salary_k" in m.columns:
        m = m.filter((F.col("salary_k") != 0) & (F.col("salary_k") != 10_000_000))

    if set(["job_role", "salary_k"]).issubset(m.columns):
        janitors = m.filter(F.col("job_role") == "JANITOR")
        if janitors.take(1):
            q1, q3 = janitors.approxQuantile("salary_k", [0.25, 0.75], 0.01)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            janitors_clean = janitors.filter((F.col("salary_k") >= lower) & (F.col("salary_k") <= upper))
            non_janitors = m.filter(F.col("job_role") != "JANITOR")
            m = janitors_clean.unionByName(non_janitors)

    logger.info("Final dataset ready for saving: %d rows", m.count())
    return m
