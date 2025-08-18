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

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

def pre_split_feature_engineering_spark(df: DataFrame) -> DataFrame:
    """
    Create pre-split features that do not use target (salary).
    """

    # --- Dictionaries as Spark maps ---
    edu_map = {
        "NONE": 0,
        "HIGH_SCHOOL": 1,
        "BACHELORS": 2,
        "MASTERS": 3,
        "DOCTORAL": 4,
    }
    role_rank = {
        "CEO": 6,
        "CTO": 5, "CFO": 5,
        "VICE_PRESIDENT": 4,
        "MANAGER": 3,
        "SENIOR": 2,
        "JUNIOR": 1,
        "JANITOR": 0,
    }
    industry_score = {
        "EDUCATION": 1,
        "SERVICE": 1,
        "AUTO": 2,
        "HEALTH": 3,
        "WEB": 4,
        "FINANCE": 5,
        "OIL": 5
    }
    major_score = {
        "NONE": 0,
        "LITERATURE": 1,
        "BIOLOGY": 2,
        "CHEMISTRY": 3,
        "PHYSICS": 4,
        "COMPSCI": 5,
        "MATH": 6,
        "BUSINESS": 7,
        "ENGINEERING": 8
    }

    # Helper: dictionary -> Spark map expression
    def dict_to_map(d: dict):
        return F.create_map([F.lit(x) for kv in d.items() for x in kv])

    # Apply mappings
    df = df.withColumn("education_level", dict_to_map(edu_map)[F.col("education")].cast("int"))
    df = df.withColumn("job_role_rank", dict_to_map(role_rank)[F.col("job_role")].cast("int"))
    df = df.withColumn("industry_score", dict_to_map(industry_score)[F.col("industry")].cast("int"))
    df = df.withColumn("major_score", dict_to_map(major_score)[F.col("major")].cast("int"))

    # Handcrafted score
    df = df.withColumn(
        "handcrafted_score",
        (F.col("industry_score") +
         F.col("major_score") +
         F.col("job_role_rank") +
         F.col("education_level")).cast("int")
    )

    # Cross-relations
    df = df.withColumn("edu_major", F.concat_ws("_", F.col("education"), F.col("major")))
    df = df.withColumn("industry_role", F.concat_ws("_", F.col("industry"), F.col("job_role")))

    # Drop unused columns
    df = df.drop("education", "job_role", "industry", "major", "company_id")

    return df


