import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _as_nullable_int(s: pd.Series) -> pd.Series:
    """Coerce to pandas nullable Int64, handling bad parses as NA."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def clean_and_merge_employee_salary(
    df_employee: pd.DataFrame, df_salary: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean employee & salary datasets, merge, deduplicate, and handle outliers.
    Logs only major checkpoints (merge + deduplication).
    """

    # --- EMPLOYEE CLEANING ----------------------------------------------------
    e = df_employee.copy()

    e = e.rename(
        columns={
            "jobId": "job_id",
            "companyId": "company_id",
            "jobRole": "job_role",
            "education": "education",
            "major": "major",
            "Industry": "industry",
            "yearsExperience": "years_experience",
            "distanceFromCBD": "distance_from_cbd",
        }
    )

    if "company_id" in e.columns:
        e["company_id"] = (
            e["company_id"]
            .astype("string")
            .str.replace("COMP", "", regex=False)
            .replace("<NA>", pd.NA)
            .pipe(_as_nullable_int)
        )

    if "job_id" in e.columns:
        mask_valid = (
            e["job_id"].astype("string").str.fullmatch(r"JOB\d+") | e["job_id"].isna()
        )
        invalid_count = (~mask_valid).sum()
        if invalid_count:
            logger.warning("Found %d job_id not matching 'JOB\\d+' format.", invalid_count)

        e["job_id"] = (
            e["job_id"]
            .astype("string")
            .str.replace("JOB", "", regex=False)
            .replace(["<NA>", "nan", "NaN"], pd.NA)
            .pipe(_as_nullable_int)
        )

    for col in ["years_experience", "distance_from_cbd"]:
        if col in e.columns:
            e[col] = pd.to_numeric(e[col], errors="coerce")

    for col in ["education", "major"]:
        if col in e.columns:
            e[col] = (
                e[col]
                .replace(["NA", "na", "NaN", "nan", "<NA>"], pd.NA)
                .fillna("NONE")
            )

    to_check = [c for c in ["job_role", "industry"] if c in e.columns]
    if to_check:
        e = e.dropna(subset=to_check)

    int_like = e.select_dtypes(include=["int64", "Int64"]).columns
    if len(int_like):
        e = e.dropna(subset=int_like)

    if "company_id" in e.columns and str(e["company_id"].dtype) in ("Int64", "int64"):
        e["company_id"] = e["company_id"].astype("category")

    for col in e.select_dtypes(include=["object", "string"]).columns:
        e[col] = e[col].astype("category")

    int64_nullable_cols = e.select_dtypes(include="Int64").columns
    for col in int64_nullable_cols:
        e[col] = e[col].astype("int64")

    # --- SALARY CLEANING ------------------------------------------------------
    s = df_salary.copy()
    s = s.rename(columns={"jobId": "job_id", "salaryInThousands": "salary_k"})

    s["job_id"] = _as_nullable_int(
        s["job_id"].astype("string").str.replace("JOB", "", regex=False)
    )
    s["salary_k"] = pd.to_numeric(s["salary_k"], errors="coerce")

    s = s.dropna(subset=["job_id", "salary_k"])
    s["job_id"] = s["job_id"].astype("int64")
    s["salary_k"] = s["salary_k"].astype("int64")

    # --- MERGE & DEDUP --------------------------------------------------------
    rows_emp_before, rows_sal_before = len(e), len(s)
    m = e.merge(s, on="job_id", how="inner")
    logger.info(
        "Merged employee (%d) x salary (%d) -> merged (%d)",
        rows_emp_before,
        rows_sal_before,
        len(m),
    )

    dup_full = m.duplicated().sum()
    dup_id = m["job_id"].duplicated().sum()
    if dup_full or dup_id:
        logger.warning(
            "Duplicates found: full=%d, by job_id=%d. Dropping duplicates.", dup_full, dup_id
        )
        m = m.drop_duplicates()
        m = m.drop_duplicates(subset=["job_id"], keep="first")
    logger.info("After deduplication: %d rows", len(m))

    # --- OUTLIERS & FINAL TIDY ------------------------------------------------
    if "job_role" in m.columns:
        m = m[m["job_role"] != "PRESIDENT"]
        if pd.api.types.is_categorical_dtype(m["job_role"]):
            m["job_role"] = m["job_role"].cat.remove_unused_categories()

    if "industry" in m.columns and pd.api.types.is_categorical_dtype(m["industry"]):
        m["industry"] = m["industry"].cat.remove_unused_categories()

    if "distance_from_cbd" in m.columns:
        m = m[m["distance_from_cbd"] <= 100]

    if "salary_k" in m.columns:
        m = m[m["salary_k"] != 0]
        m = m[m["salary_k"] != 10_000_000]

    if "job_role" in m.columns:
        jan_mask = m["job_role"] == "JANITOR"
        janitors = m.loc[jan_mask].copy()
        if not janitors.empty:
            Q1 = janitors["salary_k"].quantile(0.25)
            Q3 = janitors["salary_k"].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            janitors_clean = janitors[
                (janitors["salary_k"] >= lower) & (janitors["salary_k"] <= upper)
            ]
            non_janitors = m.loc[~jan_mask]
            m = pd.concat([janitors_clean, non_janitors], ignore_index=True)

    logger.info("Final dataset ready for saving: %d rows", len(m))
    return m
