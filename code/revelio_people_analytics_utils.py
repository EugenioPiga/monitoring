from __future__ import annotations

import os
import zipfile
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


OBS_START_YEAR = 1950
DEFAULT_PROJECT_ROOT = "/labs/khanna/predictive_capital/revelio_people_analytics"

DEFAULT_RAW_PATHS = {
    "company_ref": "/labs/bharadwajlab/linkedin/company_ref",
    "education": "/labs/bharadwajlab/linkedin/individual_education",
    "position": "/labs/bharadwajlab/linkedin/individual_position",
    "skill": "/labs/bharadwajlab/linkedin/individual_skill",
    "user": "/labs/bharadwajlab/linkedin/individual_user",
    "postings": "/labs/bharadwajlab/postings/postings_aggregator_individual_all_location.zip.1",
}

TEXT_COLUMNS_COMPANY = [
    "company",
    "primary_name",
    "ticker",
    "exchange_name",
    "url",
    "naics_code",
    "child_company",
    "ultimate_parent_rcid_name",
    "factset_entity_id",
    "linkedin_url",
    "child_linkedin_url",
]

TEXT_COLUMNS_USER = [
    "firstname",
    "lastname",
    "fullname",
    "highest_degree",
    "sex_predicted",
    "ethnicity_predicted",
    "linkedin_url",
    "user_location",
    "country",
    "title",
    "summary",
]

TEXT_COLUMNS_EDUCATION = [
    "school",
    "university_name",
    "degree",
    "field",
    "degree_raw",
    "field_raw",
    "university_country",
    "university_location",
]

TEXT_COLUMNS_SKILL = [
    "skill_raw",
    "skill_source",
    "skill_mapped_v1",
    "skill_k25_v1",
    "skill_k50_v1",
    "skill_k75_v1",
    "skill_mapped",
    "skill_k25",
    "skill_k50",
    "skill_k75",
]

TEXT_COLUMNS_POSITION = [
    "company_raw",
    "company_linkedin_url",
    "company_cleaned",
    "location_raw",
    "region",
    "country",
    "state",
    "metro_area",
    "jobtitle_raw",
    "mapped_role",
    "job_category",
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "description",
    "company_name",
    "ultimate_parent_company_name",
    "onet_code",
    "onet_title",
    "ticker",
    "exchange",
    "cusip",
    "naics",
    "naics_desc",
    "final_parent_factset_id",
    "final_parent_factset_name",
]

TEXT_COLUMNS_POSTING = [
    "company",
    "country",
    "state",
    "mapped_role",
    "role_k50",
    "role_k150",
    "job_category",
    "ultimate_parent_company_name",
    "jobtitle_raw",
    "jobtitle",
    "jobtitle_translated",
    "description",
    "location",
    "city",
    "metro_area",
]

STEM_REGEX = (
    r"\b(?:engineering|computer|information technology|it\b|mathematics|math|statistics|"
    r"physics|chemistry|biology|biotech|data science|computer science|software)\b"
)
BUSINESS_REGEX = r"\b(?:business|finance|accounting|marketing|management|mba|economics|commerce)\b"
LAW_REGEX = r"\b(?:law|legal|juris)\b"
HEALTH_REGEX = r"\b(?:medicine|medical|nursing|pharmacy|public health|healthcare|dentistry)\b"

SKILL_DOMAIN_REGEX = {
    "data_skill": (
        r"\b(?:sql|python|tableau|power bi|machine learning|data analysis|analytics|statistics|"
        r"business intelligence|bi\b|data science|r programming)\b"
    ),
    "software_skill": (
        r"\b(?:javascript|java\b|html|css|php|node js|git\b|mysql|linux|c#|c\+\+|"
        r"software development|web development|react|typescript|angular)\b"
    ),
    "management_skill": (
        r"\b(?:management|leadership|project management|strategic planning|team leadership|"
        r"operations management|program management)\b"
    ),
    "hr_skill": (
        r"\b(?:human resources|recruiting|talent management|employee relations|benefits|payroll|"
        r"training|onboarding|performance management|compensation|hris|workday)\b"
    ),
    "sales_marketing_skill": r"\b(?:sales|marketing|social media|seo|brand|customer service|advertising|crm)\b",
    "finance_skill": r"\b(?:financial analysis|accounting|auditing|budgeting|forecasting|banking|valuation|quickbooks)\b",
    "operations_skill": (
        r"\b(?:supply chain|logistics|inventory|warehouse|procurement|lean|six sigma|operations|manufacturing)\b"
    ),
    "employee_feedback_tool_skill": r"\b(?:qualtrics|glint|culture amp|visier|peakon)\b",
    "hr_technology_skill": (
        r"\b(?:workday|successfactors|oracle hcm|ukg|dayforce|adp|peoplesoft|hris|human capital management|hcm)\b"
    ),
}

PEOPLE_ANALYTICS_STUDY_REGEX = (
    r"\b(?:people analytics|hr analytics|human resource analytics|human resources analytics|"
    r"workforce analytics|talent analytics|employee analytics|human capital analytics)\b"
)

PEOPLE_ANALYTICS_ENRICHED_REGEX = (
    r"\b(?:people analytics|hr analytics|human resource analytics|human resources analytics|"
    r"workforce analytics|talent analytics|employee analytics|human capital analytics|"
    r"people insights|employee insights|people science|workforce planning|workforce intelligence|"
    r"talent intelligence|employee listening|organizational effectiveness|organization effectiveness|"
    r"org effectiveness|people data|performance analytics|engagement analytics|retention analytics|"
    r"attrition analytics|employee experience analytics)\b"
)

DATA_ANALYTICS_ROLE_REGEX = (
    r"\b(?:data analyst|data scientist|analytics|business intelligence|statistician|"
    r"quantitative analyst|machine learning)\b"
)

HR_PEOPLE_ROLE_REGEX = (
    r"\b(?:human resources|recruiter|benefits|payroll|compensation|talent acquisition|"
    r"hr business partner|hris|talent development)\b"
)

JOB_CATEGORIES = ["Admin", "Engineer", "Sales", "Scientist", "Marketing", "Finance", "Operations"]


def build_default_paths(project_root: str = DEFAULT_PROJECT_ROOT) -> dict[str, str]:
    processed_root = os.path.join(project_root, "processed")
    return {
        "project_root": project_root,
        "code_root": os.path.join(project_root, "code"),
        "logs_root": os.path.join(project_root, "logs"),
        "scratch_root": os.path.join(project_root, "scratch"),
        "processed_root": processed_root,
        "intermediate_root": os.path.join(processed_root, "intermediate"),
        "final_root": os.path.join(processed_root, "final"),
        "firm_year_output": os.path.join(processed_root, "final", "firm_year_panel"),
        "worker_year_output": os.path.join(processed_root, "final", "worker_year_panel"),
        "firm_year_intermediate": os.path.join(processed_root, "intermediate", "firm_year"),
        "worker_year_intermediate": os.path.join(processed_root, "intermediate", "worker_year"),
    }


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_spark(app_name: str, threads: int, shuffle_partitions: int, tmpdir: str | None = None) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{threads}]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "LEGACY")
        .config("spark.sql.parquet.int96RebaseModeInRead", "LEGACY")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "LEGACY")
        .config("spark.sql.parquet.int96RebaseModeInWrite", "LEGACY")
        .config("spark.sql.parquet.mergeSchema", "false")
    )
    if tmpdir:
        java_tmpdir = tmpdir.replace("\\", "/")
        if " " in java_tmpdir:
            java_tmpdir = f'"{java_tmpdir}"'
        ensure_directory(tmpdir)
        ensure_directory(os.path.join(tmpdir, "warehouse"))
        builder = (
            builder
            .config("spark.local.dir", tmpdir)
            .config("spark.sql.warehouse.dir", os.path.join(tmpdir, "warehouse"))
            .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={java_tmpdir}")
            .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={java_tmpdir}")
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def parquet_reader(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("datetimeRebaseMode", "LEGACY")
        .option("int96RebaseMode", "LEGACY")
        .parquet(path)
    )


def nullify_blank(column_name: str) -> F.Column:
    as_text = F.trim(F.col(column_name).cast("string"))
    return F.when(
        as_text.isNull() | as_text.rlike(r"(?i)^(|none|null|nan|empty)$"),
        F.lit(None),
    ).otherwise(F.regexp_replace(as_text, r"\s+", " "))


def clean_text_columns(frame: DataFrame, columns: list[str]) -> DataFrame:
    for column in columns:
        if column in frame.columns:
            frame = frame.withColumn(column, nullify_blank(column))
    return frame


def normalize_text_expr(*column_names: str) -> F.Column:
    pieces = [F.coalesce(F.col(name).cast("string"), F.lit("")) for name in column_names if name]
    if not pieces:
        return F.lit("")
    joined = F.concat_ws(" ", *pieces)
    normalized = F.lower(F.regexp_replace(joined, r"[^a-z0-9]+", " "))
    return F.trim(F.regexp_replace(normalized, r"\s+", " "))


def slug_expr(column_name: str) -> F.Column:
    return F.regexp_replace(
        F.regexp_replace(F.lower(F.coalesce(F.col(column_name).cast("string"), F.lit("missing"))), r"[^a-z0-9]+", "_"),
        r"(^_+|_+$)",
        "",
    )


def safe_divide(numerator: F.Column, denominator: F.Column) -> F.Column:
    return F.when(denominator.isNull() | (denominator == 0), F.lit(None)).otherwise(numerator / denominator)


def cast_numeric_if_present(frame: DataFrame, column_name: str, dtype: str) -> DataFrame:
    if column_name in frame.columns:
        frame = frame.withColumn(column_name, F.col(column_name).cast(dtype))
    return frame


def first_not_null(column_name: str, alias: str | None = None) -> F.Column:
    return F.first(F.col(column_name), ignorenulls=True).alias(alias or column_name)


def clip_year(column: F.Column, analysis_end_year: int) -> F.Column:
    return F.least(F.greatest(column.cast("int"), F.lit(OBS_START_YEAR)), F.lit(int(analysis_end_year)))


def extract_postings_if_needed(postings_path: str, extract_dir: str | None = None) -> str:
    lower = postings_path.lower()
    if not (lower.endswith(".zip") or ".zip." in lower):
        return postings_path

    if extract_dir is None:
        raise ValueError("A scratch extract directory is required when postings are provided as a zip archive.")

    ensure_directory(extract_dir)
    marker = os.path.join(extract_dir, "_EXTRACT_COMPLETE")
    if os.path.exists(marker):
        return extract_dir

    with zipfile.ZipFile(postings_path, "r") as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".parquet") and not name.endswith("/")]
        members.sort()
        for idx, member in enumerate(members, start=1):
            archive.extract(member, path=extract_dir)
            if idx % 100 == 0:
                print(f"[INFO] Extracted {idx:,} posting parquet shards")

    Path(marker).write_text("ok\n", encoding="utf-8")
    return extract_dir


def outer_join_all(frames: list[DataFrame], on: list[str]) -> DataFrame | None:
    valid = [frame for frame in frames if frame is not None]
    if not valid:
        return None
    return reduce(lambda left, right: left.join(right, on=on, how="outer"), valid[1:], valid[0])


def add_degree_score(frame: DataFrame, source_column: str, output_column: str) -> DataFrame:
    lowered = F.lower(F.coalesce(F.col(source_column), F.lit("")))
    scored = (
        F.when(lowered.contains("doctor"), F.lit(5.0))
        .when(lowered.contains("phd"), F.lit(5.0))
        .when(lowered.contains("mba"), F.lit(4.0))
        .when(lowered.contains("master"), F.lit(4.0))
        .when(lowered.contains("bachelor"), F.lit(3.0))
        .when(lowered.contains("associate"), F.lit(2.0))
        .when(lowered.contains("high school"), F.lit(1.0))
        .otherwise(F.lit(None))
    )
    return frame.withColumn(output_column, scored)


def build_company_lookup(spark: SparkSession, company_ref_path: str) -> tuple[DataFrame, DataFrame]:
    company = parquet_reader(spark, company_ref_path)
    company = clean_text_columns(company, TEXT_COLUMNS_COMPANY)
    for name in ["rcid", "child_rcid", "ultimate_parent_rcid", "year_founded"]:
        company = cast_numeric_if_present(company, name, "long")

    child_lookup = (
        company.select(
            F.col("child_rcid").alias("entity_rcid"),
            F.col("ultimate_parent_rcid"),
            F.col("child_company").alias("entity_company_name"),
            F.col("ultimate_parent_rcid_name").alias("parent_company_name"),
        )
        .where(F.col("entity_rcid").isNotNull())
    )
    direct_lookup = (
        company.select(
            F.col("rcid").alias("entity_rcid"),
            F.col("ultimate_parent_rcid"),
            F.col("company").alias("entity_company_name"),
            F.col("ultimate_parent_rcid_name").alias("parent_company_name"),
        )
        .where(F.col("entity_rcid").isNotNull())
    )

    entity_lookup = (
        child_lookup.unionByName(direct_lookup)
        .dropDuplicates()
        .groupBy("entity_rcid")
        .agg(
            first_not_null("ultimate_parent_rcid"),
            first_not_null("entity_company_name"),
            first_not_null("parent_company_name"),
        )
    )

    parent_static = (
        company.where(F.col("ultimate_parent_rcid").isNotNull())
        .groupBy("ultimate_parent_rcid")
        .agg(
            first_not_null("ultimate_parent_rcid_name", "parent_company_name"),
            first_not_null("primary_name", "parent_primary_name"),
            first_not_null("year_founded"),
            first_not_null("ticker"),
            first_not_null("exchange_name"),
            first_not_null("naics_code"),
            first_not_null("url"),
            first_not_null("factset_entity_id"),
            first_not_null("linkedin_url"),
            F.countDistinct("rcid").alias("company_ref_sample_rcids"),
            F.countDistinct("child_rcid").alias("company_ref_sample_child_rcids"),
        )
        .withColumn("is_public_company", F.when(F.col("ticker").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    )
    return entity_lookup, parent_static


def build_education_features(spark: SparkSession, education_path: str) -> DataFrame:
    education = parquet_reader(spark, education_path)
    education = clean_text_columns(education, TEXT_COLUMNS_EDUCATION)
    education = education.withColumn("degree_clean", F.coalesce(F.col("degree"), F.col("degree_raw")))
    education = education.withColumn("field_clean", F.coalesce(F.col("field"), F.col("field_raw")))
    education = add_degree_score(education, "degree_clean", "degree_score_edu")
    education = cast_numeric_if_present(education, "world_rank", "double")
    education = cast_numeric_if_present(education, "us_rank", "double")

    field_text = F.lower(F.coalesce(F.col("field_clean"), F.lit("")))
    education = (
        education
        .withColumn("edu_is_stem", F.when(field_text.rlike(STEM_REGEX), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("edu_is_business", F.when(field_text.rlike(BUSINESS_REGEX), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("edu_is_law", F.when(field_text.rlike(LAW_REGEX), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("edu_is_health", F.when(field_text.rlike(HEALTH_REGEX), F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "has_top500_school",
            F.when((F.col("world_rank") <= 500) | (F.col("us_rank") <= 100), F.lit(1)).otherwise(F.lit(0)),
        )
    )

    return (
        education.groupBy("user_id")
        .agg(
            F.count(F.lit(1)).alias("education_records"),
            F.max("degree_score_edu").alias("best_degree_score_edu"),
            F.min("world_rank").alias("best_world_rank"),
            F.min("us_rank").alias("best_us_rank"),
            F.max("has_top500_school").alias("has_top500_school"),
            F.max("edu_is_stem").alias("has_stem_education"),
            F.max("edu_is_business").alias("has_business_education"),
            F.max("edu_is_law").alias("has_law_education"),
            F.max("edu_is_health").alias("has_health_education"),
        )
    )


def build_skill_features(spark: SparkSession, skill_path: str) -> DataFrame:
    skills = parquet_reader(spark, skill_path)
    skills = clean_text_columns(skills, TEXT_COLUMNS_SKILL)
    skills = skills.withColumn("skill_text", F.coalesce(F.col("skill_mapped"), F.col("skill_raw")))
    skills = skills.where(F.col("skill_text").isNotNull()).dropDuplicates(["user_id", "skill_text"])

    normalized_skill = F.lower(F.coalesce(F.col("skill_text"), F.lit("")))
    for feature_name, regex in SKILL_DOMAIN_REGEX.items():
        skills = skills.withColumn(feature_name, F.when(normalized_skill.rlike(regex), F.lit(1)).otherwise(F.lit(0)))
    skills = skills.withColumn(
        "is_predicted_skill",
        F.when(F.lower(F.coalesce(F.col("skill_source"), F.lit(""))) == F.lit("predicted"), F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    return (
        skills.groupBy("user_id")
        .agg(
            F.countDistinct("skill_text").alias("distinct_skills"),
            F.avg("is_predicted_skill").alias("predicted_skill_share"),
            F.max("data_skill").alias("has_data_skill"),
            F.max("software_skill").alias("has_software_skill"),
            F.max("management_skill").alias("has_management_skill"),
            F.max("hr_skill").alias("has_hr_skill"),
            F.max("sales_marketing_skill").alias("has_sales_marketing_skill"),
            F.max("finance_skill").alias("has_finance_skill"),
            F.max("operations_skill").alias("has_operations_skill"),
            F.max("employee_feedback_tool_skill").alias("has_employee_feedback_tool_skill"),
            F.max("hr_technology_skill").alias("has_hr_technology_skill"),
        )
    )


def build_user_features(
    spark: SparkSession,
    user_path: str,
    education_path: str,
    skill_path: str,
) -> DataFrame:
    users = parquet_reader(spark, user_path)
    users = clean_text_columns(users, TEXT_COLUMNS_USER).dropDuplicates(["user_id"])
    users = users.withColumn("profile_snapshot_date", F.to_date("updated_dt"))
    users = users.withColumn("profile_snapshot_year", F.year("profile_snapshot_date"))
    users = cast_numeric_if_present(users, "numconnections", "double")
    users = users.withColumn("connections_log1p", F.log1p(F.coalesce(F.col("numconnections"), F.lit(0.0))))
    users = cast_numeric_if_present(users, "prestige", "double")
    for probability in [
        "f_prob",
        "m_prob",
        "white_prob",
        "black_prob",
        "api_prob",
        "hispanic_prob",
        "native_prob",
        "multiple_prob",
    ]:
        users = cast_numeric_if_present(users, probability, "double")
    users = add_degree_score(users, "highest_degree", "degree_score_user")

    education_features = build_education_features(spark, education_path)
    skill_features = build_skill_features(spark, skill_path)

    user_features = users.join(education_features, on="user_id", how="left").join(skill_features, on="user_id", how="left")
    user_features = user_features.withColumn(
        "highest_degree_score",
        F.greatest(F.col("degree_score_user"), F.col("best_degree_score_edu")),
    )
    user_features = user_features.withColumn(
        "has_bachelor_plus",
        F.when(F.col("highest_degree_score") >= 3, F.lit(1.0)).otherwise(F.lit(0.0)),
    )
    user_features = user_features.withColumn(
        "has_advanced_degree",
        F.when(F.col("highest_degree_score") >= 4, F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    zero_fill = [
        "education_records",
        "distinct_skills",
        "predicted_skill_share",
        "has_top500_school",
        "has_stem_education",
        "has_business_education",
        "has_law_education",
        "has_health_education",
        "has_data_skill",
        "has_software_skill",
        "has_management_skill",
        "has_hr_skill",
        "has_sales_marketing_skill",
        "has_finance_skill",
        "has_operations_skill",
        "has_employee_feedback_tool_skill",
        "has_hr_technology_skill",
    ]
    user_features = user_features.fillna(0, subset=[name for name in zero_fill if name in user_features.columns])
    return user_features


def add_people_analytics_flags(
    frame: DataFrame,
    title_columns: list[str],
    description_columns: list[str],
) -> DataFrame:
    title_text = normalize_text_expr(*title_columns)
    description_text = normalize_text_expr(*description_columns)

    frame = frame.withColumn("pa_title_text", title_text)
    frame = frame.withColumn("pa_description_text", description_text)

    flags = {
        "people_analytics_title_study": F.col("pa_title_text").rlike(PEOPLE_ANALYTICS_STUDY_REGEX),
        "people_analytics_description_study": F.col("pa_description_text").rlike(PEOPLE_ANALYTICS_STUDY_REGEX),
        "people_analytics_title_enriched": F.col("pa_title_text").rlike(PEOPLE_ANALYTICS_ENRICHED_REGEX),
        "people_analytics_description_enriched": F.col("pa_description_text").rlike(PEOPLE_ANALYTICS_ENRICHED_REGEX),
    }
    for column_name, condition in flags.items():
        frame = frame.withColumn(column_name, F.when(condition, F.lit(1)).otherwise(F.lit(0)))

    frame = frame.withColumn(
        "people_analytics_any_study",
        F.greatest(F.col("people_analytics_title_study"), F.col("people_analytics_description_study")),
    )
    frame = frame.withColumn(
        "people_analytics_any_enriched",
        F.greatest(F.col("people_analytics_title_enriched"), F.col("people_analytics_description_enriched")),
    )
    return frame


def prepare_positions(
    spark: SparkSession,
    position_path: str,
    user_features: DataFrame,
    entity_lookup: DataFrame,
    analysis_end_year: int,
) -> DataFrame:
    positions = parquet_reader(spark, position_path)
    positions = clean_text_columns(positions, TEXT_COLUMNS_POSITION).dropDuplicates(["position_id"])
    for column_name in ["rcid", "ultimate_parent_rcid"]:
        positions = cast_numeric_if_present(positions, column_name, "long")
    for column_name in [
        "remote_suitability",
        "weight",
        "start_mean_sampled_salary",
        "end_mean_sampled_salary",
        "seniority",
        "salary",
    ]:
        positions = cast_numeric_if_present(positions, column_name, "double")

    lookup = entity_lookup.select(
        F.col("entity_rcid").alias("lookup_entity_rcid"),
        F.col("ultimate_parent_rcid").alias("lookup_parent_rcid"),
        F.col("entity_company_name").alias("lookup_entity_company_name"),
        F.col("parent_company_name").alias("lookup_parent_company_name"),
    )
    positions = positions.join(lookup, positions["rcid"] == lookup["lookup_entity_rcid"], how="left")
    positions = positions.withColumn("parent_rcid", F.coalesce(F.col("ultimate_parent_rcid"), F.col("lookup_parent_rcid")))
    positions = positions.withColumn(
        "firm_name",
        F.coalesce(
            F.col("ultimate_parent_company_name"),
            F.col("lookup_parent_company_name"),
            F.col("company_name"),
            F.col("lookup_entity_company_name"),
            F.col("company_cleaned"),
            F.col("company_raw"),
        ),
    )
    positions = positions.withColumn(
        "firm_key",
        F.when(F.col("parent_rcid").isNotNull(), F.concat(F.lit("parent_"), F.col("parent_rcid").cast("string")))
        .when(F.col("firm_name").isNotNull(), F.concat(F.lit("name_"), slug_expr("firm_name")))
        .otherwise(F.concat(F.lit("unknown_position_"), F.col("position_id").cast("string"))),
    )

    positions = positions.withColumn("start_date", F.to_date("startdate"))
    positions = positions.withColumn("end_date", F.to_date("enddate"))
    positions = positions.withColumn(
        "end_date",
        F.when(
            F.col("start_date").isNotNull() & F.col("end_date").isNotNull() & (F.col("start_date") > F.col("end_date")),
            F.col("start_date"),
        ).otherwise(F.col("end_date")),
    )

    user_join_columns = [
        "user_id",
        "profile_snapshot_date",
        "profile_snapshot_year",
        "f_prob",
        "white_prob",
        "black_prob",
        "api_prob",
        "hispanic_prob",
        "native_prob",
        "multiple_prob",
        "prestige",
        "numconnections",
        "connections_log1p",
        "highest_degree_score",
        "has_bachelor_plus",
        "has_advanced_degree",
        "education_records",
        "has_top500_school",
        "has_stem_education",
        "has_business_education",
        "has_law_education",
        "has_health_education",
        "distinct_skills",
        "predicted_skill_share",
        "has_data_skill",
        "has_software_skill",
        "has_management_skill",
        "has_hr_skill",
        "has_sales_marketing_skill",
        "has_finance_skill",
        "has_operations_skill",
        "has_employee_feedback_tool_skill",
        "has_hr_technology_skill",
    ]
    positions = positions.join(user_features.select(*[c for c in user_join_columns if c in user_features.columns]), on="user_id", how="left")
    positions = positions.withColumn(
        "profile_snapshot_year",
        F.coalesce(F.col("profile_snapshot_year"), F.lit(int(analysis_end_year))).cast("int"),
    )
    positions = positions.withColumn("start_year_obs", F.year("start_date"))
    positions = positions.withColumn("end_year_obs", F.year("end_date"))
    positions = positions.withColumn(
        "panel_start_year",
        clip_year(
            F.coalesce(F.col("start_year_obs"), F.col("end_year_obs"), F.col("profile_snapshot_year"), F.lit(int(analysis_end_year))),
            analysis_end_year,
        ),
    )
    positions = positions.withColumn(
        "panel_end_year",
        clip_year(F.coalesce(F.col("end_year_obs"), F.col("profile_snapshot_year"), F.lit(int(analysis_end_year))), analysis_end_year),
    )
    positions = positions.withColumn(
        "panel_start_year",
        F.when(F.col("panel_start_year") > F.col("panel_end_year"), F.col("panel_end_year")).otherwise(F.col("panel_start_year")),
    )
    positions = positions.withColumn("start_year_imputed", F.when(F.col("start_year_obs").isNull(), F.lit(1)).otherwise(F.lit(0)))
    positions = positions.withColumn("end_year_imputed", F.when(F.col("end_year_obs").isNull(), F.lit(1)).otherwise(F.lit(0)))
    positions = positions.withColumn(
        "both_dates_missing",
        F.when(F.col("start_date").isNull() & F.col("end_date").isNull(), F.lit(1)).otherwise(F.lit(0)),
    )
    positions = positions.withColumn("weight", F.greatest(F.coalesce(F.col("weight"), F.lit(1.0)), F.lit(0.0)))
    positions = add_people_analytics_flags(positions, ["jobtitle_raw"], ["description"])

    role_text = normalize_text_expr("mapped_role", "role_k50", "role_k150", "onet_title")
    positions = positions.withColumn(
        "is_data_analytics_role",
        F.when(role_text.rlike(DATA_ANALYTICS_ROLE_REGEX), F.lit(1)).otherwise(F.lit(0)),
    )
    positions = positions.withColumn(
        "is_hr_people_role",
        F.when(role_text.rlike(HR_PEOPLE_ROLE_REGEX), F.lit(1)).otherwise(F.lit(0)),
    )
    positions = positions.withColumn(
        "is_us_position",
        F.when(F.col("country") == F.lit("United States"), F.lit(1)).otherwise(F.lit(0)),
    )
    positions = positions.withColumn("salary_observed", F.when(F.col("salary").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    for category in JOB_CATEGORIES:
        positions = positions.withColumn(
            f"jobcat_{category.lower()}",
            F.when(F.col("job_category") == F.lit(category), F.lit(1)).otherwise(F.lit(0)),
        )
    return positions.drop("lookup_entity_rcid")


def expand_positions_to_years(positions: DataFrame) -> DataFrame:
    expanded = positions.withColumn("year", F.explode(F.sequence(F.col("panel_start_year"), F.col("panel_end_year"))))
    expanded = expanded.withColumn("year", F.col("year").cast("int"))
    expanded = expanded.withColumn(
        "is_hire_year_observed",
        F.when((F.col("year") == F.col("panel_start_year")) & F.col("start_year_obs").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
    )
    expanded = expanded.withColumn(
        "is_exit_year_observed",
        F.when((F.col("year") == F.col("panel_end_year")) & F.col("end_year_obs").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
    )
    expanded = expanded.withColumn(
        "is_censored_current_year",
        F.when((F.col("year") == F.col("panel_end_year")) & F.col("end_year_obs").isNull(), F.lit(1)).otherwise(F.lit(0)),
    )
    return expanded


def prepare_weighted_columns(frame: DataFrame, value_columns: list[str], weight_column: str = "weight") -> DataFrame:
    for column_name in value_columns:
        frame = frame.withColumn(
            f"w_{column_name}",
            F.when(F.col(column_name).isNotNull(), F.col(column_name).cast("double") * F.col(weight_column).cast("double")).otherwise(F.lit(0.0)),
        )
        frame = frame.withColumn(
            f"wobs_{column_name}",
            F.when(F.col(column_name).isNotNull(), F.col(weight_column).cast("double")).otherwise(F.lit(0.0)),
        )
    return frame


def add_weighted_means(
    frame: DataFrame,
    value_columns: list[str],
    rename_map: dict[str, str],
) -> DataFrame:
    for column_name in value_columns:
        target_name = rename_map.get(column_name, column_name)
        frame = frame.withColumn(target_name, safe_divide(F.col(f"w_{column_name}"), F.col(f"wobs_{column_name}")))
    return frame


def build_first_event_dates(
    frame: DataFrame,
    firm_key_column: str,
    date_column: str,
    specs: list[tuple[str, str]],
) -> DataFrame | None:
    outputs = []
    for flag_column, output_name in specs:
        if flag_column not in frame.columns:
            continue
        outputs.append(
            frame.where(F.col(flag_column) == 1)
            .groupBy(firm_key_column)
            .agg(F.min(F.col(date_column)).alias(output_name))
        )
    return outer_join_all(outputs, on=[firm_key_column])


def build_position_firm_year(positions: DataFrame) -> tuple[DataFrame, DataFrame | None, DataFrame]:
    expanded = expand_positions_to_years(positions)
    weighted_columns = [
        "remote_suitability",
        "salary",
        "start_mean_sampled_salary",
        "end_mean_sampled_salary",
        "seniority",
        "is_us_position",
        "is_data_analytics_role",
        "is_hr_people_role",
        "people_analytics_title_study",
        "people_analytics_description_study",
        "people_analytics_any_study",
        "people_analytics_title_enriched",
        "people_analytics_description_enriched",
        "people_analytics_any_enriched",
        "f_prob",
        "white_prob",
        "black_prob",
        "api_prob",
        "hispanic_prob",
        "native_prob",
        "multiple_prob",
        "prestige",
        "numconnections",
        "connections_log1p",
        "highest_degree_score",
        "has_bachelor_plus",
        "has_advanced_degree",
        "education_records",
        "has_top500_school",
        "has_stem_education",
        "has_business_education",
        "has_law_education",
        "has_health_education",
        "distinct_skills",
        "predicted_skill_share",
        "has_data_skill",
        "has_software_skill",
        "has_management_skill",
        "has_hr_skill",
        "has_sales_marketing_skill",
        "has_finance_skill",
        "has_operations_skill",
        "has_employee_feedback_tool_skill",
        "has_hr_technology_skill",
        "start_year_imputed",
        "end_year_imputed",
        "both_dates_missing",
        "is_hire_year_observed",
        "is_exit_year_observed",
        "is_censored_current_year",
        "jobcat_admin",
        "jobcat_engineer",
        "jobcat_sales",
        "jobcat_scientist",
        "jobcat_marketing",
        "jobcat_finance",
        "jobcat_operations",
        "salary_observed",
    ]
    expanded = prepare_weighted_columns(expanded, weighted_columns)

    group_exprs = [
        first_not_null("parent_rcid"),
        first_not_null("firm_name"),
        F.count(F.lit(1)).alias("active_position_records"),
        F.countDistinct("user_id").alias("unique_sampled_users"),
        F.sum("weight").alias("workforce_weighted"),
        F.countDistinct("state").alias("distinct_position_states"),
        F.countDistinct("metro_area").alias("distinct_position_metros"),
        F.countDistinct("country").alias("distinct_position_countries"),
        F.countDistinct("job_category").alias("distinct_position_job_categories"),
        F.countDistinct("mapped_role").alias("distinct_position_mapped_roles"),
        F.countDistinct("role_k50").alias("distinct_position_role_k50"),
    ]
    for column_name in weighted_columns:
        group_exprs.append(F.sum(f"w_{column_name}").alias(f"w_{column_name}"))
        group_exprs.append(F.sum(f"wobs_{column_name}").alias(f"wobs_{column_name}"))

    firm_year = expanded.groupBy("firm_key", "year").agg(*group_exprs)
    rename_map = {
        "remote_suitability": "avg_remote_suitability",
        "salary": "avg_salary",
        "start_mean_sampled_salary": "avg_start_salary",
        "end_mean_sampled_salary": "avg_end_salary",
        "seniority": "avg_seniority",
        "is_us_position": "us_position_share",
        "is_data_analytics_role": "data_analytics_role_share",
        "is_hr_people_role": "hr_people_role_share",
        "people_analytics_title_study": "people_analytics_positions_title_study_share",
        "people_analytics_description_study": "people_analytics_positions_description_study_share",
        "people_analytics_any_study": "people_analytics_positions_any_study_share",
        "people_analytics_title_enriched": "people_analytics_positions_title_enriched_share",
        "people_analytics_description_enriched": "people_analytics_positions_description_enriched_share",
        "people_analytics_any_enriched": "people_analytics_positions_any_enriched_share",
        "f_prob": "female_share",
        "white_prob": "white_share",
        "black_prob": "black_share",
        "api_prob": "api_share",
        "hispanic_prob": "hispanic_share",
        "native_prob": "native_share",
        "multiple_prob": "multiple_share",
        "prestige": "avg_prestige",
        "numconnections": "avg_numconnections",
        "connections_log1p": "avg_log_connections",
        "highest_degree_score": "avg_highest_degree_score",
        "has_bachelor_plus": "bachelor_plus_share",
        "has_advanced_degree": "advanced_degree_share",
        "education_records": "avg_education_records",
        "has_top500_school": "top500_school_share",
        "has_stem_education": "stem_education_share",
        "has_business_education": "business_education_share",
        "has_law_education": "law_education_share",
        "has_health_education": "health_education_share",
        "distinct_skills": "avg_distinct_skills",
        "predicted_skill_share": "avg_predicted_skill_share",
        "has_data_skill": "workers_with_data_skill_share",
        "has_software_skill": "workers_with_software_skill_share",
        "has_management_skill": "workers_with_management_skill_share",
        "has_hr_skill": "workers_with_hr_skill_share",
        "has_sales_marketing_skill": "workers_with_sales_marketing_skill_share",
        "has_finance_skill": "workers_with_finance_skill_share",
        "has_operations_skill": "workers_with_operations_skill_share",
        "has_employee_feedback_tool_skill": "workers_with_employee_feedback_tool_skill_share",
        "has_hr_technology_skill": "workers_with_hr_technology_skill_share",
        "start_year_imputed": "share_start_year_imputed",
        "end_year_imputed": "share_end_year_imputed",
        "both_dates_missing": "share_both_dates_missing",
        "is_censored_current_year": "share_censored_current_positions",
        "jobcat_admin": "admin_role_share",
        "jobcat_engineer": "engineer_role_share",
        "jobcat_sales": "sales_role_share",
        "jobcat_scientist": "scientist_role_share",
        "jobcat_marketing": "marketing_role_share",
        "jobcat_finance": "finance_role_share",
        "jobcat_operations": "operations_role_share",
        "salary_observed": "position_salary_coverage_share",
    }
    firm_year = add_weighted_means(firm_year, weighted_columns, rename_map)
    firm_year = (
        firm_year
        .withColumn("hires_weighted", F.col("w_is_hire_year_observed"))
        .withColumn("exits_weighted", F.col("w_is_exit_year_observed"))
        .withColumn("hire_rate", safe_divide(F.col("w_is_hire_year_observed"), F.col("workforce_weighted")))
        .withColumn("exit_rate", safe_divide(F.col("w_is_exit_year_observed"), F.col("workforce_weighted")))
        .withColumn("people_analytics_positions_title_study_weighted", F.col("w_people_analytics_title_study"))
        .withColumn("people_analytics_positions_description_study_weighted", F.col("w_people_analytics_description_study"))
        .withColumn("people_analytics_positions_any_study_weighted", F.col("w_people_analytics_any_study"))
        .withColumn("people_analytics_positions_title_enriched_weighted", F.col("w_people_analytics_title_enriched"))
        .withColumn("people_analytics_positions_description_enriched_weighted", F.col("w_people_analytics_description_enriched"))
        .withColumn("people_analytics_positions_any_enriched_weighted", F.col("w_people_analytics_any_enriched"))
    )
    drop_helper = [f"w_{name}" for name in weighted_columns] + [f"wobs_{name}" for name in weighted_columns]
    firm_year = firm_year.drop(*drop_helper)

    first_dates = build_first_event_dates(
        positions,
        "firm_key",
        "start_date",
        [
            ("people_analytics_title_study", "first_people_analytics_position_date_title_study"),
            ("people_analytics_description_study", "first_people_analytics_position_date_description_study"),
            ("people_analytics_any_study", "first_people_analytics_position_date_any_study"),
            ("people_analytics_title_enriched", "first_people_analytics_position_date_title_enriched"),
            ("people_analytics_description_enriched", "first_people_analytics_position_date_description_enriched"),
            ("people_analytics_any_enriched", "first_people_analytics_position_date_any_enriched"),
        ],
    )

    detail_columns = [
        "user_id",
        "position_id",
        "firm_key",
        "firm_name",
        "parent_rcid",
        "start_date",
        "panel_start_year",
        "jobtitle_raw",
        "description",
        "mapped_role",
        "role_k50",
        "role_k150",
        "job_category",
        "people_analytics_title_study",
        "people_analytics_description_study",
        "people_analytics_any_study",
        "people_analytics_title_enriched",
        "people_analytics_description_enriched",
        "people_analytics_any_enriched",
    ]
    detail = (
        positions.where(F.col("people_analytics_any_enriched") == 1)
        .select(*[name for name in detail_columns if name in positions.columns])
        .orderBy("firm_name", "panel_start_year", "position_id")
    )
    return firm_year, first_dates, detail


def prepare_postings(
    spark: SparkSession,
    postings_path: str,
    entity_lookup: DataFrame,
    extract_dir: str | None = None,
) -> DataFrame:
    resolved_postings = extract_postings_if_needed(postings_path, extract_dir=extract_dir)
    postings = parquet_reader(spark, resolved_postings)
    postings = clean_text_columns(postings, TEXT_COLUMNS_POSTING).dropDuplicates(["job_id"])
    for column_name in ["job_id", "rcid", "ultimate_parent_rcid"]:
        postings = cast_numeric_if_present(postings, column_name, "long")
    postings = cast_numeric_if_present(postings, "salary", "double")

    lookup = entity_lookup.select(
        F.col("entity_rcid").alias("lookup_entity_rcid"),
        F.col("ultimate_parent_rcid").alias("lookup_parent_rcid"),
        F.col("parent_company_name").alias("lookup_parent_company_name"),
    )
    postings = postings.join(lookup, postings["rcid"] == lookup["lookup_entity_rcid"], how="left")
    postings = postings.withColumn("parent_rcid", F.coalesce(F.col("ultimate_parent_rcid"), F.col("lookup_parent_rcid")))
    postings = postings.withColumn(
        "firm_name",
        F.coalesce(F.col("ultimate_parent_company_name"), F.col("lookup_parent_company_name"), F.col("company")),
    )
    postings = postings.withColumn(
        "firm_key",
        F.when(F.col("parent_rcid").isNotNull(), F.concat(F.lit("parent_"), F.col("parent_rcid").cast("string")))
        .otherwise(F.concat(F.lit("name_"), slug_expr("firm_name"))),
    )
    postings = postings.withColumn("post_date", F.to_date("post_date"))
    postings = postings.withColumn("remove_date", F.to_date("remove_date"))
    postings = postings.withColumn("year", F.year("post_date").cast("int"))
    postings = postings.withColumn(
        "posting_duration_days",
        F.when(F.col("remove_date").isNotNull() & F.col("post_date").isNotNull(), F.datediff(F.col("remove_date"), F.col("post_date"))).otherwise(F.lit(None)),
    )
    postings = postings.withColumn(
        "posting_duration_days",
        F.when(F.col("posting_duration_days") < 0, F.lit(None)).otherwise(F.col("posting_duration_days")),
    )
    postings = postings.withColumn("salary_observed", F.when(F.col("salary").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    postings = add_people_analytics_flags(postings, ["jobtitle_raw", "jobtitle", "jobtitle_translated"], ["description"])
    for category in JOB_CATEGORIES:
        postings = postings.withColumn(
            f"jobcat_{category.lower()}",
            F.when(F.col("job_category") == F.lit(category), F.lit(1)).otherwise(F.lit(0)),
        )
    return postings.drop("lookup_entity_rcid")


def build_posting_firm_year(postings: DataFrame) -> tuple[DataFrame, DataFrame | None, DataFrame]:
    group_exprs = [
        first_not_null("parent_rcid"),
        first_not_null("firm_name"),
        F.count(F.lit(1)).alias("posting_count"),
        F.sum("salary_observed").alias("postings_with_salary"),
        F.avg("salary").alias("avg_posting_salary"),
        F.avg("posting_duration_days").alias("avg_posting_duration_days"),
        F.countDistinct("state").alias("distinct_posting_states"),
        F.countDistinct("metro_area").alias("distinct_posting_metros"),
        F.countDistinct("city").alias("distinct_posting_cities"),
        F.countDistinct("job_category").alias("distinct_posting_job_categories"),
        F.countDistinct("mapped_role").alias("distinct_posting_mapped_roles"),
        F.countDistinct("role_k50").alias("distinct_posting_role_k50"),
        F.min("post_date").alias("first_observed_posting_date"),
        F.sum("people_analytics_title_study").alias("people_analytics_postings_title_study"),
        F.sum("people_analytics_description_study").alias("people_analytics_postings_description_study"),
        F.sum("people_analytics_any_study").alias("people_analytics_postings_any_study"),
        F.sum("people_analytics_title_enriched").alias("people_analytics_postings_title_enriched"),
        F.sum("people_analytics_description_enriched").alias("people_analytics_postings_description_enriched"),
        F.sum("people_analytics_any_enriched").alias("people_analytics_postings_any_enriched"),
        F.avg("jobcat_admin").alias("admin_posting_share"),
        F.avg("jobcat_engineer").alias("engineer_posting_share"),
        F.avg("jobcat_sales").alias("sales_posting_share"),
        F.avg("jobcat_scientist").alias("scientist_posting_share"),
        F.avg("jobcat_marketing").alias("marketing_posting_share"),
        F.avg("jobcat_finance").alias("finance_posting_share"),
        F.avg("jobcat_operations").alias("operations_posting_share"),
    ]
    firm_year = postings.groupBy("firm_key", "year").agg(*group_exprs)
    firm_year = (
        firm_year
        .withColumn("salary_coverage_share", safe_divide(F.col("postings_with_salary"), F.col("posting_count")))
        .withColumn("people_analytics_postings_title_study_share", safe_divide(F.col("people_analytics_postings_title_study"), F.col("posting_count")))
        .withColumn("people_analytics_postings_description_study_share", safe_divide(F.col("people_analytics_postings_description_study"), F.col("posting_count")))
        .withColumn("people_analytics_postings_any_study_share", safe_divide(F.col("people_analytics_postings_any_study"), F.col("posting_count")))
        .withColumn("people_analytics_postings_title_enriched_share", safe_divide(F.col("people_analytics_postings_title_enriched"), F.col("posting_count")))
        .withColumn("people_analytics_postings_description_enriched_share", safe_divide(F.col("people_analytics_postings_description_enriched"), F.col("posting_count")))
        .withColumn("people_analytics_postings_any_enriched_share", safe_divide(F.col("people_analytics_postings_any_enriched"), F.col("posting_count")))
    )

    first_dates = build_first_event_dates(
        postings,
        "firm_key",
        "post_date",
        [
            ("people_analytics_title_study", "first_people_analytics_posting_date_title_study"),
            ("people_analytics_description_study", "first_people_analytics_posting_date_description_study"),
            ("people_analytics_any_study", "first_people_analytics_posting_date_any_study"),
            ("people_analytics_title_enriched", "first_people_analytics_posting_date_title_enriched"),
            ("people_analytics_description_enriched", "first_people_analytics_posting_date_description_enriched"),
            ("people_analytics_any_enriched", "first_people_analytics_posting_date_any_enriched"),
        ],
    )

    detail_columns = [
        "job_id",
        "firm_key",
        "firm_name",
        "parent_rcid",
        "post_date",
        "jobtitle_raw",
        "jobtitle",
        "jobtitle_translated",
        "description",
        "mapped_role",
        "role_k50",
        "role_k150",
        "job_category",
        "people_analytics_title_study",
        "people_analytics_description_study",
        "people_analytics_any_study",
        "people_analytics_title_enriched",
        "people_analytics_description_enriched",
        "people_analytics_any_enriched",
    ]
    detail = (
        postings.where(F.col("people_analytics_any_enriched") == 1)
        .select(*[name for name in detail_columns if name in postings.columns])
        .orderBy("firm_name", "post_date", "job_id")
    )
    return firm_year, first_dates, detail


def detect_analysis_end_year(positions_path: str, postings_path: str | None, users_path: str | None, spark: SparkSession) -> int:
    years: list[int] = []
    positions = parquet_reader(spark, positions_path)
    pos_summary = positions.select(
        F.max(F.year(F.to_date("startdate"))).alias("max_start_year"),
        F.max(F.year(F.to_date("enddate"))).alias("max_end_year"),
    ).collect()[0]
    for value in pos_summary:
        if value is not None:
            years.append(int(value))

    if postings_path and not (os.path.isfile(postings_path) and (postings_path.lower().endswith(".zip") or ".zip." in postings_path.lower())):
        postings = parquet_reader(spark, postings_path)
        posting_year = postings.select(F.max(F.year(F.to_date("post_date"))).alias("max_posting_year")).collect()[0][0]
        if posting_year is not None:
            years.append(int(posting_year))

    if users_path:
        users = parquet_reader(spark, users_path)
        snapshot_year = users.select(F.max(F.year(F.to_date("updated_dt"))).alias("max_snapshot_year")).collect()[0][0]
        if snapshot_year is not None:
            years.append(int(snapshot_year))

    return max(years) if years else 2023


def attach_parent_static(panel: DataFrame, parent_static: DataFrame) -> DataFrame:
    joined = panel.join(parent_static, panel["parent_rcid"] == parent_static["ultimate_parent_rcid"], how="left")
    joined = joined.withColumn(
        "firm_name",
        F.coalesce(F.col("firm_name"), F.col("parent_company_name"), F.col("parent_primary_name")),
    )
    joined = joined.withColumn(
        "firm_age",
        F.when(
            F.col("year_founded").isNotNull() & F.col("year").isNotNull(),
            F.col("year").cast("double") - F.col("year_founded").cast("double"),
        ).otherwise(F.lit(None)),
    )
    joined = joined.withColumn("firm_age", F.when(F.col("firm_age") < 0, F.lit(None)).otherwise(F.col("firm_age")))
    return joined.drop(parent_static["ultimate_parent_rcid"])


def add_first_event_years(frame: DataFrame, date_columns: list[str]) -> DataFrame:
    for column_name in date_columns:
        if column_name in frame.columns:
            year_name = column_name.replace("_date_", "_year_").replace("_date", "_year")
            frame = frame.withColumn(year_name, F.year(F.col(column_name)))
    return frame


def prefix_columns(frame: DataFrame, prefix: str, exclude: list[str]) -> DataFrame:
    renamed = frame
    for column_name in frame.columns:
        if column_name not in exclude:
            renamed = renamed.withColumnRenamed(column_name, f"{prefix}{column_name}")
    return renamed


def choose_primary_position(expanded_positions: DataFrame) -> DataFrame:
    ordering = Window.partitionBy("user_id", "year").orderBy(
        F.col("weight").desc(),
        F.col("start_date").desc_nulls_last(),
        F.col("position_id").asc_nulls_last(),
    )
    return (
        expanded_positions
        .withColumn("primary_rank", F.row_number().over(ordering))
        .where(F.col("primary_rank") == 1)
        .drop("primary_rank")
    )
