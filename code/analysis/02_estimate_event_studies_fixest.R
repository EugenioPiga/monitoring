#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  options(warn = 1)
})

parse_args <- function(raw_args) {
  args <- list(
    sample_dir = NULL,
    visibility_sample_dir = NULL,
    output_dir = NULL,
    config_path = NULL,
    outcomes = NULL,
    run_base = "1",
    run_stacked = "1",
    run_visibility = "0",
    run_visibility_stacked = "1"
  )
  index <- 1
  while (index <= length(raw_args)) {
    key <- raw_args[[index]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected positional argument: %s", key))
    }
    value <- if (index == length(raw_args)) NULL else raw_args[[index + 1]]
    name <- gsub("^--", "", key)
    name <- gsub("-", "_", name)
    if (!(name %in% names(args))) {
      stop(sprintf("Unknown argument: %s", key))
    }
    if (is.null(value) || startsWith(value, "--")) {
      stop(sprintf("Missing value for argument: %s", key))
    }
    args[[name]] <- value
    index <- index + 2
  }
  args
}

require_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      sprintf(
        "Missing required R packages: %s. Install them in the revelio environment before running estimation.",
        paste(missing, collapse = ", ")
      )
    )
  }
}

split_csv <- function(value) {
  if (is.null(value) || identical(value, "")) {
    return(character())
  }
  parts <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  parts[nzchar(parts)]
}

as_flag <- function(value) {
  tolower(as.character(value)) %in% c("1", "true", "t", "yes", "y")
}

event_dummy_name <- function(event_time, prefix = "event") {
  if (event_time < 0) {
    sprintf("%s_m%s", prefix, abs(event_time))
  } else {
    sprintf("%s_p%s", prefix, event_time)
  }
}

safe_visibility_name <- function(value) {
  cleaned <- tolower(gsub("[^A-Za-z0-9_]+", "_", value))
  cleaned <- gsub("^_+|_+$", "", cleaned)
  if (!nzchar(cleaned)) {
    return("visibility")
  }
  cleaned
}

visibility_interaction_name <- function(event_time, visibility_name, prefix = "event") {
  sprintf("%s_x_%s", event_dummy_name(event_time, prefix = prefix), safe_visibility_name(visibility_name))
}

build_event_map <- function(config, prefix = "event") {
  bin_min <- as.integer(config$event_time$bin_min)
  bin_max <- as.integer(config$event_time$bin_max)
  omit_event_time <- as.integer(config$event_time$omit_event_time)
  values <- seq(bin_min, bin_max)
  data.table::data.table(
    event_time = values,
    dummy_name = vapply(values, event_dummy_name, character(1), prefix = prefix),
    omitted = values == omit_event_time
  )
}

build_visibility_event_map <- function(config, visibility_name, prefix = "event") {
  base_map <- build_event_map(config, prefix = prefix)
  base_map[, term := vapply(event_time, visibility_interaction_name, character(1), visibility_name = visibility_name, prefix = prefix)]
  base_map
}

load_dataset <- function(path, columns) {
  ds <- arrow::open_dataset(path, format = "parquet")
  ds |>
    dplyr::select(dplyr::all_of(columns)) |>
    dplyr::collect() |>
    data.table::as.data.table()
}

dataset_columns <- function(path) {
  ds <- arrow::open_dataset(path, format = "parquet")
  as.character(ds$schema$names)
}

count_treated_parents <- function(dt, rhs_terms) {
  if (length(rhs_terms) == 0 || nrow(dt) == 0) {
    return(0L)
  }
  cols <- intersect(rhs_terms, names(dt))
  if (length(cols) == 0) {
    return(0L)
  }
  treated_flag <- rep(FALSE, nrow(dt))
  for (col in cols) {
    values <- dt[[col]]
    treated_flag <- treated_flag | (!is.na(values) & abs(values) > 0)
  }
  data.table::uniqueN(dt[treated_flag == TRUE, parent_rcid])
}

extract_wald <- function(model, lead_terms) {
  if (length(lead_terms) == 0) {
    return(list(stat = NA_real_, p = NA_real_))
  }
  keep_regex <- paste0("^(", paste(lead_terms, collapse = "|"), ")$")
  wald_result <- NULL
  invisible(capture.output({
    wald_result <- tryCatch(fixest::wald(model, keep = keep_regex), error = function(e) NULL)
  }))
  if (is.null(wald_result)) {
    return(list(stat = NA_real_, p = NA_real_))
  }
  stat <- NA_real_
  p_value <- NA_real_
  if (is.matrix(wald_result) || is.data.frame(wald_result)) {
    if ("stat" %in% colnames(wald_result)) stat <- suppressWarnings(as.numeric(wald_result[1, "stat"]))
    if ("p" %in% colnames(wald_result)) p_value <- suppressWarnings(as.numeric(wald_result[1, "p"]))
  } else if (is.list(wald_result)) {
    if (!is.null(wald_result$stat)) stat <- suppressWarnings(as.numeric(wald_result$stat))
    if (!is.null(wald_result$p)) p_value <- suppressWarnings(as.numeric(wald_result$p))
  } else if (is.atomic(wald_result) && !is.null(names(wald_result))) {
    if ("stat" %in% names(wald_result)) stat <- suppressWarnings(as.numeric(wald_result[["stat"]]))
    if ("p" %in% names(wald_result)) p_value <- suppressWarnings(as.numeric(wald_result[["p"]]))
  }
  list(stat = stat, p = p_value)
}

get_unique_count <- function(dt, candidates) {
  for (candidate in candidates) {
    if (candidate %in% names(dt)) {
      return(data.table::uniqueN(dt[[candidate]]))
    }
  }
  0L
}

empty_result <- function(estimator_id, estimator_label, outcome_cfg, visibility_variable = NA_character_, visibility_label = NA_character_) {
  list(
    coefficients = data.table::data.table(),
    pretrend_summary = data.table::data.table(
      estimator = estimator_id,
      estimator_label = estimator_label,
      outcome = outcome_cfg$name,
      outcome_label = outcome_cfg$label,
      outcome_group = outcome_cfg$group,
      visibility_variable = visibility_variable,
      visibility_label = visibility_label,
      n_pre_coefficients = 0L,
      mean_pre_estimate = NA_real_,
      max_abs_pre_estimate = NA_real_,
      max_abs_pre_tstat = NA_real_,
      n_sig_10 = NA_integer_,
      n_sig_05 = NA_integer_,
      n_sig_01 = NA_integer_,
      joint_test_statistic = NA_real_,
      joint_test_p_value = NA_real_,
      nobs = 0L,
      n_clusters = 0L,
      n_parents = 0L,
      n_parent_occ = 0L,
      n_occupations = 0L,
      n_years = 0L,
      mean_outcome = NA_real_,
      sd_outcome = NA_real_
    ),
    pretrend_leads = data.table::data.table(),
    status = data.table::data.table(
      estimator = estimator_id,
      estimator_label = estimator_label,
      outcome = outcome_cfg$name,
      visibility_variable = visibility_variable,
      visibility_label = visibility_label,
      status = "skipped",
      note = "empty_result",
      nobs = 0L,
      n_clusters = 0L,
      treated_parents = 0L,
      n_parents = 0L,
      n_parent_occ = 0L,
      n_occupations = 0L,
      n_years = 0L
    )
  )
}

run_event_model <- function(
  dt,
  outcome_cfg,
  rhs_terms,
  fe_terms,
  estimator_id,
  estimator_label,
  event_map,
  visibility_variable = NA_character_,
  visibility_label = NA_character_,
  required_filter_col = NULL,
  minimum_rows = 0L,
  minimum_clusters = 1L
) {
  event_lookup <- data.table::copy(event_map)
  if ("dummy_name" %in% names(event_lookup) && !("term" %in% names(event_lookup))) {
    data.table::setnames(event_lookup, "dummy_name", "term")
  }
  working <- data.table::copy(dt[!is.na(get(outcome_cfg$name))])
  if (!is.null(required_filter_col)) {
    working <- working[!is.na(get(required_filter_col))]
  }
  if (nrow(working) == 0) {
    result <- empty_result(estimator_id, estimator_label, outcome_cfg, visibility_variable, visibility_label)
    result$status[, note := "no_nonmissing_rows"]
    return(result)
  }

  n_parents <- data.table::uniqueN(working$parent_rcid)
  n_parent_occ <- get_unique_count(working, c("parent_occ_fe", "stack_parent_occ_fe"))
  n_occupations <- get_unique_count(working, c("occupation"))
  n_years <- get_unique_count(working, c("year"))
  treated_parents <- count_treated_parents(working, rhs_terms)
  mean_outcome <- mean(working[[outcome_cfg$name]], na.rm = TRUE)
  sd_outcome <- stats::sd(working[[outcome_cfg$name]], na.rm = TRUE)

  if (nrow(working) < minimum_rows) {
    result <- empty_result(estimator_id, estimator_label, outcome_cfg, visibility_variable, visibility_label)
    result$pretrend_summary[, `:=`(
      n_parents = n_parents,
      n_parent_occ = n_parent_occ,
      n_occupations = n_occupations,
      n_years = n_years,
      mean_outcome = mean_outcome,
      sd_outcome = sd_outcome
    )]
    result$status[, `:=`(
      note = sprintf("too_few_rows_below_threshold_%s", minimum_rows),
      nobs = nrow(working),
      n_clusters = n_parents,
      treated_parents = treated_parents,
      n_parents = n_parents,
      n_parent_occ = n_parent_occ,
      n_occupations = n_occupations,
      n_years = n_years
    )]
    return(result)
  }
  if (n_parents < minimum_clusters) {
    result <- empty_result(estimator_id, estimator_label, outcome_cfg, visibility_variable, visibility_label)
    result$pretrend_summary[, `:=`(
      nobs = nrow(working),
      n_clusters = n_parents,
      n_parents = n_parents,
      n_parent_occ = n_parent_occ,
      n_occupations = n_occupations,
      n_years = n_years,
      mean_outcome = mean_outcome,
      sd_outcome = sd_outcome
    )]
    result$status[, `:=`(
      note = sprintf("too_few_clusters_below_threshold_%s", minimum_clusters),
      nobs = nrow(working),
      n_clusters = n_parents,
      treated_parents = treated_parents,
      n_parents = n_parents,
      n_parent_occ = n_parent_occ,
      n_occupations = n_occupations,
      n_years = n_years
    )]
    return(result)
  }

  formula_text <- sprintf(
    "%s ~ %s | %s",
    outcome_cfg$name,
    paste(rhs_terms, collapse = " + "),
    paste(fe_terms, collapse = " + ")
  )
  model <- tryCatch(
    fixest::feols(stats::as.formula(formula_text), data = working, vcov = ~parent_rcid, warn = FALSE, notes = FALSE),
    error = function(e) e
  )
  if (inherits(model, "error")) {
    result <- empty_result(estimator_id, estimator_label, outcome_cfg, visibility_variable, visibility_label)
    result$pretrend_summary[, `:=`(
      nobs = nrow(working),
      n_clusters = n_parents,
      n_parents = n_parents,
      n_parent_occ = n_parent_occ,
      n_occupations = n_occupations,
      n_years = n_years,
      mean_outcome = mean_outcome,
      sd_outcome = sd_outcome
    )]
    result$status[, `:=`(
      status = "failed",
      note = conditionMessage(model),
      nobs = nrow(working),
      n_clusters = n_parents,
      treated_parents = treated_parents,
      n_parents = n_parents,
      n_parent_occ = n_parent_occ,
      n_occupations = n_occupations,
      n_years = n_years
    )]
    return(result)
  }

  coefs <- stats::coef(model)
  ses <- fixest::se(model)
  coef_dt <- data.table::data.table(
    term = names(coefs),
    estimate = as.numeric(coefs),
    std_error = as.numeric(ses)
  )
  coef_dt <- merge(coef_dt, event_lookup[, .(term, event_time)], by = "term", all.x = TRUE, sort = FALSE)
  coef_dt[, `:=`(
    statistic = estimate / std_error,
    p_value = 2 * stats::pnorm(-abs(estimate / std_error)),
    conf_low = estimate - 1.96 * std_error,
    conf_high = estimate + 1.96 * std_error,
    estimator = estimator_id,
    estimator_label = estimator_label,
    outcome = outcome_cfg$name,
    outcome_label = outcome_cfg$label,
    outcome_group = outcome_cfg$group,
    visibility_variable = visibility_variable,
    visibility_label = visibility_label,
    nobs = stats::nobs(model),
    n_clusters = n_parents,
    n_parents = n_parents,
    n_parent_occ = n_parent_occ,
    n_occupations = n_occupations,
    n_years = n_years,
    mean_outcome = mean_outcome,
    sd_outcome = sd_outcome
  )]
  data.table::setcolorder(
    coef_dt,
    c(
      "estimator", "estimator_label", "outcome", "outcome_label", "outcome_group",
      "visibility_variable", "visibility_label",
      "term", "event_time", "estimate", "std_error", "statistic", "p_value",
      "conf_low", "conf_high", "nobs", "n_clusters", "n_parents", "n_parent_occ",
      "n_occupations", "n_years", "mean_outcome", "sd_outcome"
    )
  )
  coef_dt <- coef_dt[order(event_time)]

  pretrend_leads <- coef_dt[event_time <= -2]
  lead_terms <- pretrend_leads$term
  wald_info <- extract_wald(model, lead_terms)
  pretrend_summary <- data.table::data.table(
    estimator = estimator_id,
    estimator_label = estimator_label,
    outcome = outcome_cfg$name,
    outcome_label = outcome_cfg$label,
    outcome_group = outcome_cfg$group,
    visibility_variable = visibility_variable,
    visibility_label = visibility_label,
    n_pre_coefficients = nrow(pretrend_leads),
    mean_pre_estimate = if (nrow(pretrend_leads) > 0) mean(pretrend_leads$estimate, na.rm = TRUE) else NA_real_,
    max_abs_pre_estimate = if (nrow(pretrend_leads) > 0) max(abs(pretrend_leads$estimate), na.rm = TRUE) else NA_real_,
    max_abs_pre_tstat = if (nrow(pretrend_leads) > 0) max(abs(pretrend_leads$statistic), na.rm = TRUE) else NA_real_,
    n_sig_10 = if (nrow(pretrend_leads) > 0) sum(pretrend_leads$p_value < 0.10, na.rm = TRUE) else 0L,
    n_sig_05 = if (nrow(pretrend_leads) > 0) sum(pretrend_leads$p_value < 0.05, na.rm = TRUE) else 0L,
    n_sig_01 = if (nrow(pretrend_leads) > 0) sum(pretrend_leads$p_value < 0.01, na.rm = TRUE) else 0L,
    joint_test_statistic = wald_info$stat,
    joint_test_p_value = wald_info$p,
    nobs = stats::nobs(model),
    n_clusters = n_parents,
    n_parents = n_parents,
    n_parent_occ = n_parent_occ,
    n_occupations = n_occupations,
    n_years = n_years,
    mean_outcome = mean_outcome,
    sd_outcome = sd_outcome
  )

  status <- data.table::data.table(
    estimator = estimator_id,
    estimator_label = estimator_label,
    outcome = outcome_cfg$name,
    visibility_variable = visibility_variable,
    visibility_label = visibility_label,
    status = "ok",
    note = "",
    nobs = stats::nobs(model),
    n_clusters = n_parents,
    treated_parents = treated_parents,
    n_parents = n_parents,
    n_parent_occ = n_parent_occ,
    n_occupations = n_occupations,
    n_years = n_years
  )

  list(
    coefficients = coef_dt,
    pretrend_summary = pretrend_summary,
    pretrend_leads = pretrend_leads[, .(
      estimator, estimator_label, outcome, outcome_label, outcome_group,
      visibility_variable, visibility_label, term, event_time, estimate,
      std_error, statistic, p_value, conf_low, conf_high
    )],
    status = status
  )
}

write_skip_memo <- function(path, lines) {
  writeLines(lines, con = path, useBytes = TRUE)
}

run_base_estimators <- function(args, config, outcomes_cfg) {
  results_dir <- file.path(args$output_dir, "results")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  base_sample_path <- file.path(args$sample_dir, "parent_occ_event_study_sample.parquet")
  stacked_sample_path <- file.path(args$sample_dir, "parent_occ_event_study_stacked_sample.parquet")
  if (!dir.exists(base_sample_path)) {
    stop(sprintf("Base sample parquet directory not found: %s", base_sample_path))
  }
  base_columns <- dataset_columns(base_sample_path)
  outcomes_cfg <- outcomes_cfg[name %in% base_columns]
  if (nrow(outcomes_cfg) == 0) {
    write_skip_memo(
      file.path(results_dir, "99_results_memo.txt"),
      c(
        "Base event-study estimation skipped.",
        "No requested outcomes were present in the base sample."
      )
    )
    return(invisible(NULL))
  }

  event_map <- build_event_map(config, prefix = "event")
  rhs_terms_twfe <- event_map[omitted == FALSE, dummy_name]
  needed_base_cols <- unique(c(
    "parent_rcid", "occupation", "year", "parent_occ_fe", "occupation_year_fe",
    outcomes_cfg$name, rhs_terms_twfe
  ))
  base_dt <- load_dataset(base_sample_path, needed_base_cols)

  coefficient_tables <- list()
  pretrend_summary_tables <- list()
  pretrend_lead_tables <- list()
  status_tables <- list()

  for (row_id in seq_len(nrow(outcomes_cfg))) {
    outcome_cfg <- outcomes_cfg[row_id]
    result <- run_event_model(
      dt = base_dt,
      outcome_cfg = outcome_cfg,
      rhs_terms = rhs_terms_twfe,
      fe_terms = c("parent_occ_fe", "occupation_year_fe"),
      estimator_id = "twfe_parent_occ_occ_year",
      estimator_label = "TWFE with parent x occupation FE and occupation x year FE",
      event_map = event_map
    )
    coefficient_tables[[length(coefficient_tables) + 1L]] <- result$coefficients
    pretrend_summary_tables[[length(pretrend_summary_tables) + 1L]] <- result$pretrend_summary
    pretrend_lead_tables[[length(pretrend_lead_tables) + 1L]] <- result$pretrend_leads
    status_tables[[length(status_tables) + 1L]] <- result$status
  }

  if (as_flag(args$run_stacked) && dir.exists(stacked_sample_path)) {
    stack_event_map <- build_event_map(config, prefix = "stack_event")
    rhs_terms_stacked <- stack_event_map[omitted == FALSE, dummy_name]
    needed_stacked_cols <- unique(c(
      "parent_rcid", "occupation", "year", "parent_occ_fe", "stack_parent_occ_fe",
      "stack_occupation_year_fe", outcomes_cfg$name, rhs_terms_stacked
    ))
    stacked_dt <- load_dataset(stacked_sample_path, needed_stacked_cols)
    for (row_id in seq_len(nrow(outcomes_cfg))) {
      outcome_cfg <- outcomes_cfg[row_id]
      result <- run_event_model(
        dt = stacked_dt,
        outcome_cfg = outcome_cfg,
        rhs_terms = rhs_terms_stacked,
        fe_terms = c("stack_parent_occ_fe", "stack_occupation_year_fe"),
        estimator_id = "stacked_not_yet_treated",
        estimator_label = "Stacked cohort event study with not-yet-treated and never-treated controls",
        event_map = data.table::data.table(term = rhs_terms_stacked, event_time = stack_event_map[omitted == FALSE, event_time])
      )
      coefficient_tables[[length(coefficient_tables) + 1L]] <- result$coefficients
      pretrend_summary_tables[[length(pretrend_summary_tables) + 1L]] <- result$pretrend_summary
      pretrend_lead_tables[[length(pretrend_lead_tables) + 1L]] <- result$pretrend_leads
      status_tables[[length(status_tables) + 1L]] <- result$status
    }
  } else {
    status_tables[[length(status_tables) + 1L]] <- data.table::data.table(
      estimator = "stacked_not_yet_treated",
      estimator_label = "Stacked cohort event study with not-yet-treated and never-treated controls",
      outcome = NA_character_,
      visibility_variable = NA_character_,
      visibility_label = NA_character_,
      status = "skipped",
      note = if (as_flag(args$run_stacked)) "stacked_sample_missing" else "run_stacked_disabled",
      nobs = 0L,
      n_clusters = 0L,
      treated_parents = 0L,
      n_parents = 0L,
      n_parent_occ = 0L,
      n_occupations = 0L,
      n_years = 0L
    )
  }

  coefficients_dt <- data.table::rbindlist(coefficient_tables, use.names = TRUE, fill = TRUE)
  pretrend_summary_dt <- data.table::rbindlist(pretrend_summary_tables, use.names = TRUE, fill = TRUE)
  pretrend_leads_dt <- data.table::rbindlist(pretrend_lead_tables, use.names = TRUE, fill = TRUE)
  status_dt <- data.table::rbindlist(status_tables, use.names = TRUE, fill = TRUE)

  metadata <- list(
    sample_dir = args$sample_dir,
    output_dir = normalizePath(args$output_dir, mustWork = FALSE),
    cluster_var = config$metadata$cluster_var,
    baseline_fixed_effects = unname(unlist(config$metadata[c("baseline_unit_fe", "baseline_time_fe")])),
    stacked_fixed_effects = unname(unlist(config$metadata[c("stacked_unit_fe", "stacked_time_fe")])),
    omit_event_time = as.integer(config$event_time$omit_event_time),
    event_time_min = as.integer(config$event_time$bin_min),
    event_time_max = as.integer(config$event_time$bin_max)
  )

  jsonlite::write_json(metadata, file.path(results_dir, "01_estimation_metadata.json"), pretty = TRUE, auto_unbox = TRUE)
  data.table::fwrite(coefficients_dt, file.path(results_dir, "02_event_study_coefficients.csv"))
  data.table::fwrite(pretrend_summary_dt, file.path(results_dir, "03_pretrend_summary.csv"))
  data.table::fwrite(pretrend_leads_dt, file.path(results_dir, "04_pretrend_leads.csv"))
  data.table::fwrite(status_dt, file.path(results_dir, "05_model_status.csv"))
}

run_visibility_estimators <- function(args, config, outcomes_cfg) {
  visibility_results_dir <- file.path(args$output_dir, "visibility_results")
  dir.create(visibility_results_dir, recursive = TRUE, showWarnings = FALSE)

  visibility_sample_dir <- if (!is.null(args$visibility_sample_dir) && nzchar(args$visibility_sample_dir)) args$visibility_sample_dir else file.path(args$output_dir, "visibility_sample")
  sample_path <- file.path(visibility_sample_dir, "parent_occ_visibility_event_study_sample.parquet")
  stacked_sample_path <- file.path(visibility_sample_dir, "parent_occ_visibility_event_study_stacked_sample.parquet")
  visibility_summary_path <- file.path(visibility_sample_dir, "01_visibility_variable_summary.csv")
  support_path <- file.path(visibility_sample_dir, "02_visibility_event_time_support.csv")
  missingness_path <- file.path(visibility_sample_dir, "03_visibility_missingness.csv")

  if (!dir.exists(sample_path)) {
    write_skip_memo(
      file.path(visibility_results_dir, "99_visibility_results_memo.txt"),
      c(
        "Visibility estimation skipped.",
        sprintf("Visibility sample parquet directory not found: %s", sample_path)
      )
    )
    return(invisible(NULL))
  }
  sample_columns <- dataset_columns(sample_path)
  outcomes_cfg <- outcomes_cfg[name %in% sample_columns]
  if (nrow(outcomes_cfg) == 0) {
    write_skip_memo(
      file.path(visibility_results_dir, "99_visibility_results_memo.txt"),
      c(
        "Visibility estimation skipped.",
        "No requested outcomes were present in the visibility sample."
      )
    )
    return(invisible(NULL))
  }
  if (!file.exists(visibility_summary_path)) {
    stop(sprintf("Visibility variable summary not found: %s", visibility_summary_path))
  }

  visibility_summary <- data.table::fread(visibility_summary_path)
  data.table::fwrite(visibility_summary, file.path(visibility_results_dir, "00_visibility_variable_summary.csv"))
  if (file.exists(support_path)) {
    cat("[visibility] support by event time\n")
    print(data.table::fread(support_path))
  }
  if (file.exists(missingness_path)) {
    cat("[visibility] missingness by visibility variable\n")
    print(data.table::fread(missingness_path))
  }

  minimum_rows <- as.integer(config$visibility_event_studies$minimum_rows %||% 5000L)
  minimum_clusters <- as.integer(config$visibility_event_studies$minimum_parent_clusters %||% 25L)

  valid_visibility <- visibility_summary[skip_regression == 0]
  if (nrow(valid_visibility) == 0) {
    write_skip_memo(
      file.path(visibility_results_dir, "99_visibility_results_memo.txt"),
      c(
        "Visibility estimation skipped.",
        "All configured visibility variables were flagged as unusable in the sample builder.",
        paste("Variables:", paste(visibility_summary$visibility_variable, collapse = ", "))
      )
    )
    data.table::fwrite(visibility_summary, file.path(visibility_results_dir, "00_visibility_variable_summary.csv"))
    return(invisible(NULL))
  }

  coefficient_tables <- list()
  pretrend_summary_tables <- list()
  status_tables <- list()

  for (row_id in seq_len(nrow(valid_visibility))) {
    visibility_row <- valid_visibility[row_id]
    visibility_name <- visibility_row$visibility_variable
    visibility_label <- visibility_row$visibility_label
    safe_name <- visibility_row$safe_name
    std_col <- sprintf("%s_std", safe_name)
    event_map <- build_visibility_event_map(config, visibility_name, prefix = "event")
    rhs_terms <- event_map[omitted == FALSE, term]
    needed_cols <- unique(c(
      "parent_rcid", "occupation", "year", "parent_occ_fe", "parent_year_fe", "occupation_year_fe",
      std_col, outcomes_cfg$name, rhs_terms
    ))
    dt <- load_dataset(sample_path, needed_cols)
    cat(sprintf("[visibility] estimator=twfe_visibility_parent_year variable=%s rows=%s parents=%s occupations=%s years=%s\n",
                visibility_name, nrow(dt), data.table::uniqueN(dt$parent_rcid), data.table::uniqueN(dt$occupation), data.table::uniqueN(dt$year)))
    for (outcome_id in seq_len(nrow(outcomes_cfg))) {
      outcome_cfg <- outcomes_cfg[outcome_id]
      result <- run_event_model(
        dt = dt,
        outcome_cfg = outcome_cfg,
        rhs_terms = rhs_terms,
        fe_terms = c("parent_occ_fe", "parent_year_fe", "occupation_year_fe"),
        estimator_id = "twfe_visibility_parent_year",
        estimator_label = "Visibility-interacted TWFE with parent x occupation, parent x year, and occupation x year FE",
        event_map = event_map[omitted == FALSE, .(term, event_time)],
        visibility_variable = visibility_name,
        visibility_label = visibility_label,
        required_filter_col = std_col,
        minimum_rows = minimum_rows,
        minimum_clusters = minimum_clusters
      )
      coefficient_tables[[length(coefficient_tables) + 1L]] <- result$coefficients
      pretrend_summary_tables[[length(pretrend_summary_tables) + 1L]] <- result$pretrend_summary
      status_tables[[length(status_tables) + 1L]] <- result$status
    }
  }

  if (as_flag(args$run_visibility_stacked) && dir.exists(stacked_sample_path)) {
    for (row_id in seq_len(nrow(valid_visibility))) {
      visibility_row <- valid_visibility[row_id]
      visibility_name <- visibility_row$visibility_variable
      visibility_label <- visibility_row$visibility_label
      safe_name <- visibility_row$safe_name
      std_col <- sprintf("%s_std", safe_name)
      event_map <- build_visibility_event_map(config, visibility_name, prefix = "stack_event")
      rhs_terms <- event_map[omitted == FALSE, term]
      needed_cols <- unique(c(
        "parent_rcid", "occupation", "year", "parent_occ_fe",
        "stack_parent_occ_fe", "stack_parent_year_fe", "stack_occupation_year_fe",
        std_col, outcomes_cfg$name, rhs_terms
      ))
      dt <- load_dataset(stacked_sample_path, needed_cols)
      cat(sprintf("[visibility] estimator=stacked_visibility_not_yet_treated variable=%s rows=%s parents=%s occupations=%s years=%s\n",
                  visibility_name, nrow(dt), data.table::uniqueN(dt$parent_rcid), data.table::uniqueN(dt$occupation), data.table::uniqueN(dt$year)))
      for (outcome_id in seq_len(nrow(outcomes_cfg))) {
        outcome_cfg <- outcomes_cfg[outcome_id]
        result <- run_event_model(
          dt = dt,
          outcome_cfg = outcome_cfg,
          rhs_terms = rhs_terms,
          fe_terms = c("stack_parent_occ_fe", "stack_parent_year_fe", "stack_occupation_year_fe"),
          estimator_id = "stacked_visibility_not_yet_treated",
          estimator_label = "Stacked visibility event study with not-yet-treated and never-treated controls",
          event_map = event_map[omitted == FALSE, .(term, event_time)],
          visibility_variable = visibility_name,
          visibility_label = visibility_label,
          required_filter_col = std_col,
          minimum_rows = minimum_rows,
          minimum_clusters = minimum_clusters
        )
        coefficient_tables[[length(coefficient_tables) + 1L]] <- result$coefficients
        pretrend_summary_tables[[length(pretrend_summary_tables) + 1L]] <- result$pretrend_summary
        status_tables[[length(status_tables) + 1L]] <- result$status
      }
    }
  } else {
    status_tables[[length(status_tables) + 1L]] <- data.table::data.table(
      estimator = "stacked_visibility_not_yet_treated",
      estimator_label = "Stacked visibility event study with not-yet-treated and never-treated controls",
      outcome = NA_character_,
      visibility_variable = NA_character_,
      visibility_label = NA_character_,
      status = "skipped",
      note = if (as_flag(args$run_visibility_stacked)) "stacked_visibility_sample_missing" else "run_visibility_stacked_disabled",
      nobs = 0L,
      n_clusters = 0L,
      treated_parents = 0L,
      n_parents = 0L,
      n_parent_occ = 0L,
      n_occupations = 0L,
      n_years = 0L
    )
  }

  coefficients_dt <- data.table::rbindlist(coefficient_tables, use.names = TRUE, fill = TRUE)
  pretrend_summary_dt <- data.table::rbindlist(pretrend_summary_tables, use.names = TRUE, fill = TRUE)
  status_dt <- data.table::rbindlist(status_tables, use.names = TRUE, fill = TRUE)

  metadata <- list(
    visibility_sample_dir = visibility_sample_dir,
    output_dir = normalizePath(args$output_dir, mustWork = FALSE),
    cluster_var = config$metadata$cluster_var,
    visibility_fixed_effects = unname(unlist(config$metadata[c("visibility_unit_fe", "visibility_parent_time_fe", "visibility_occ_time_fe")])),
    stacked_visibility_fixed_effects = unname(unlist(config$metadata[c("stacked_visibility_unit_fe", "stacked_visibility_parent_time_fe", "stacked_visibility_occ_time_fe")])),
    omit_event_time = as.integer(config$event_time$omit_event_time),
    event_time_min = as.integer(config$event_time$bin_min),
    event_time_max = as.integer(config$event_time$bin_max)
  )

  jsonlite::write_json(metadata, file.path(visibility_results_dir, "01_visibility_estimation_metadata.json"), pretty = TRUE, auto_unbox = TRUE)
  data.table::fwrite(coefficients_dt, file.path(visibility_results_dir, "02_visibility_event_study_coefficients.csv"))
  data.table::fwrite(pretrend_summary_dt, file.path(visibility_results_dir, "03_visibility_pretrend_summary.csv"))
  data.table::fwrite(status_dt, file.path(visibility_results_dir, "04_visibility_model_status.csv"))

  if (nrow(coefficients_dt) == 0) {
    write_skip_memo(
      file.path(visibility_results_dir, "99_visibility_results_memo.txt"),
      c(
        "Visibility estimation produced no coefficient rows.",
        "Consult 04_visibility_model_status.csv for per-model skip/failure reasons."
      )
    )
  }
}

`%||%` <- function(left, right) {
  if (is.null(left)) right else left
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  require_packages(c("arrow", "data.table", "dplyr", "fixest", "jsonlite"))

  library(arrow)
  library(data.table)
  library(dplyr)
  library(fixest)
  library(jsonlite)

  config <- jsonlite::fromJSON(args$config_path, simplifyVector = TRUE)
  outcomes_cfg <- data.table::as.data.table(config$outcomes)
  selected_outcomes <- split_csv(args$outcomes)
  if (length(selected_outcomes) > 0) {
    outcomes_cfg <- outcomes_cfg[name %in% selected_outcomes]
  }
  if (nrow(outcomes_cfg) == 0) {
    stop("No outcomes selected for estimation.")
  }

  cat(sprintf("[setup] outcomes used: %s\n", paste(outcomes_cfg$name, collapse = ", ")))
  if (as_flag(args$run_base)) {
    run_base_estimators(args, config, outcomes_cfg)
  }
  if (as_flag(args$run_visibility)) {
    configured_visibility <- if (is.data.frame(config$visibility_event_studies$visibility_variables)) {
      as.character(config$visibility_event_studies$visibility_variables$name)
    } else {
      vapply(config$visibility_event_studies$visibility_variables, function(x) if (is.list(x)) x$name else as.character(x), character(1))
    }
    cat(sprintf("[setup] visibility variables configured: %s\n", paste(configured_visibility, collapse = ", ")))
    run_visibility_estimators(args, config, outcomes_cfg)
  }
}

main()
