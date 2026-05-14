#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  options(warn = 1)
})

parse_args <- function(raw_args) {
  args <- list(
    sample_path = NULL,
    output_dir = NULL,
    config_path = NULL,
    outcomes = NULL,
    treatments = NULL,
    run_advanced = "0",
    run_heterogeneity = "0"
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

as_flag <- function(value) {
  tolower(as.character(value)) %in% c("1", "true", "t", "yes", "y")
}

split_csv <- function(value) {
  if (is.null(value) || identical(value, "")) {
    return(character())
  }
  parts <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  parts[nzchar(parts)]
}

sanitize_tag <- function(value) {
  clean <- gsub("[^A-Za-z0-9]+", "_", value)
  clean <- gsub("^_+|_+$", "", clean)
  tolower(clean)
}

event_tag <- function(event_time) {
  if (event_time < 0) {
    sprintf("m%s", abs(event_time))
  } else {
    sprintf("p%s", event_time)
  }
}

prepare_event_dummies <- function(dt, prefix) {
  treated_col <- paste0(prefix, "_balanced_treated")
  event_col <- paste0(prefix, "_event_time_binned")
  dt[, treated_group_tmp := as.integer(get(treated_col) == 1)]
  available_times <- sort(unique(dt[treated_group_tmp == 1 & !is.na(get(event_col)), as.integer(get(event_col))]))
  available_times <- available_times[available_times != -1]
  dummy_map <- data.table::data.table(
    event_time = integer(),
    dummy_name = character()
  )
  if (length(available_times) == 0) {
    return(dummy_map)
  }
  for (event_time in available_times) {
    dummy_name <- sprintf("evt_%s", event_tag(event_time))
    dt[, (dummy_name) := as.integer(treated_group_tmp == 1 & get(event_col) == event_time)]
    dummy_map <- data.table::rbindlist(
      list(dummy_map, data.table::data.table(event_time = event_time, dummy_name = dummy_name)),
      use.names = TRUE
    )
  }
  dummy_map
}

extract_pretrend <- function(model, dummy_map, estimator, model_id, spec_id, outcome, subset_label) {
  leads <- dummy_map[event_time < -1, dummy_name]
  if (length(leads) == 0) {
    return(data.table::data.table(
      model_id = model_id,
      spec_id = spec_id,
      outcome = outcome,
      estimator = estimator,
      subset_label = subset_label,
      n_lead_terms = 0L,
      wald_stat = NA_real_,
      p_value = NA_real_,
      df1 = NA_real_,
      df2 = NA_real_
    ))
  }
  keep_regex <- paste0("^(", paste(leads, collapse = "|"), ")$")
  test <- tryCatch(
    fixest::wald(model, keep = keep_regex),
    error = function(e) NULL
  )
  if (is.null(test)) {
    return(data.table::data.table(
      model_id = model_id,
      spec_id = spec_id,
      outcome = outcome,
      estimator = estimator,
      subset_label = subset_label,
      n_lead_terms = length(leads),
      wald_stat = NA_real_,
      p_value = NA_real_,
      df1 = NA_real_,
      df2 = NA_real_
    ))
  }
  data.table::data.table(
    model_id = model_id,
    spec_id = spec_id,
    outcome = outcome,
    estimator = estimator,
    subset_label = subset_label,
    n_lead_terms = length(leads),
    wald_stat = as.numeric(test[1, "stat"]),
    p_value = as.numeric(test[1, "p"]),
    df1 = as.numeric(test[1, "df1"]),
    df2 = as.numeric(test[1, "df2"])
  )
}

quick_plot <- function(frame, output_png, title, outcome_label) {
  if (nrow(frame) == 0) {
    return(invisible(NULL))
  }
  grDevices::png(output_png, width = 900, height = 600, res = 140)
  on.exit(grDevices::dev.off(), add = TRUE)
  x <- frame$event_time
  y <- frame$estimate
  lower <- frame$ci_low
  upper <- frame$ci_high
  graphics::plot(
    x,
    y,
    type = "b",
    pch = 16,
    col = "#1b4965",
    lwd = 2,
    xlab = "Event time",
    ylab = outcome_label,
    main = title,
    ylim = range(c(lower, upper, 0), na.rm = TRUE)
  )
  graphics::segments(x0 = x, y0 = lower, x1 = x, y1 = upper, col = "#1b4965")
  graphics::abline(h = 0, col = "#333333", lwd = 1)
  graphics::abline(v = -1, col = "#777777", lty = 2, lwd = 1)
}

estimate_twfe_model <- function(dt, outcome_cfg, spec_cfg, quicklook_dir) {
  treatment_name <- spec_cfg$treatment_name
  prefix <- treatment_name
  analysis_col <- paste0(prefix, "_analysis_row")
  subset_label <- spec_cfg$subset_label
  working <- data.table::copy(dt[get(analysis_col) == 1])

  if (spec_cfg$both_only) {
    working <- working[has_both_data_by_year == 1]
  }
  if (!is.null(spec_cfg$subset_col)) {
    working <- working[get(spec_cfg$subset_col) == 1]
  }
  working <- working[!is.na(get(outcome_cfg$name))]
  if (nrow(working) == 0) {
    return(list(status = data.table::data.table(
      model_id = spec_cfg$model_id,
      spec_id = spec_cfg$spec_id,
      outcome = outcome_cfg$name,
      outcome_group = outcome_cfg$group,
      treatment_name = treatment_name,
      subset_label = subset_label,
      estimator = "twfe",
      status = "skipped",
      note = "no_nonmissing_rows_after_filters",
      nobs = 0L,
      treated_firms = 0L
    )))
  }

  dummy_map <- prepare_event_dummies(working, prefix)
  if (nrow(dummy_map) == 0) {
    return(list(status = data.table::data.table(
      model_id = spec_cfg$model_id,
      spec_id = spec_cfg$spec_id,
      outcome = outcome_cfg$name,
      outcome_group = outcome_cfg$group,
      treatment_name = treatment_name,
      subset_label = subset_label,
      estimator = "twfe",
      status = "skipped",
      note = "no_supported_event_dummies",
      nobs = nrow(working),
      treated_firms = uniqueN(working[treated_group_tmp == 1, firm_key])
    )))
  }

  rhs <- paste(dummy_map$dummy_name, collapse = " + ")
  fe_rhs <- if (identical(spec_cfg$extra_fe, "naics2_year")) {
    "firm_key + year + naics2^year"
  } else {
    "firm_key + year"
  }
  model_formula <- stats::as.formula(sprintf("%s ~ %s | %s", outcome_cfg$name, rhs, fe_rhs))
  model <- tryCatch(
    fixest::feols(model_formula, data = working, vcov = ~firm_key, warn = FALSE, notes = FALSE),
    error = function(e) e
  )
  if (inherits(model, "error")) {
    return(list(status = data.table::data.table(
      model_id = spec_cfg$model_id,
      spec_id = spec_cfg$spec_id,
      outcome = outcome_cfg$name,
      outcome_group = outcome_cfg$group,
      treatment_name = treatment_name,
      subset_label = subset_label,
      estimator = "twfe",
      status = "failed",
      note = conditionMessage(model),
      nobs = nrow(working),
      treated_firms = uniqueN(working[treated_group_tmp == 1, firm_key])
    )))
  }

  estimates <- stats::coef(model)
  std_errors <- fixest::se(model)
  estimate_dt <- data.table::data.table(
    term = names(estimates),
    estimate = as.numeric(estimates),
    std_error = as.numeric(std_errors)
  )
  estimate_dt <- merge(estimate_dt, dummy_map, by.x = "term", by.y = "dummy_name", all.x = TRUE, sort = FALSE)
  estimate_dt[, `:=`(
    ci_low = estimate - 1.96 * std_error,
    ci_high = estimate + 1.96 * std_error,
    model_id = spec_cfg$model_id,
    spec_id = spec_cfg$spec_id,
    spec_label = spec_cfg$spec_label,
    estimator = "twfe",
    treatment_name = treatment_name,
    outcome = outcome_cfg$name,
    outcome_label = outcome_cfg$label,
    outcome_group = outcome_cfg$group,
    subset_label = subset_label,
    nobs = fixest::nobs(model)
  )]
  data.table::setcolorder(
    estimate_dt,
    c("model_id", "spec_id", "spec_label", "estimator", "treatment_name", "outcome", "outcome_label", "outcome_group", "subset_label", "term", "event_time", "estimate", "std_error", "ci_low", "ci_high", "nobs")
  )

  pretrend <- extract_pretrend(model, dummy_map, "twfe", spec_cfg$model_id, spec_cfg$spec_id, outcome_cfg$name, subset_label)
  status <- data.table::data.table(
    model_id = spec_cfg$model_id,
    spec_id = spec_cfg$spec_id,
    outcome = outcome_cfg$name,
    outcome_group = outcome_cfg$group,
    treatment_name = treatment_name,
    subset_label = subset_label,
    estimator = "twfe",
    status = "ok",
    note = "",
    nobs = fixest::nobs(model),
    treated_firms = uniqueN(working[treated_group_tmp == 1, firm_key])
  )

  plot_title <- sprintf("%s | %s", outcome_cfg$label, spec_cfg$spec_label)
  plot_path <- file.path(quicklook_dir, sprintf("%s.png", spec_cfg$model_id))
  quick_plot(estimate_dt[order(event_time)], plot_path, plot_title, outcome_cfg$label)
  list(coefficients = estimate_dt, pretrend = pretrend, status = status)
}

estimate_sunab_model <- function(dt, outcome_cfg, spec_cfg, quicklook_dir) {
  treatment_name <- spec_cfg$treatment_name
  analysis_col <- paste0(treatment_name, "_analysis_row")
  working <- data.table::copy(dt[get(analysis_col) == 1])
  working <- working[!is.na(get(outcome_cfg$name))]
  if (nrow(working) == 0) {
    return(list(status = data.table::data.table(
      model_id = spec_cfg$model_id,
      spec_id = spec_cfg$spec_id,
      outcome = outcome_cfg$name,
      outcome_group = outcome_cfg$group,
      treatment_name = treatment_name,
      subset_label = "all",
      estimator = "sunab",
      status = "skipped",
      note = "no_nonmissing_rows_after_filters",
      nobs = 0L,
      treated_firms = 0L
    )))
  }
  model_formula <- stats::as.formula(
    sprintf(
      "%s ~ fixest::sunab(first_people_analytics_firm_year_any_enriched, year, ref.p = -1) | firm_key + year",
      outcome_cfg$name
    )
  )
  model <- tryCatch(
    fixest::feols(model_formula, data = working, vcov = ~firm_key, warn = FALSE, notes = FALSE),
    error = function(e) e
  )
  if (inherits(model, "error")) {
    return(list(status = data.table::data.table(
      model_id = spec_cfg$model_id,
      spec_id = spec_cfg$spec_id,
      outcome = outcome_cfg$name,
      outcome_group = outcome_cfg$group,
      treatment_name = treatment_name,
      subset_label = "all",
      estimator = "sunab",
      status = "failed",
      note = conditionMessage(model),
      nobs = nrow(working),
      treated_firms = uniqueN(working[first_people_analytics_firm_year_any_enriched %in% working$year, firm_key])
    )))
  }

  params <- tryCatch(
    fixest::iplot(model, only.params = TRUE),
    error = function(e) NULL
  )
  if (is.null(params)) {
    return(list(status = data.table::data.table(
      model_id = spec_cfg$model_id,
      spec_id = spec_cfg$spec_id,
      outcome = outcome_cfg$name,
      outcome_group = outcome_cfg$group,
      treatment_name = treatment_name,
      subset_label = "all",
      estimator = "sunab",
      status = "skipped",
      note = "iplot_parameter_extraction_failed",
      nobs = fixest::nobs(model),
      treated_firms = uniqueN(working[!is.na(first_people_analytics_firm_year_any_enriched), firm_key])
    )))
  }

  estimate_dt <- data.table::data.table(
    event_time = as.integer(params$x),
    estimate = as.numeric(params$coef),
    ci_low = as.numeric(params$ci_low),
    ci_high = as.numeric(params$ci_high),
    std_error = (as.numeric(params$ci_high) - as.numeric(params$coef)) / 1.96,
    term = paste0("sunab_event_", sanitize_tag(as.character(params$x))),
    model_id = spec_cfg$model_id,
    spec_id = spec_cfg$spec_id,
    spec_label = spec_cfg$spec_label,
    estimator = "sunab",
    treatment_name = treatment_name,
    outcome = outcome_cfg$name,
    outcome_label = outcome_cfg$label,
    outcome_group = outcome_cfg$group,
    subset_label = "all",
    nobs = fixest::nobs(model)
  )
  plot_title <- sprintf("%s | %s", outcome_cfg$label, spec_cfg$spec_label)
  plot_path <- file.path(quicklook_dir, sprintf("%s.png", spec_cfg$model_id))
  quick_plot(estimate_dt[order(event_time)], plot_path, plot_title, outcome_cfg$label)
  status <- data.table::data.table(
    model_id = spec_cfg$model_id,
    spec_id = spec_cfg$spec_id,
    outcome = outcome_cfg$name,
    outcome_group = outcome_cfg$group,
    treatment_name = treatment_name,
    subset_label = "all",
    estimator = "sunab",
    status = "ok",
    note = "",
    nobs = fixest::nobs(model),
    treated_firms = uniqueN(working[!is.na(first_people_analytics_firm_year_any_enriched), firm_key])
  )
  list(coefficients = estimate_dt, status = status)
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
  treatments_cfg <- data.table::as.data.table(config$treatments)

  selected_outcomes <- split_csv(args$outcomes)
  selected_treatments <- split_csv(args$treatments)
  if (length(selected_outcomes) > 0) {
    outcomes_cfg <- outcomes_cfg[name %in% selected_outcomes]
  }
  if (length(selected_treatments) > 0) {
    treatments_cfg <- treatments_cfg[name %in% selected_treatments]
  }

  if (nrow(outcomes_cfg) == 0) {
    stop("No outcomes selected for estimation.")
  }
  if (nrow(treatments_cfg) == 0) {
    stop("No treatments selected for estimation.")
  }

  output_dir <- normalizePath(args$output_dir, mustWork = FALSE)
  results_dir <- file.path(output_dir, "results")
  quicklook_dir <- file.path(output_dir, "quicklook")
  notes_dir <- file.path(output_dir, "notes")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(quicklook_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(notes_dir, recursive = TRUE, showWarnings = FALSE)

  needed_cols <- unique(c(
    "firm_key",
    "year",
    "naics2",
    "has_both_data_by_year",
    "first_people_analytics_firm_year_any_enriched",
    outcomes_cfg$name,
    unlist(lapply(treatments_cfg$name, function(prefix) {
      c(
        paste0(prefix, "_analysis_row"),
        paste0(prefix, "_balanced_treated"),
        paste0(prefix, "_event_time_binned")
      )
    })),
    unlist(config$heterogeneity[, "column"])
  ))

  ds <- arrow::open_dataset(args$sample_path, format = "parquet")
  frame <- ds |>
    dplyr::select(dplyr::all_of(needed_cols)) |>
    dplyr::collect()
  dt <- data.table::as.data.table(frame)

  spec_rows <- list()
  if ("main" %in% treatments_cfg$name) {
    spec_rows <- append(spec_rows, list(
      list(spec_id = "spec1_main_twfe", spec_label = "Spec 1: firm FE and year FE", treatment_name = "main", both_only = FALSE, extra_fe = NULL, subset_col = NULL, subset_label = "all"),
      list(spec_id = "spec2_main_naics2_year", spec_label = "Spec 2: firm FE, year FE, and NAICS2-by-year FE", treatment_name = "main", both_only = FALSE, extra_fe = "naics2_year", subset_col = NULL, subset_label = "all"),
      list(spec_id = "spec3_main_both_sources", spec_label = "Spec 3: main treatment, both data sources by year", treatment_name = "main", both_only = TRUE, extra_fe = NULL, subset_col = NULL, subset_label = "all")
    ))
  }
  if ("position" %in% treatments_cfg$name) {
    spec_rows <- append(spec_rows, list(
      list(spec_id = "spec4_position_twfe", spec_label = "Spec 4a: position-based treatment", treatment_name = "position", both_only = FALSE, extra_fe = NULL, subset_col = NULL, subset_label = "all")
    ))
  }
  if ("posting" %in% treatments_cfg$name) {
    spec_rows <- append(spec_rows, list(
      list(spec_id = "spec4_posting_twfe", spec_label = "Spec 4b: posting-based treatment", treatment_name = "posting", both_only = FALSE, extra_fe = NULL, subset_col = NULL, subset_label = "all")
    ))
  }

  if (as_flag(args$run_heterogeneity) && "main" %in% treatments_cfg$name) {
    for (hetero in seq_len(nrow(config$heterogeneity))) {
      spec_rows <- append(spec_rows, list(
        list(
          spec_id = sprintf("heterogeneity_main_%s", config$heterogeneity[hetero, "name"]),
          spec_label = sprintf("Main treatment | %s firms", config$heterogeneity[hetero, "name"]),
          treatment_name = "main",
          both_only = FALSE,
          extra_fe = NULL,
          subset_col = config$heterogeneity[hetero, "column"],
          subset_label = config$heterogeneity[hetero, "name"]
        )
      ))
    }
  }

  coefficient_tables <- list()
  pretrend_tables <- list()
  status_tables <- list()

  for (spec in spec_rows) {
    spec$model_id <- spec$spec_id
    for (row_id in seq_len(nrow(outcomes_cfg))) {
      outcome_cfg <- outcomes_cfg[row_id]
      if (grepl("^heterogeneity_", spec$spec_id) && outcome_cfg$group != "primary") {
        next
      }
      model_id <- sprintf("%s__%s", spec$spec_id, sanitize_tag(outcome_cfg$name))
      spec$model_id <- model_id
      result <- estimate_twfe_model(dt, outcome_cfg, spec, quicklook_dir)
      if (!is.null(result$coefficients)) {
        coefficient_tables[[length(coefficient_tables) + 1L]] <- result$coefficients
      }
      if (!is.null(result$pretrend)) {
        pretrend_tables[[length(pretrend_tables) + 1L]] <- result$pretrend
      }
      status_tables[[length(status_tables) + 1L]] <- result$status
    }
  }

  advanced_tables <- list()
  if (as_flag(args$run_advanced) && "main" %in% treatments_cfg$name && exists("iplot", where = asNamespace("fixest"), mode = "function")) {
    primary_outcomes <- outcomes_cfg[group == "primary"]
    for (row_id in seq_len(nrow(primary_outcomes))) {
      outcome_cfg <- primary_outcomes[row_id]
      spec <- list(
        spec_id = "advanced_sunab_main",
        spec_label = "Advanced: Sun-Abraham",
        treatment_name = "main",
        model_id = sprintf("advanced_sunab_main__%s", sanitize_tag(outcome_cfg$name))
      )
      result <- estimate_sunab_model(dt, outcome_cfg, spec, quicklook_dir)
      if (!is.null(result$coefficients)) {
        advanced_tables[[length(advanced_tables) + 1L]] <- result$coefficients
      }
      status_tables[[length(status_tables) + 1L]] <- result$status
    }
  } else {
    note <- if (!as_flag(args$run_advanced)) {
      "Advanced estimator was not requested."
    } else {
      "Advanced estimator skipped because fixest::iplot parameter extraction is unavailable."
    }
    writeLines(note, con = file.path(notes_dir, "advanced_estimator_note.txt"))
  }

  coefficients_dt <- if (length(coefficient_tables) > 0) data.table::rbindlist(coefficient_tables, use.names = TRUE, fill = TRUE) else data.table::data.table()
  pretrend_dt <- if (length(pretrend_tables) > 0) data.table::rbindlist(pretrend_tables, use.names = TRUE, fill = TRUE) else data.table::data.table()
  status_dt <- if (length(status_tables) > 0) data.table::rbindlist(status_tables, use.names = TRUE, fill = TRUE) else data.table::data.table()
  advanced_dt <- if (length(advanced_tables) > 0) data.table::rbindlist(advanced_tables, use.names = TRUE, fill = TRUE) else data.table::data.table()

  data.table::fwrite(coefficients_dt, file.path(results_dir, "twfe_event_study_coefficients.csv"))
  data.table::fwrite(pretrend_dt, file.path(results_dir, "twfe_pretrend_tests.csv"))
  data.table::fwrite(status_dt, file.path(results_dir, "model_status.csv"))
  data.table::fwrite(advanced_dt, file.path(results_dir, "advanced_event_study_coefficients.csv"))
  data.table::fwrite(data.table::rbindlist(spec_rows, fill = TRUE), file.path(results_dir, "spec_manifest.csv"))
}

main()
