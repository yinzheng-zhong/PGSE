validate_paths_input <- function(x) {
  if (!is.character(x)) {
    stop("x must be a character vector of file paths.")
  }
  if (any(dir.exists(x))) {
    stop("The following paths are directories (not files): ",
         paste(x[dir.exists(x)], collapse = ", "))
  }
  if (!all(file.exists(x))) {
    stop("The following files do not exist: ",
         paste(x[!file.exists(x)], collapse = ", "))
  }
}

process_labels <- function(labels, paths) {
  if (length(labels) != length(paths)) {
    stop("The number of labels does not match the number of paths.")
  }

  tmp_label_file <- tempfile(fileext = ".csv")
  label_file <- data.frame(files = paths,
                           labels = labels)

  utils::write.csv(label_file, tmp_label_file, row.names = FALSE)
  tmp_label_file

}

#' Run PGSE
#'
#' @param x character vector of file paths
#' @param labels vector of labels
#' @param pre_kfold_info_file path to pre-kfold info file (not supported yet)
#' @param k initial $k$-mer length
#' @param ext $p$ extension parameter
#' @param target target number of features
#' @param features number of features to select
#' @param folds number of folds for cross-validation (0 for train test split)
#' @param ea_min minimum essential agreement
#' @param ea_max maximum essential agreement
#' @param num_rounds number of rounds for XGBoost training
#' @param lr learning rate for XGBoost
#' @param dist logical, whether to use distributed training
#' @param nodes number of nodes for distributed training
#' @param workers number of workers for distributed training (or XGBoost)
#'
#' @return pgse_output_simple or pgse_output_cv object
#' @export
pgse <- function(x,
                 labels,
                 pre_kfold_info_file = NULL,
                 k = 6L,
                 ext = 2L,
                 target = 70L,
                 features = 10000L,
                 folds = 0L,
                 ea_min = NULL,
                 ea_max = NULL,
                 num_rounds = 1500L,
                 lr = 0.03,
                 dist = FALSE,
                 nodes = 1L,
                 workers = 1L) {

  if (!is.null(pre_kfold_info_file)) {
    stop("pre_kfold_info_file is not supported yet.")
  }

  validate_paths_input(x)
  labels <- process_labels(labels, x)

  pgse_module <- reticulate::import("pgse")

  # shutdown ray if it is running
  reticulate::py_run_string("import ray; ray.shutdown()")

  k <- as.integer(k)
  ext <- as.integer(ext)
  target <- as.integer(target)
  features <- as.integer(features)
  folds <- as.integer(folds)
  num_rounds <- as.integer(num_rounds)
  lr <- as.numeric(lr)
  nodes <- as.integer(nodes)
  workers <- as.integer(workers)

  pipeline <- pgse_module$TrainingPipeline(data_dir = x,
                                           label_file = labels,
                                           pre_kfold_info_file = pre_kfold_info_file,
                                           save_file = "",
                                           export_file = "",
                                           k = k,
                                           ext = ext,
                                           target = target,
                                           features = features,
                                           folds = folds,
                                           ea_min = ea_min,
                                           ea_max = ea_max,
                                           num_rounds = num_rounds,
                                           lr = lr,
                                           dist = dist,
                                           nodes = nodes,
                                           workers = workers)
  pipe_out <- pipeline$train()

  pgse_models <- lapply(pipe_out$folds, \(fold) {
    tmp_model <- tempfile(fileext = ".json")
    fold$model$booster$save_model(tmp_model)
    r_model <- xgboost::xgb.load(tmp_model)
    pgse_model <- list(model = r_model,
                       segments = fold$segments$segments,
                       importance = fold$segments$importances,
                       score = fold$score)
    class(pgse_model) <- append(class(pgse_model), "pgse_model", after = 0)
    pgse_model
  })

  results <- lapply(pipe_out$folds, \(fold) fold$predictions)

  if (length(results) == 1) {
    output <- list(result = results[[1]],
                   model = pgse_models[[1]])
    class(output) <- append(class(output), "pgse_output_simple", after = 0)
    return(output)
  }

  output <- list(results = results,
                 models = pgse_models)
  class(output) <- append(class(output), "pgse_output_cv", after = 0)
  return(output)
}

#' Low-level API for PGSE training pipeline
#'
#' @param x character vector of file paths
#' @param labels vector of labels
#' @param pre_kfold_info_file path to pre-kfold info file
#' @param save_file path to save file
#' @param export_file path to export file
#' @param k initial $k$-mer length
#' @param ext $p$ extension parameter
#' @param target target number of features
#' @param features number of features to select
#' @param folds number of folds for cross-validation (0 for train test split)
#' @param ea_min minimum essential agreement
#' @param ea_max maximum essential agreement
#' @param num_rounds number of rounds for XGBoost training
#' @param lr learning rate for XGBoost
#' @param dist logical, whether to use distributed training
#' @param nodes number of nodes for distributed training
#' @param workers number of workers for distributed training (or XGBoost)
#' @param ... additional arguments to pass to the pipeline
#'
#' @return NULL (results are saved to disk)
#' @export
pgse_api_train <- function(x,
                           labels,
                           pre_kfold_info_file = NULL,
                           save_file = "",
                           export_file = "./default.export",
                           k = 6L,
                           ext = 2L,
                           target = 70L,
                           features = 10000L,
                           folds = 0L,
                           ea_min = NULL,
                           ea_max = NULL,
                           num_rounds = 1500L,
                           lr = 0.03,
                           dist = FALSE,
                           nodes = 1L,
                           workers = 1L,
                           ...) {

  pgse_module <- reticulate::import("pgse")

  # shutdown ray if it is running
  reticulate::py_run_string("import ray; ray.shutdown()")

  k <- as.integer(k)
  ext <- as.integer(ext)
  target <- as.integer(target)
  features <- as.integer(features)
  folds <- as.integer(folds)
  num_rounds <- as.integer(num_rounds)
  lr <- as.numeric(lr)
  nodes <- as.integer(nodes)
  workers <- check_workers(workers)

  pipeline <- pgse_module$TrainingPipeline(data_dir = x,
                                           label_file = labels,
                                           pre_kfold_info_file = pre_kfold_info_file,
                                           save_file = save_file,
                                           export_file = export_file,
                                           k = k,
                                           ext = ext,
                                           target = target,
                                           features = features,
                                           folds = folds,
                                           ea_min = ea_min,
                                           ea_max = ea_max,
                                           num_rounds = num_rounds,
                                           lr = lr,
                                           dist = dist,
                                           nodes = nodes,
                                           workers = workers,
                                           ...)

  pipeline$run()
}
