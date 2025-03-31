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
  
  write.csv(label_file, tmp_label_file, row.names = FALSE)
  tmp_label_file
  
}

pgse <- function(x,
                 labels,
                 pre_kfold_info_file = NULL,
                 k = 6,
                 ext = 2,
                 target = 70,
                 features = 10000,
                 folds = 0,
                 ea_min = NULL,
                 ea_max = NULL,
                 num_rounds = 1500,
                 lr = 0.03,
                 dist = FALSE,
                 nodes = 1,
                 workers = 8) {
  
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
  
  pgse_models <- Map(\(m, s) {
    tmp_model <- tempfile(fileext = ".json")
    m$save_model(tmp_model)
    r_model <- xgboost::xgb.load(tmp_model)
    pgse_model <- list(model = r_model, segments = s)
    class(pgse_model) <- append(class(pgse_model), "pgse_model", after = 0)
    pgse_model
  }, pipe_out$models, pipe_out$segments)
  
  output <- list(results = pipe_out$results,
                 models = pgse_models)
  
  class(output) <- append(class(output), "pgse_output", after = 0)
  return(output)
}

pgse_api_train <- function(x,
                           labels,
                           pre_kfold_info_file = NULL,
                           save_file = "",
                           export_file = "./default.export",
                           k = 6,
                           ext = 2,
                           target = 70,
                           features = 10000,
                           folds = 0,
                           ea_min = NULL,
                           ea_max = NULL,
                           num_rounds = 1500,
                           lr = 0.03,
                           dist = FALSE,
                           nodes = 1,
                           workers = 8,
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
  workers <- as.integer(workers)
  
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
