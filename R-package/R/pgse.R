pgse <- function(x,
                 labels,
                 pre_kfold_info_file = NULL,
                 save_file = NULL,
                 export_file = NULL,
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
  pgse_module <- reticulate::import("pgse")
  if (is.null(save_file)) {
    save_file <- tempfile("save_file_", fileext = ".txt")
  }
  if (is.null(export_file)) {
    export_file <- tempfile("export_file_")
  }
  k <- as.integer(k)
  ext <- as.integer(ext)
  target <- as.integer(target)
  features <- as.integer(features)
  folds <- as.integer(folds)
  num_rounds <- as.integer(num_rounds)
  lr <- as.numeric(lr)
  nodes <- as.integer(nodes)
  workers <- as.integer(workers)

  reticulate::py_run_string("import ray")
  reticulate::py_run_string("ray.shutdown()")

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
                                           workers = workers)
  pipeline$run()
  output <- list()

  # fold_indices <- ifelse(
  #   folds == 0,
  #   0,
  #   seq(from=0, to=(folds-1), by=1))

  if (folds == 0) {
    fold_indices <- 0
  } else {
    fold_indices <- seq(from=0, to=(folds-1), by=1)
  }

  # print(folds)
  # print(fold_indices)
  export_file_names <- tools::file_path_sans_ext(basename(export_file))
  segment_file_names_full <- file.path(dirname(export_file),
                                      paste0(export_file_names, "_fold_", fold_indices, ".txt"))
  # print(segment_file_names_full)
  model_file_names_full <- file.path(dirname(export_file),
                                       paste0(export_file_names, "_fold_", fold_indices, ".json"))


  if (folds == 0) {
    output[["segments"]] <- readLines(segment_file_names_full)
    output[["models"]] <- xgboost::xgb.load(model_file_names_full)
  } else {
    output[["segments"]] <- lapply(segment_file_names_full, readLines)
    output[["models"]] <- lapply(model_file_names_full, xgboost::xgb.load)
  }

  class(output) <- append(class(output), "pgse", after = 0)
  return(output)
}
