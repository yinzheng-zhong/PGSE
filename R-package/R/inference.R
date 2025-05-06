#' Predict method for PGSE model
#'
#' @param object pgse_model object
#' @param newdata character list of file paths to sequences
#' @param workers number of workers to use for inference
#' @return array of predictions
#' @export
predict.pgse_model <- function(object,
                               newdata,
                               workers = 1L) {
  workers <- check_workers(workers)
  pgse_module <- reticulate::import("pgse")

  tmp_model_file <- tempfile(fileext = ".json")
  xgboost::xgb.save(object$model, tmp_model_file)

  tmp_segments_file <- tempfile(fileext = ".txt")
  writeLines(object$segments, tmp_segments_file)

  pipeline <- pgse_module$InferencePipeline(model_path = tmp_model_file,
                                            segment_path = tmp_segments_file,
                                            workers = workers)
  pipeline$run(newdata)
}

#' Predict method for PGSE output
#'
#' @param object pgse_output object
#' @param newdata character list of file paths to sequences
#' @param workers number of workers to use for inference
#'
#' @return list of predictions (arrays)
#' @export
predict.pgse_output_simple <- function(object,
                                       newdata,
                                       workers = 1L) {
  workers <- check_workers(workers)
  stats::predict(object$model, newdata, workers = workers)
}

#' Predict method for cross-validation PGSE output
#'
#' @param object pgse_output_cv object
#' @param newdata character list of file paths to sequences
#' @param workers number of workers to use for inference
#'
#' @export
predict.pgse_output_cv <- function(object,
                                   newdata,
                                   workers = 1L) {
  workers <- check_workers(workers)
  lapply(object$models, \(m) {
    stats::predict(m, newdata, workers = workers)
  })
}

#' Low-level API wrapper for PGSE inference pipeline
#'
#' @param model_path path to the model file (usually .json)
#' @param segment_path path to the segments file (usually .txt)
#' @param files character vector of file paths to sequences
#' @param workers number of workers to use for inference
#' @param ... additional arguments to pass to the pipeline
#'
#' @return array of predictions
#' @export
#'
#' @description
#' This is low-level wrapper around the Python PGSE inference pipeline.
#' In general, it is recommended to use the `predict` method on
#' `pgse()` instead.
pgse_api_inference <- function(model_path,
                               segment_path,
                               files,
                               workers = 1L,
                               ...) {
  workers <- check_workers(workers)
  pgse_module <- reticulate::import("pgse")

  # shutdown ray if it is running
  reticulate::py_run_string("import ray; ray.shutdown()")

  pipeline <- pgse_module$InferencePipeline(model_path = model_path,
                                            segment_path = segment_path,
                                            ...)
  pipeline$run(files)
}
