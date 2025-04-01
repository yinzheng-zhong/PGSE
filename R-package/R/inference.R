predict.pgse_model <- function(object,
                         newdata) {
  pgse_module <- reticulate::import("pgse")
  
  tmp_model_file <- tempfile(fileext = ".json")
  xgboost::xgb.save(object$model, tmp_model_file)
  
  tmp_segments_file <- tempfile(fileext = ".txt")
  writeLines(object$segments, tmp_segments_file)
  
  pipeline <- pgse_module$InferencePipeline(model_path = tmp_model_file,
                                            segment_path = tmp_segments_file)
  pipeline$run(newdata)
}

predict.pgse_output <- function(object,
                          newdata) {
  if (length(object$models) == 1) {
    return(stats::predict(object$models[[1]], newdata))
  }
  message(
  "Appear to be using a multi-model PGSE output,
  probably from cross-validation. Predicting with each model.."
  )
  predictions <- lapply(object$models, \(m) {
    stats::predict(m, newdata)
  })
  return(predictions)
}

pgse_api_inference <- function(model_path,
                               segment_path,
                               files,
                               ...) {
  pgse_module <- reticulate::import("pgse")
  
  # shutdown ray if it is running
  reticulate::py_run_string("import ray; ray.shutdown()")
  
  pipeline <- pgse_module$InferencePipeline(model_path = model_path,
                                            segment_path = segment_path)
  pipeline$run(files)
}