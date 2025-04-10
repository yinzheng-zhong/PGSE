#' Plot a simple PGSE output
#'
#' @param x pgse_output_simple object
#' @param y not used
#' @param ... additional arguments
#'
#' @export
plot.pgse_output_simple <- function(x, y, ...) {
  plot(x$result$Actual, x$result$Prediction,
       xlab = "Actual",
       ylab = "Predicted",
       main = "PGSE Model Predictions")
}

#' Plot a cross-validation PGSE output
#'
#' @param x pgse_output_cv object
#' @param y not used
#' @param ... additional arguments
#'
#' @export
plot.pgse_output_cv <- function(x, y, ...) {
  results <- do.call(rbind, x$results)
  plot(results$Actual, results$Prediction,
       xlab = "Actual",
       ylab = "Predicted",
       main = "PGSE Model Predictions")
}
