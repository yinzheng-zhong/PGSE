genomes_dir <- file.path("fixtures", "genomes")
genomes_paths <- list.files(genomes_dir, full.names = TRUE,
                            pattern = ".fna")
labels_path <- file.path("fixtures", "labels.csv")
labels <- read.csv(labels_path)$labels

p <- pgse(x = genomes_paths,
          labels = labels,
          folds = 2)


test_that("single model prediction", {
  predictions <- predict(p$models[[1]], newdata = genomes_paths)
  expect_vector(predictions, ptype = array(numeric()))
  expect_equal(length(predictions), length(genomes_paths))
})

test_that("multi-model prediction", {
  predictions <- stats::predict(p, newdata = genomes_paths)
  expect_type(predictions, "list")
  expect_equal(length(predictions), length(p$models))
  expect_equal(length(predictions[[1]]), length(genomes_paths))
  expect_vector(predictions[[1]], ptype = array(numeric()))
})

test_that("low-level API inference", {
  folds <- 2
  save_file <- tempfile()
  export_file <- tempfile(pattern = "pgse_export")

  pgse_api_train(x = genomes_dir,
                 labels = labels_path,
                 save_file = save_file,
                 export_file = export_file,
                 folds = folds)

  fold_indices <- seq(from=0, to=(folds-1), by=1)
  export_file_names <- tools::file_path_sans_ext(basename(export_file))

  segment_file_names_full <- file.path(dirname(export_file),
                                       paste0(export_file_names, "_fold_", fold_indices, ".txt"))

  model_file_names_full <- file.path(dirname(export_file),
                                     paste0(export_file_names, "_fold_", fold_indices, ".json"))

  predictions <- mapply(pgse_api_inference,
    model_path = model_file_names_full,
    segment_path = segment_file_names_full,
    MoreArgs = list(files = genomes_paths),
    SIMPLIFY = FALSE)

  expect_length(predictions, folds)
  for (i in seq_along(predictions)) {
    expect_vector(predictions[[i]], ptype = array(numeric()))
  }

})