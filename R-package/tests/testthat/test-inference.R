genomes_dir <- file.path("fixtures", "genomes")
labels_path <- file.path("fixtures", "labels.csv")
p <- pgse(x = genomes_dir, labels = labels_path,
          folds = 2)

genomes_paths <- list.files(genomes_dir, full.names = TRUE,
                            pattern = ".fna")

test_that("single model prediction", {
  
  predictions <- predict(p$models[[1]], newdata = genomes_paths)
  expect_vector(predictions, ptype = array(numeric()))
  expect_equal(length(predictions), length(genomes_paths))
})

test_that("multi-model prediction", {
  
  predictions <- predict(p, newdata = genomes_paths)
  expect_type(predictions, "list")
  expect_equal(length(predictions), length(p$models))
  expect_equal(length(predictions[[1]]), length(genomes_paths))
  expect_vector(predictions[[1]], ptype = array(numeric()))
})
