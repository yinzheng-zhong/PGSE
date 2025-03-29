test_that("basic pgse run", {
  genomes_path <- file.path("fixtures", "genomes")
  labels_path <- file.path("fixtures", "labels.csv")
  p <- pgse(x = genomes_path, labels = labels_path)
  expect_s3_class(p, "pgse_output")
})
