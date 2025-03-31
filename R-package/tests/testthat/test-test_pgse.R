test_that("test validate inputs", {
  expect_error(validate_paths_input("fixtures/genomes/"))
  expect_error(validate_paths_input("fixtures/genomes"))
  expect_no_error(validate_paths_input("fixtures/genomes/001.fna"))
  expect_no_error(validate_paths_input(list.files("fixtures/genomes/",
                                                      full.names = TRUE)))
  
  expect_error(validate_paths_input("fixtures/genomes/i_do_not_exist.fna"))
  expect_error(validate_paths_input(c("fixtures/genomes/",
                                      "fixtures/genomes/001.fna")))
  expect_error(validate_paths_input(c("fixtures/genomes",
                                      "fixtures/genomes/001.fna")))
})

test_that("test label processing", {
  # expect_no_error(process_labels("fixtures/labels.csv","fixtures/genomes/"))
  # expect_true(file.exists(process_labels("fixtures/labels.csv",
  #                                        "fixtures/genomes/")))
  
  converted_labels_file <- process_labels(
    labels = rep(1, length(list.files("fixtures/genomes/"))),
    paths = list.files("fixtures/genomes/", full.names = TRUE))
  
  expect_true(file.exists(converted_labels_file))
  converted_labels <- read.csv(converted_labels_file)
  
  expect_equal(converted_labels$labels,
               rep(1, length(list.files("fixtures/genomes/"))))
  
  expect_error(process_labels(
    labels = rep(1, length(list.files("fixtures/genomes/")) - 1),
    paths = list.files("fixtures/genomes/", full.names = TRUE)))
})

test_that("basic pgse run", {
  genomes_paths <- list.files("fixtures/genomes/", full.names = TRUE)
  labels <- read.csv(file.path("fixtures", "labels.csv"))$labels
  p <- pgse(x = genomes_paths, labels = labels)
  expect_s3_class(p, "pgse_output")
})

test_that("pgse simple API", {
  folds <- 2
  genomes_path <- file.path("fixtures", "genomes")
  labels_path <- file.path("fixtures", "labels.csv")
  save_file <- tempfile()
  export_file <- tempfile(pattern = "pgse_export")
  
  pgse_api_train(x = genomes_path,
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
  
  expect_length(segment_file_names_full, folds)
  expect_length(model_file_names_full, folds)
})
