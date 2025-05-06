.onLoad <- function(libname, pkgname) {
  reticulate::py_require("git+https://github.com/yinzheng-zhong/PGSE@main")
}

check_workers <- function(x) {
  tryCatch(x <- as.integer(x),
           error = \(e) {
             stop("workers must be an integer")
           })
  stopifnot("workers must be length 1" = length(x) == 1L,
            "workers must be positive" = x > 0L)
  x
}
