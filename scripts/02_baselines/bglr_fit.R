# BGLR Bayesian marker regressions (BayesB / BayesC) for one fold.
# usage: Rscript bglr_fit.R <fold_dir> <model> <nIter> <burnIn>
# <fold_dir> contains either Xtr.int8 / Xte.int8 / dims.txt (any scenario; written by
# bglr_prepare.py) or, for older GRIN folds, train_rows.txt / test_rows.txt indexing the GRIN
# dosage matrix.  Predictions are written to <fold_dir>/<model>.pred.txt
suppressMessages(library(BGLR))
args <- commandArgs(trailingOnly = TRUE)
fold_dir <- args[1]; model <- args[2]
n_iter <- as.integer(args[3]); burn_in <- as.integer(args[4])

read_int8 <- function(path, n, p) {
  con <- file(path, "rb"); on.exit(close(con))
  matrix(readBin(con, "integer", n = n * p, size = 1, signed = TRUE), nrow = n, ncol = p, byrow = TRUE)
}

if (file.exists(file.path(fold_dir, "dims.txt"))) {
  dims <- scan(file.path(fold_dir, "dims.txt"), quiet = TRUE)
  X <- read_int8(file.path(fold_dir, "Xtr.int8"), dims[1], dims[3])
  Xt <- read_int8(file.path(fold_dir, "Xte.int8"), dims[2], dims[3])
} else {
  root <- "/home/user/zhangzhishuai/myhermes/2080/HapGAM/data/processed/grin"
  meta <- readLines(file.path(root, "dosage.int8.json"), warn = FALSE)
  p <- as.integer(sub('.*"p": ([0-9]+).*', "\\1", meta))
  read_rows <- function(rows) {
    con <- file(file.path(root, "dosage.int8"), "rb"); on.exit(close(con))
    out <- matrix(0, length(rows), p)
    for (i in seq_along(rows)) {
      seek(con, where = as.numeric(rows[i]) * p, origin = "start")
      out[i, ] <- readBin(con, "integer", n = p, size = 1, signed = TRUE)
    }
    out
  }
  X <- read_rows(scan(file.path(fold_dir, "train_rows.txt"), quiet = TRUE))
  Xt <- read_rows(scan(file.path(fold_dir, "test_rows.txt"), quiet = TRUE))
}
y <- scan(file.path(fold_dir, "y_train.txt"), quiet = TRUE)
storage.mode(X) <- "double"
mu_x <- colMeans(X)
X <- sweep(X, 2, mu_x)
set.seed(20261020)
t0 <- Sys.time()
fit <- BGLR(y = y, ETA = list(list(X = X, model = model, saveEffects = FALSE)),
            nIter = n_iter, burnIn = burn_in, thin = 10, verbose = FALSE,
            saveAt = file.path(fold_dir, paste0(model, "_")))
rm(X); invisible(gc())
storage.mode(Xt) <- "double"
Xt <- sweep(Xt, 2, mu_x)
pred <- as.numeric(fit$mu + Xt %*% fit$ETA[[1]]$b)
write(pred, file.path(fold_dir, paste0(model, ".pred.txt")), ncolumns = 1)
cat(sprintf("done %s n=%d p=%d minutes=%.1f varE=%.4f\n", model, length(y), ncol(Xt),
            as.numeric(difftime(Sys.time(), t0, units = "mins")), fit$varE))
