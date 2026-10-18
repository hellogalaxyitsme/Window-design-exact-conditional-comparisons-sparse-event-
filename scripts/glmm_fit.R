# Poisson GLMM comparator: fixed quadratic trend in window position, response indicator,
# and trial-level diagonal random intercept, slope and curvature (glmmTMB).
# Usage: Rscript glmm_fit.R <input.csv.gz> <output.csv>
# packages: glmmTMB, data.table, R.utils (set R_LIB to a private library folder if needed)
if (nzchar(Sys.getenv('R_LIB'))) .libPaths(c(Sys.getenv('R_LIB'), .libPaths())) else .libPaths(c('E:/R_lib', .libPaths()))
suppressPackageStartupMessages({library(glmmTMB); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
d <- fread(args[1])
d[, u2 := u^2]
res <- list()
for (s in split(d, by = c('cell', 'unit'))) {
  p <- NA_real_; est <- NA_real_; status <- 'ok'
  if (sum(s$y[s$resp == 1]) == 0 || sum(s$y[s$resp == 0]) == 0) {
    p <- 1; status <- 'no_events'
  } else {
    fit <- tryCatch(suppressWarnings(glmmTMB(y ~ u + u2 + resp + diag(1 + u + u2 | trial),
                                            family = poisson, data = s)),
                    error = function(e) NULL)
    if (is.null(fit)) { p <- 1; status <- 'fit_error' } else {
      co <- tryCatch(summary(fit)$coefficients$cond, error = function(e) NULL)
      if (is.null(co) || !('resp' %in% rownames(co)) || !is.finite(co['resp', 4])) {
        p <- 1; status <- 'nonconvergence'
      } else {
        p <- co['resp', 4]; est <- co['resp', 1]
        if (!isTRUE(fit$sdr$pdHess)) status <- 'non_pd_hessian'
      }
    }
  }
  res[[length(res) + 1]] <- data.table(cell = s$cell[1], unit = s$unit[1], p = p, estimate = est, status = status)
}
fwrite(rbindlist(res), args[2])
