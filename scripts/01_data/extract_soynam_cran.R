# Write the SoyNAM line and check phenotypes shipped with the CRAN package SoyNAM 1.6.2 (Diers et al. 2018)
# as tab-separated tables: data/interim/soynam/data_{line,check}_{qa,raw}.tsv (read by src/hapgam/scenarios.py).
# The QA tables are in data/soybase.RData, the raw tables (data.line, data.check) in data/soynam.RData.
# usage: Rscript scripts/01_data/extract_soynam_cran.R [SoyNAM package directory] [out_dir]
args <- commandArgs(trailingOnly = TRUE)
pkg <- if (length(args) >= 1) args[1] else "data/raw/soynam_cran"
out <- if (length(args) >= 2) args[2] else "data/interim/soynam"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
env <- new.env()
for (rd in c("soybase.RData", "soynam.RData")) load(file.path(pkg, "data", rd), envir = env)
tabs <- c(data_line_qa = "data.line.qa", data_line_raw = "data.line", data_check_qa = "data.check.qa",
          data_check_raw = "data.check")
for (f in names(tabs)) {
  write.table(get(tabs[[f]], envir = env), file.path(out, paste0(f, ".tsv")), sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
}
cat("wrote", length(tabs), "tables to", out, "\n")
