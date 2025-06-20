# R Script Running fastBCR for sequence clustering in OASCS
# Script code was adapted from the fastBCR github
# https://github.com/ZhangLabTJU/fastBCR
library("optparse")
library(fastBCR)
#
# Argument Parser
options <- OptionParser()
options <- add_option(options, c("-d", "--directory"),
                      type = "character",
                      default = "./R_scripts/data",
                      help = "Dataset directory path")
options <- add_option(options, c("-p", "--paired"),
                      action = "store_true",
                      default = FALSE,
                      help = "Set flag to run fastBCR-p")
options <- add_option(options, c("-c", "--min_depth_thre"),
                      type = "integer",
                      default = 3,
                      help = "Minimum Depth Threshold")
options <- add_option(options, c("-x", "--max_depth_thre"),
                      type = "integer",
                      default = 1000,
                      help = "Maximum Depth Threshold")
options <- add_option(options, c("-o", "--overlap_thre"),
                      type = "numeric",
                      default = 0.1,
                      help = "Overlap Threshold")
options <- add_option(options, c("-n", "--consensus_thre"),
                      type = "numeric",
                      default = 0.8,
                      help = "Consensus Threshold")
arguments <- parse_args(options)
#
# Run FastBCR
raw_sample_list <- data.load(folder_path = arguments$directory,
                             storage_format = "csv")
if (arguments$paired == TRUE) {
  processed_sample_list <- paired.preprocess(raw_sample_list)
} else {
  processed_sample_list <- data.preprocess(raw_data_list = raw_sample_list,
                                           productive_only = TRUE)
}

cluster_lists <- data.BCR.clusters(processed_sample_list,
                                   min_depth_thre = arguments$min_depth_thre,
                                   max_depth_thre = arguments$max_depth_thre,
                                   overlap_thre = arguments$overlap_thre,
                                   consensus_thre = arguments$consensus_thre,
                                   paired = arguments$paired)
#
# Save Files
for (file_index in seq_along(cluster_lists)) {
  for (clonotype_index in seq_along(cluster_lists[[file_index]])) {
    file_name <- paste0(arguments$directory,
                        "/file_",
                        names(cluster_lists)[file_index],
                        "_clonotype_",
                        clonotype_index,
                        ".csv")
    write.csv(cluster_lists[[file_index]][[clonotype_index]],
              file = file_name,
              row.names = FALSE)
  }
}
