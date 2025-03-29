# R Script Running fastBCR for sequence clustering in OASCS
# Script code was adapted from the fastBCR github
# https://github.com/ZhangLabTJU/fastBCR
library("optparse")
library(fastBCR)
#
#
options <- OptionParser()
options <- add_option(options, c("-d", "--directory"),
                      type = "character",
                      default = "./R_scripts/data",
                      help = "Dataset directory path")
options <- add_option(options, c("-p", "--paired"),
                      action = "store_true",
                      default = FALSE,
                      help = "Set flag to run fastBCR-p")
options <- add_option(options, c("-c", "--cluster_thre"),
                      type = "integer",
                      default = 3,
                      help = "Cluster Threshold")
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
#
raw_sample_list <- data.load(folder_path = arguments$directory,
                             storage_format = "csv")
paired_sample_list <- paired.preprocess(raw_sample_list)
cluster_lists <- data.BCR.clusters(paired_sample_list,
                                   cluster_thre = arguments$cluster_thre,
                                   overlap_thre = arguments$overlap_thre,
                                   consensus_thre = arguments$consensus_thre,
                                   paired = arguments$paired)
for (x in seq_along(cluster_lists)) {
  file_name <- paste0("file_", x, ".csv")
  write.csv(cluster_lists[[x]], file = file_name, row.names = FALSE)
}
