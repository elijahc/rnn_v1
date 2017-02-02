library(ggplot2)
library(readr)
combined <- read_csv("~/dev/rnn_v1/output/n30_step_variation/run_data_10-n30-comb-FEV.csv")
ns4=data_10_n30_ns4_FEV
ns8=X10_n30_ns8_FEV
ns16=data_10_n30_ns16_FEV
ns32=data_10_n30_ns32_FEV
p <- ggplot(combined, aes(Step,Value))+xlim(0,55000)+ylab('FEV')+facet_grid(.~num_steps)
p + stat_smooth(method=loess) + geom_point(alpha=1/25)