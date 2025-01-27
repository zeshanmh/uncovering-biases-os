# Create a flag to toggle between datasets
use_lr_data <- TRUE  # Set to TRUE to use the new data

## BIASED (logistic regression)
# defaultdict(list,
#             {'SE_Y0': [0.03397128865680828,
#               0.01868766028886165,
#               0.0492549170247549],
#              'SE_Y1': [0.039067805894640525,
#               0.02288427626772275,
#               0.0552513355215583],
#              'SE_A': [-0.10025319850777348,
#               -0.1445638165558293,
#               -0.05594258045971765],
#              'SE_S': [0.018520911355769033,
#               -0.037518258899853434,
#               0.0745600816113915]})
## UNBIASED (logistic regression)
# defaultdict(list,
#             {'SE_Y0': [0.05233078304041271,
#               0.012258515088070389,
#               0.09240305099275503],
#              'SE_Y1': [0.10108212953120048,
#               0.052673158987189224,
#               0.14949110007521174],
#              'SE_A': [-0.013340700980222506,
#               -0.03312052658621503,
#               0.006439124625770018],
#              'SE_S': [-0.05232157262760371,
#               -0.07575297624064037,
#               -0.028890169014567056]})


#BIASED (random forest)
# defaultdict(list,
#             {'SE_Y0': [0.02631747211173179,
#               0.014302630241875025,
#               0.03833231398158855],
#              'SE_Y1': [0.0338352020699568,
#               0.018925002701517687,
#               0.04874540143839591],
#              'SE_A': [-0.046607697023839614,
#               -0.06725212552717151,
#               -0.025963268520507718],
#              'SE_S': [0.2266931752223158,
#               0.08999307763062633,
#               0.36339327281400524]})

#UNBIASED (random forest)
# defaultdict(list,
#             {'SE_Y0': [0.027365732502528335,
#               0.014957470116098465,
#               0.039773994888958206],
#              'SE_Y1': [0.12333890353858519,
#               0.04620130949930053,
#               0.20047649757786984],
#              'SE_A': [-0.020340027887138595,
#               -0.030514933854793706,
#               -0.010165121919483485],
#              'SE_S': [-0.04512527821289835,
#               -0.06500784326160651,
#               -0.025242713164190186]})

# Create both datasets
data_rf <- data.frame(
  signal = rep(c("ρ(b,Y0)", "ρ(b,Y1)", "ρ(b,A)", "ρ(b,S)"), each = 2),
  setup = rep(c("Biased", "Unbiased"), 4),
  mean = c(
    # SE_Y0
    0.02631747211173179, 0.027365732502528335,
    # SE_Y1
    0.0338352020699568, 0.12333890353858519,
    # SE_A
    -0.046607697023839614, -0.020340027887138595,
    # SE_S
    0.2266931752223158, -0.04512527821289835
  ),
  lower = c(
    # SE_Y0
    0.014302630241875025, 0.014957470116098465,
    # SE_Y1
    0.018925002701517687, 0.04620130949930053,
    # SE_A
    -0.06725212552717151, -0.030514933854793706,
    # SE_S
    0.08999307763062633, -0.06500784326160651
  ),
  upper = c(
    # SE_Y0
    0.03833231398158855, 0.039773994888958206,
    # SE_Y1
    0.04874540143839591, 0.20047649757786984,
    # SE_A
    -0.025963268520507718, -0.010165121919483485,
    # SE_S
    0.36339327281400524, -0.025242713164190186
  )
)

data_lr <- data.frame(
  signal = rep(c("ρ(b,Y0)", "ρ(b,Y1)", "ρ(b,A)", "ρ(b,S)"), each = 2),
  setup = rep(c("Biased", "Unbiased"), 4),
  mean = c(
    # SE_Y0
    0.03397128865680828, 0.05233078304041271,
    # SE_Y1
    0.039067805894640525, 0.10108212953120048,
    # SE_A
    -0.10025319850777348, -0.013340700980222506,
    # SE_S
    0.018520911355769033, -0.05232157262760371
  ),
  lower = c(
    # SE_Y0
    0.01868766028886165, 0.012258515088070389,
    # SE_Y1
    0.02288427626772275, 0.052673158987189224,
    # SE_A
    -0.1445638165558293, -0.03312052658621503,
    # SE_S
    -0.037518258899853434, -0.07575297624064037
  ),
  upper = c(
    # SE_Y0
    0.0492549170247549, 0.09240305099275503,
    # SE_Y1
    0.0552513355215583, 0.14949110007521174,
    # SE_A
    -0.05594258045971765, 0.006439124625770018,
    # SE_S
    0.0745600816113915, -0.028890169014567056
  )
)

# Select the dataset based on the flag
data <- if(use_lr_data) data_lr else data_rf

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(grid)
library(ggthemes)
library(scales)

# Define the custom theme and scale functions
theme_Publication <- function(base_size=14, base_family="sans") {
  (theme_foundation(base_size=base_size, base_family=base_family)
   + theme(plot.title = element_text(face = "bold",
                                     size = rel(1.2), hjust = 0.5),
           text = element_text(),
           panel.background = element_rect(colour = NA),
           plot.background = element_rect(colour = NA),
           panel.border = element_rect(colour = NA),
           axis.title = element_text(face = "bold",size = rel(1)),
           axis.title.y = element_text(angle=90,vjust =2),
           axis.title.x = element_text(vjust = -0.2),
           axis.text = element_text(), 
           axis.line = element_line(colour="black"),
           axis.ticks = element_line(),
           panel.grid.major = element_line(colour="#f0f0f0"),
           panel.grid.minor = element_blank(),
           legend.key = element_rect(colour = NA),
           legend.position = "bottom",
           legend.direction = "horizontal",
           legend.key.size= unit(0.2, "cm"),
           legend.spacing = unit(0, "cm"),
           legend.title = element_text(face="italic"),
           plot.margin=unit(c(10,5,5,5),"mm"),
           strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
           strip.text = element_text(face="bold")
      ))
}

scale_fill_Publication <- function(...){
  discrete_scale("fill","Publication",manual_pal(values = c("#386cb0","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}

scale_colour_Publication <- function(...){
  discrete_scale("colour","Publication",manual_pal(values = c("#386cb0","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}

# Create the plot
title <- if(use_lr_data) { 
    "Alignment w/ Predictive Performance using LogReg (stroke)"
} else {
    "Alignment w/ Predictive Performance using RF (stroke)"
}
p <- ggplot(data, aes(x = signal, y = mean, fill = setup)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
                position = position_dodge(width = 0.8),
                width = 0.25) +
  labs(x = "Signal Type",
       y = "Covariance",
       fill = "Setup") +
  scale_y_continuous(limits = c(-0.15, 0.38)) +
  theme_Publication() +
  scale_fill_Publication() +
  ggtitle(title) +
  theme(
    plot.title = element_text(size=25, face="bold"),
    axis.text = element_text(size = 22),
    axis.title = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 22),
    legend.text = element_text(size = 21),
    legend.key.size = unit(1,"cm")
  )

# Print the plot
print(p)

# Save the plot with a name that reflects which dataset was used
filename <- if(use_lr_data) "covariance_comparison_lr_v2_STROKE.png" else "covariance_comparison_rf_v2_STROKE.png"
ggsave(filename, p, width = 12, height = 8, dpi = 300)