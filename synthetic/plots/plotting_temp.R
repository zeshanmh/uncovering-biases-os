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

# Create a data frame with all results
data <- data.frame(
  bias_setup = rep(c("Unbiased", "Selection Type 2", "Selection Type 3"), each = 9),
  X1 = rep(rep(c("All", "X[1]==1", "X[1]==-1"), each = 3), 3),
  signal = rep(c("Outcome", "Treatment", "Selection"), 9),
  mean = c(
    # Unbiased
    1.6594, 0.5227, 0.1711,  # All
    1.6251, 0.4832, 0.0000,  # X1 = 1
    2.8528, 0.8884, 0.0000,  # X1 = -1
    # Selection Type 1/2
    1.1166, 0.6350, -0.0302, # All
    1.7641, 0.6398, 0.0000,  # X1 = 1
    1.6929, 0.6285, 0.0000,  # X1 = -1
    # Selection Type 3
    2.5998, 0.6195, 0.2197,  # All
    2.9156, 0.5675, 0.0000,  # X1 = 1
    2.8528, 0.8884, 0.0000   # X1 = -1
  ),
  ci = c(
    # Unbiased
    0.0502, 0.0070, 0.0062,  # All
    0.0545, 0.0060, 0.0000,  # X1 = 1
    0.2858, 0.0467, 0.0000,  # X1 = -1
    # Selection Type 1/2
    0.0542, 0.0071, 0.0017,  # All
    0.0818, 0.0103, 0.0000,  # X1 = 1
    0.0758, 0.0123, 0.0000,  # X1 = -1
    # Selection Type 3
    0.0612, 0.0112, 0.0067,  # All
    0.0601, 0.0098, 0.0000,  # X1 = 1
    0.2858, 0.0467, 0.0000   # X1 = -1
  )
)

# Calculate confidence interval bounds directly
data$ci_lower <- data$mean - data$ci
data$ci_upper <- data$mean + data$ci

# Create the plot with modified aesthetics
p <- ggplot(data, aes(x = signal, y = mean, fill = bias_setup)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                position = position_dodge(width = 0.8),
                width = 0.25) +
  facet_wrap(~X1, scales = "free_y", labeller=label_parsed) +
  labs(x = "Covariance Signal Type",
       y = "Covariance",
       fill = "Bias Setup") +
  theme_Publication() +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold"),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 11),
    strip.text = element_text(size = 12, face = "bold"),
    panel.spacing = unit(2, "lines")
  ) +
  scale_fill_Publication()

# Print the plot
print(p)

# Save the plot
ggsave("covariance_results.png", p, width = 12, height = 6, dpi = 300)