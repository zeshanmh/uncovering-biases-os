library(ggplot2)
library(tidyr)
library(dplyr)
library(gridExtra)

# Get the directory of the current script
current_dir <- dirname(rstudioapi::getSourceEditorContext()$path)

# Read the CSV files from the script's directory
data_rct <- read.csv(file.path(current_dir, "data_rct.csv"))
data_obs <- read.csv(file.path(current_dir, "data_obs.csv"))

# Add a 'Source' column to both datasets
data_rct <- data_rct %>% mutate(Source = "RCT")
data_obs <- data_obs %>% mutate(Source = "OBS")

# Combine the datasets
combined_data <- rbind(data_rct, data_obs)

# Define Tableau colors
tableau_colors <- c("RCT" = "#4E79A7", "OBS" = "#E15759")

# 1. Scatter plot (A = 1 only)
p_scatter <- ggplot(combined_data %>% filter(A == 1), aes(x = X, y = Y, color = Source)) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = tableau_colors) +
  labs(title = "Y vs X (Treatment Group Only)",
       x = "X", y = "Y") +
  theme_bw() +
  theme(legend.position = "bottom")

# 2. Density plot for Y
p_density <- ggplot(combined_data, aes(x = Y, fill = Source)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = tableau_colors) +
  labs(title = "Distribution of Y",
       x = "Y", y = "Density") +
  theme_bw() +
  theme(legend.position = "bottom")

# 3. Bar plot for A
p_bar <- ggplot(combined_data, aes(x = factor(A), fill = Source)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = tableau_colors) +
  labs(title = "Distribution of A",
       x = "A", y = "Count") +
  theme_bw() +
  theme(legend.position = "bottom")

# Combine the plots
combined_plot <- grid.arrange(p_scatter, p_density, p_bar, ncol = 2, nrow = 2)

# Save the combined plot
ggsave(
  file.path(current_dir, "combined_analysis_plot_selection_bias.png"),
  plot = combined_plot, width = 12, height = 10, dpi = 300
)

# Print the combined plot
print(combined_plot)
