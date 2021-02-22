# Load necessary packages

# Clear environment
rm(list = ls())

library(tidyverse)
library(caret)
library(skimr)
library(janitor)
library(factoextra)
library(NbClust)
library(ISLR)
library(data.table)
library(ggplot2)
library(plyr)
library(psych)


# Load datasets

genes <- read_csv("https://www.statlearning.com/s/Ch10Ex11.csv", col_names = FALSE) %>%
  t() %>% as_tibble()  # the original dataset is of dimension 1000x40 so we transpose it
dim(data)

# a) PCA on this data with scaling features

pca_result <- prcomp(genes, scale = TRUE)

# b) Visualize data points in the space of the first two principal components (look at the fviz_pca_ind function). 
# What do you see in the figure?

fviz_pca_ind(pca_result)

# The graph shows a quite nice spacial differentiation when the PC1 is being considered. 
# A considerable amount of observations can easily be distinguished from the other.
# PC2 doesn't seem to be that influential.

# c) Which individual features can matter the most in separating diseased from healthy? 

# PC1 - retrieve the most important variables
PC1_important_vars <- head(sort(abs(pca_result$rotation[,1]), decreasing = TRUE),2)
PC1_important_vars

# V502 and V589

genes %>%
  ggplot(aes( x = V502, y = V589)) +
  geom_point()

cor(genes$V502, genes$V589 , method = c("pearson", "kendall", "spearman"))

# The most important variables seems to be correlated to each other demonstrating a quite linear correlation pattern 
# with a higher coefficient (0.78). 
