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
library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)


# Load data

data <- USArrests 
skim(data)

# a) Think about any data pre-processing steps you may/should want to do before applying clustering methods. Are there any?

# 1) Data has only numeric values and few number of dimensions, because K-means use distance measures to determine similarities
sapply(data, class)
# The data set has only numeric values and only four variables. 

# 2) Noises or outliers: K-means is sensitive to outliers. (Symmetric Distributions).
multi.hist(data[,sapply(data, is.numeric)])
# Although it's possible to visualize some outliers in the variables distribution (Rape) and some skewness, 
# due to the lower amount of observations, no transformation or exclusion will be done.

# 3) Variables on the same scale. Scaling will be done on the prcomp (scale = true)

# 4) No multicollinearity between the variables: 
# Multicolinearity Check

ggpairs(data)

# Despite the fact that murder and assault showed a quite expressive correlation value - 0.8, due to the reduced amount of variables
# and that the value is still lower than 0.9.(what could be considered a really problematic colinearity). 

# b) Determine the optimal number of clusters as indicated by NbClust heuristics.

nb <- NbClust(data, method = "kmeans",
              min.nc = 2, max.nc = 49, index = "all")

# Considering the amount of observations (50), the maximum number of clusters should be set to 49. 
# Any other higher number of observations wouldn't be possible to run (n-1).
# The best number of clusters based on the Hurbet Index is 2. 

# c) Use the k-means method to cluster states using the number of clusters found in a) and anything else that you think that makes sense. 
# Plot observations colored by clusters in the space of urban population and another (crime-related) variable.

# KM Clusters
km <- kmeans(data, centers = 2)

# Creating a column with a clusters
data_w_clusters <- cbind(data,
                         data.table("cluster" = factor(km$cluster)))

# Graph UrbanPop vs Assaults
ggplot(data_w_clusters,
       aes(x = UrbanPop, y = Assault, color = cluster)) +
  geom_point()

# Graph UrbanPop vs Rape
ggplot(data_w_clusters,
       aes(x = UrbanPop, y = Rape, color = cluster)) +
geom_point()

# d) Perform PCA and get the first two principal component coordinates for all observations by

pca_result <- prcomp(data, scale = TRUE)
first_two_pc <- as_tibble(pca_result$x[, 1:2])
data_w_pca <- cbind(data_w_clusters, first_two_pc)
PCs <- data_w_pca %>%
     ggplot(aes(x = PC1, y = PC2, fill = cluster, color = cluster)) +
     geom_point()

Variance <- pca_result$sdev^2
percentage_variance <- Variance / sum(Variance)
percentage_variance

# Cumulative PVE plot
PVE <- qplot(c(1:4), cumsum(percentage_variance)) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab(NULL) + 
  ggtitle("Cumulative Scree Plot") +
  ylim(0,1)

grid.arrange(PCs, PVE, ncol = 2)

# How do clusters relate to these?

# Taking a closer look under the PCs composition, it's clear to assume that the PC1 is related to the variables Murder, Rape and Assault.
# On the other hand, the PC2 is roughly explained by the Urbanization one. The PC1 x PC2 graph shows the distribution of the dots based on those characteristics. 
# Cities far more right in the graph showed those crimes variables with higher values and cities on the top, higher population. 
# The optimal clustering separated the dataset into two groups and the distribution of the dots and his colors are impacted mostly by the variance explained by those 2 PC components.
# The cumulative scree plot on the right demonstrates how much it is explained by each PC, PC1 (62%) and PC2 (24%).
