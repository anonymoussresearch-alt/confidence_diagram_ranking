################################
## draw heatmap
###############################
library(reshape2)
library(ggplot2)

nodeLevels = function(data){
  labels = c()
  parameters = list()
  # Set defaults
  if (is.null(parameters$newpage)) {
    parameters$newpage <- TRUE
  } else {
    stopifnot(is.logical(parameters$newpage))
  }
  if (is.null(parameters[["cluster"]])) {
    parameters$cluster <- TRUE
  } else {
    stopifnot(is.logical(parameters$cluster))
  }
  if (is.null(parameters$clusterMerge)) {
    parameters$clusterMerge <- FALSE
  } else {
    stopifnot(is.logical(parameters$clusterMerge))
  }
  if (is.null(parameters$clusterNonAdjacent)) {
    parameters$clusterNonAdjacent <- FALSE
  } else {
    stopifnot(is.logical(parameters$clusterNonAdjacent))
  }
  if (is.null(parameters$transitiveReduction))
    parameters$transitiveReduction <- TRUE
  if (!is.character(parameters$shape))
    parameters$shape <- "roundrect"
  if (is.character(parameters$arrow)) {
    stopifnot(parameters$arrow %in% c("forward", "backward", "both", "none"))
  } else {
    parameters$arrow = "forward"
  }
  if (is.null(parameters$margin)) {
    parameters$margin <- list()
    parameters$margin$rl <- parameters$margin$tb <- 0.125
    parameters$margin$orl <- parameters$margin$otb <- 0.08
  }
  if (is.character(parameters$edgeColor)) {
    stopifnot(parameters$edgeColor %in% colors())
  } else {
    parameters$edgeColor <- "black"
  }
  if (is.character(parameters$nodeColor)) {
    stopifnot(parameters$nodeColor %in% colors())
  } else {
    parameters$nodeColor <- "black"
  }

  nrNodes <- nrow(data)

  # Prepare node identifiers
  if (is.null(rownames(data))) {
    colnames(data) <- rownames(data) <- paste("a", seq_len(nrNodes), sep = "")
  }

  # Setup labels if missing
  if (is.null(labels)) {
    labels <- rownames(data)
  }

  # Convert labels to list with named elements
  labels <- as.list(labels)
  names(labels) <- rownames(data)

  # Remove self-loops
  for (i in seq_len(nrNodes)) {
    data [i, i] <- FALSE
  }

  # Cluster
  groups <- extractGroups(data, parameters$clusterNonAdjacent)
  toRemove <- c()

  for (group in groups) {
    for (i in group) {
      for (j in group) {
        data[i, j] <- FALSE
      }
    }

    if (parameters$cluster) {
      first <- group[1]
      rest <- group[-1]

      rownames(data)[first] <-
        colnames(data)[first] <-
        names(labels)[first] <- paste(rownames(data)[group], collapse = "")

      toRemove <- c(toRemove, rest)
      labels[[first]] <- c(unlist(labels[group]))
    }
  }

  if (!is.null(toRemove)) {
    data <- data[-toRemove, -toRemove]
    labels <- labels[-toRemove]
  }

  nrNodes <- nrow(data)

  # Detect cycles
  tmpData <- data
  toVisit <- which(sapply(1:nrow(data), function(x) {length(which(tmpData[, x])) == 0}) == TRUE)

  while (length(toVisit) > 0) {
    n <- toVisit[1]
    toVisit <- toVisit[-1]

    for (m in which(tmpData[n, ] == TRUE)) {
      tmpData[n, m] <- FALSE

      if (length(which(tmpData[, m])) == 0) {
        toVisit <- c(toVisit, m)
      }
    }
  }

  notRemovedEdges <- which (tmpData == TRUE, arr.ind = TRUE)

  if (nrow(notRemovedEdges) > 0) {
    stop(paste("Cycle detected. Check edges: ",
               paste(sapply(seq_len(nrow(notRemovedEdges)),
                            function(x) { paste(rownames(notRemovedEdges)[notRemovedEdges[x, ]], collapse = "-")} ),
                     collapse = ", "),
               sep = ""))
  }

  # Perform transitive reduction
  if (parameters$transitiveReduction) {
    for (source in seq_len(nrNodes)) {
      stack <- which(data[source, ])
      visited <- rep(F, nrNodes)
      visited[stack] <- T

      while (length(stack) > 0) {
        element <- stack[1]
        stack <- stack[-1]

        children <- which(data[element, ])
        for (child in children) {
          data[source, child] = FALSE
          if (!visited[child]) {
            stack <- c(child, stack)
          }
        }
      }
    }
  }

  # Calculate node levels
  ranks <- rep(1, nrNodes)
  queue <- which(sapply(1:nrow(data), function(x) {length(which(data[, x])) == 0}) == TRUE)
  distances <- rep(1, length(queue))

  while (length(queue) > 0) {
    element <- queue[1]
    queue <- queue[-1]
    dist <- distances[1]
    distances <- distances[-1]
    children <- which(data[element, ])

    for (i in seq_len(length(children))) {
      idx <- which(queue == children[i])

      if (length(idx) == 0) {
        ranks[children[i]] <- dist + 1
        queue <- c(queue, children[i])
        distances <- c(distances, dist + 1)
      } else {
        distances[idx] <- max(distances[idx], dist + 1)
        ranks[children[i]] <- max(ranks[children[i]], dist + 1)
      }
    }
  }

  return(ranks)
}

extractGroups <- function(data, groupNonAdjacent) {
  result <- list()
  itemGroup <- seq_len(nrow(data))

  for (i in seq_len(nrow(data))) {
    for (j in seq_len(nrow(data))) {
      if (isGroup(data, i, j, groupNonAdjacent)) {
        iGroup <- which(itemGroup == itemGroup[i])
        mergable <- TRUE

        for (k in iGroup) {
          if (k != i) {
            if (!isGroup(data, j, k, groupNonAdjacent)) {
              mergable <- FALSE
              break
            }
          }
        }

        if (mergable) {
          itemGroup[j] <- itemGroup[i]
        }
      }
    }
  }

  for (g in unique(itemGroup)) {
    items <- which(itemGroup == g)
    if (length(items) > 1) {
      result[[length(result) + 1]] <- items
    }
  }

  return (result)
}

isGroup <- function(data, i, j, groupNonAdjacent) {
  if ((data[i, j] == TRUE && data[j, i] == TRUE) || groupNonAdjacent == TRUE) {
    iParents <- data[, i]
    jParents <- data[, j]
    iChildren <- data[i, ]
    jChildren <- data[j, ]

    iParents[j] <- FALSE
    jParents[i] <- FALSE
    iChildren[j] <- FALSE
    jChildren[i] <- FALSE

    if (all(iParents == jParents) && all(iChildren == jChildren)) {
      return (TRUE)
    }
  }

  return (FALSE)
}

get_models_by_level <- function(levels) {
  models = seq_along(levels)
  unique_levels <- sort(unique(levels), decreasing = TRUE)
  models_by_level <- lapply(unique_levels, function(lvl) {
    models[levels == lvl & !duplicated(cummax(levels))]
  })
  names(models_by_level) <- unique_levels
  return(models_by_level)
}

get_ranks_by_level <- function(levels) {
  unique_levels <- unique(levels)
  ranks_by_level <- lapply(unique_levels, function(level) {
    startt <- length(which(level<levels)) + 1
    endd <- length(which(level<=levels))
    return(seq(startt,endd))
  })
  names(ranks_by_level) <- unique_levels
  return(ranks_by_level)
}

# Initialize a 20x20 matrix with zeros
n = 20
rep = 50
count_matrix <- matrix(0, nrow = n, ncol = n)
for (i in 0:(rep-1)) {
  # results/out/n20_Lij50_p0.2_h0.2_dim3_m0.1_k1.0+1_alpha0.9_seedn20_Lij50_p0.5_h0.23_lambda1e-05_dim3_m0.1_k1.0+1_num400.0_seed2_c4.npz
  data  = read.table(paste0('results/out/n20_Lij100_p0.3_h0.23_dim3_m0.1_k1.0_alpha0.9_c4_seed',i))
  data = lapply(data, as.logical)
  data = do.call(rbind, data)
  data = t(data)
  
  # Get nodes level
  levels = nodeLevels(data)
  
  # Get possible ranks of nodes by level
  possible_ranks <- get_ranks_by_level(levels)
  
  for (i in seq_along(levels)) {
    level <- as.character(levels[i])
    ranks <- possible_ranks[[level]]
    count_matrix[i, ranks] <- count_matrix[i, ranks] + 1
  }

}

# Print the count matrix
print(count_matrix)


long_df <- melt(as.data.frame(count_matrix))
names(long_df) <- c("Node", "Frequency")
long_df$Rank = rep(1:n,n)
long_df$Node <- factor(long_df$Node) # Creating a Node factor for the rows
long_df$Rank <- as.numeric(as.character(long_df$Rank)) # Ensure Rank is numeric for histogram
long_df$Frequency <- long_df$Frequency/rep # Normalize the frequencies
head(long_df)
################################
## heatmap
###############################

p <- ggplot(long_df, aes(x = Rank, y = Node, fill = Frequency)) +
  geom_tile(color = "white") + # Use geom_tile to create a heatmap
  scale_fill_gradient(low = "white", high = "darkblue") + # Gradient fill from light blue to blue
  theme_minimal() +
  scale_y_discrete(breaks = 1:20) +
  labs(x = 'Rank', y = 'Model', title = 'Heatmap for Rank Frequencies: n=20, L=100, p=0.3, h=0.23, k = 1') +
  theme(
    axis.text.y = element_blank(), # Remove the text labels on the y-axis
    axis.ticks.y = element_blank(), # Remove the ticks on the y-axis
    axis.text.x = element_text(angle = 90, vjust = 0.5), # Rotate x-axis text for better readability
    legend.title = element_blank() # Remove legend title
  )
print(p)

################################
## kernel-smoothed heatmap
###############################
library(dplyr)
library(viridis) 
library(akima)
long_df <- long_df %>%
  group_by(Rank, Node) %>%
  summarise(Frequency = mean(Frequency, na.rm = TRUE), .groups = 'drop')

long_df <- long_df %>%
  mutate(Node = as.integer(sub("V", "", Node)))

long_df$Rank = as.integer(long_df$Rank)
interp_result <- interp(
  x = long_df$Rank, y = long_df$Node, z = long_df$Frequency,
  xo = seq(min(long_df$Rank), max(long_df$Rank), length.out = 500),
  yo = seq(min(long_df$Node), max(long_df$Node), length.out = 500),
  linear = FALSE
)

interp_df <- expand.grid(Rank = interp_result$x, Node = interp_result$y)
interp_df$Frequency = as.vector(interp_result$z)

ggplot(interp_df, aes(x = Rank, y = Node, fill = Frequency)) +
  geom_tile() +
  # scale_fill_viridis_c(option = "D") + 
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5) +
  # scale_fill_gradient(low = "yellow", high = "darkred") +
  # scale_y_discrete(breaks = 1:20) +
  theme_minimal() +
  labs(title = 'Smoothed Heatmap for Rank Frequencies',
       x = 'Rank', y = 'Model') +
  theme(axis.text.x = element_text(angle = 0, vjust = 0.5),
        # axis.text.y = element_text(angle = 0, vjust = 0.5),
        legend.title = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())


################################
## histogram of node ranks
###############################

p <- ggplot(long_df, aes(x = Rank, y = Frequency, fill = Frequency)) +
  geom_col(show.legend = FALSE) + # Use geom_col to create a histogram-like bar plot
  facet_wrap(~ Node, ncol = 1, strip.position = "left") + # Move the labels (V1-V20) to the left
  scale_fill_gradient(low = "lightblue", high = "darkblue") + # Gradient fill from light blue to blue
  scale_y_continuous(breaks = c(0, 1)) + # Show only 0 and 1 on the y-axis
  theme_minimal() +
  labs(x = 'Rank', y = '', title = 'Histograms for Rank Frequencies: n = 20, L = 100, p = 0.2') + # Remove y-axis label
  theme(
    strip.text.x = element_text(size = 12, hjust = 0), # Increase the font size and align left
    strip.background = element_blank(), # Remove the background to the strip labels
    strip.placement = "outside" # Place the strips outside of the panels
  )

# Set up the dimensions of the saved plot
width <- 10
height <- 10 # You may need to adjust the height to fit all 20 histograms

# Save the plot to a file
ggsave("results/out/n20_Lij50_p0.2_h0.2_dim3_m0.1_k1.0_ratio0.1_rank_histograms.png", plot = p, width = width, height = height, dpi = 300)

# P
# Print the plot
print(p)

################################
## violin plot 
###############################
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(ggplot2)
source("https://raw.githubusercontent.com/datavizpyr/data/master/half_flat_violinplot.R")
library(RColorBrewer)

# Assuming long_df is your data frame prepared for plotting
# long_df <- data.frame(Node = ..., Rank = ..., Frequency = ...

long_df_not_normalized <- long_df
long_df_not_normalized$Frequency <- long_df$Frequency*20 

# Transform the data frame
long_df_expanded <- long_df_not_normalized %>%
  ungroup() %>%
  slice(rep(row_number(), Frequency)) %>%
  mutate(Frequency = 1) %>%
  select(-Frequency) %>%
  arrange(Node, Rank)

palette <- brewer.pal(n = min(length(unique(long_df$Node)), 9), name = "PRGn")

# If you have more than 12 nodes, you may concatenate multiple palettes
if(length(unique(long_df$Node)) > 9) {
  palette <- c(palette, brewer.pal(n = min(length(unique(long_df$Node)) - 9, 12), name = "PRGn"))
}

p <- ggplot(long_df_expanded, aes(x = Node, y = Rank, fill=Node)) +
  geom_flat_violin(alpha =0.4, trim = FALSE) +
  scale_fill_manual(values = palette) +
  geom_boxplot(width = .2, alpha = .6, fatten = NULL, show.legend = FALSE) +
  # scale_fill_gradient(low = "lightblue", high = "blue") + # Gradient fill from light blue to blue
  scale_y_continuous(breaks = 1:20) + # Simplify the y-axis to show only 0 and 1
  # scale_fill_brewer(palette = "Dark2", name = "") +
  facet_grid(cols = vars(Node), scales = "free", space = "free") +
  theme_minimal() +
  labs(x = 'Rank', y = '', title = 'Violin Plots for Rank Frequencies: : n = 20, L = 50, p = 0.2') + # Remove y-axis label
  theme(text = element_text(size=10),
        legend.position = 'none',
        axis.text = element_text(size=10),
        strip.text = element_text(size = rel(1)),
        axis.text.x=element_text(angle=20, hjust=1, size=10))# Place the strips outside of the panels


print(p)


