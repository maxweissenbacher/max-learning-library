library(tidyverse)
library(geometry)

# Define base function to be modeled
sinoid <- function(x, period) {
  sin(x*(2*pi)/period)
}

# Function to generate noisy samples from base function
sample_sinoid <- function(n, # number of samples
                          period, # periodicity
                          noise, # amount of noise
                          lower, # lower bound for x
                          upper # upper bound for x
) {
  x <- runif(n, min=lower, max=upper)
  y <- sinoid(x,period) + noise * rnorm(n)
  return(data.frame('x' = x, 'y' = y))
}
