# Libraries
library(tidyverse)
library("Metrics")

#--------------------------------------------#
# Sequential Importance Sampling (bootstrap)
#--------------------------------------------#

# Utility functions
normalise <- function(vec) {
  s = sum(vec)
  for (i in 1:length(vec)) {
    vec[i] <- vec[i]/s
  }
  return(vec)
}

resample <- function(w) {
  N = length(w)
  sample <- rmultinom(1,N,w)
  res = c()
  for (i in 1:N) {
    res = append(res, rep(i,times=sample[i]))
  }
  return(res)
}

# Computing effective sample size
ess <- function(w) {
  return(1/(sum(w^2)))
}

# Sequential Importance (Re-)Sampling
SequentialImportanceSampling <- function(data, # observed data
                                         params, # model parameters
                                         resample = FALSE, # Resampling?
                                         thresh = 0.7, # Threshold for resampling
                                         move = FALSE, # Particle Move?
                                         num_particles = 100, # number of particles
                                         rsmpl_steps = 5, # number of resampling steps,
                                         stepSize = 1 # MCMC step size
                                         ) {
  # Extracting parameters
  Y <- data
  N <- num_particles
  M <- rsmpl_steps
  t <- length(Y)
  rho   <- params$rho
  sigma <- params$sigma
  tau   <- params$tau
  # Setting up matrices to store particles,
  # weights and effective sample size
  Xsim <- matrix(0,N,t)
  W    <- matrix(0,N,t)
  ESS  <- numeric(t)
  # Parameters for resampling move
  numAccepted <- c()
  # Metropolis Hastings random walk move
  MCMCmove <- function(n, #time step
                       steps = M # number of MCMC steps
                       ) {
    # This vector will contain the updated particle position at time n
    moved_particles <- c()
    # Counting the number of accepted steps to compute acceptance ratio
    numAcceptedN <- 0
    # Computing standard deviation across particles to guide random walk
    s <- sd(Xsim[,n])
    for (i in 1:N) {
      # Metropolis Hastings on particle i
      tail <- Xsim[i,n]
      for (m in 1:M) {
        P <- tail + stepSize*rnorm(length(tail), sd=s)
        den <- dnorm(P,mean=rho*Xsim[i,n-1],sd=sigma)*dnorm(Y[n], mean = 0, sd = tau*exp(P/2))
        num <- dnorm(Xsim[i,n],mean=rho*Xsim[i,n-1],sd=sigma)*dnorm(Y[n], mean = 0, sd = tau*exp(Xsim[i,n]/2))
        accRatio <- min(1,den/num)
        if (runif(1) < accRatio) {
          tail <- P
          numAcceptedN <- numAcceptedN + 1
        }
      }
      moved_particles <- append(moved_particles,tail)
    }
    return(list('X_n_moved' = moved_particles, 'accRatio' = numAcceptedN/(N*M)))
  }
  
  # Initial data for simulation
  for (i in 1:N) {
    Xsim[i,1] <-  rnorm(1, mean = 0, sd = sigma^2/(1-rho^2))
    W[i,1] <- dnorm(Y[1], mean = 0, sd = tau*exp(Xsim[i,1]/2))
  }
  W[,1] <- normalise(W[,1])
  ESS[1] <- ess(W[,1])
  # Simulation steps
  for (n in 2:t) {
    # Importance sampling
    for (i in 1:N) {
      Xsim[i,n] <- rho*Xsim[i,n-1]+sigma*rnorm(1)
      W[i,n] <- W[i,n-1] * dnorm(Y[n], mean = 0, sd = tau*exp(Xsim[i,n]/2))
    }
    # Normalising the weights
    W[,n] <- normalise(W[,n])
    if (sum(W[,n]) == 0.) {
      cat("Error in iteration", n, "\n")
      stop("Sum of importance weights = 0 up to numeric precision. Possible solution: use more particles. Aborting computation.")
    }
    # Computing effective sample size
    ESS[n] <- ess(W[,n])
    # Resampling if below threshold
    if (resample == TRUE && ESS[n] < thresh * N) {
      Xsim <- Xsim[resample(W[,n]),]
      # Updating the weights to now be equidistributed
      W[,n] <- rep(1/N, times=N)
      # Metropolis Hastings steps on the tail
      if (move == TRUE) {
        res <- MCMCmove(n)
        Xsim[,n] <- res$X_n_moved
        numAccepted <- append(numAccepted, res$accRatio)
      }
    }
  }
  # Return the particles and weights
  return(list('X' = Xsim, 'W' = W, 'ESS' = ESS, 'accRatio' = numAccepted))
}

# Extracting estimate of filter mean for each time step
filterMean <- function(Xsim,W) {
  t <- dim(Xsim)[2]
  mu <- numeric(t)
  for (n in 1:t) {
    mu[n] <- dot(Xsim[,n], W[,n])
  }
  return(mu)
}

# Extracting estimate of filter variance for each time step
filterVariance <- function(Xsim,W) {
  t <- dim(Xsim)[2]
  variance <- numeric(t)
  C <- Xsim - filterMean(Xsim,W)
  C <- C**2
  for (n in 1:t) {
    variance[n] <- dot(C[,n], W[,n])
  }
  return(variance)
}

# Monte Carlo Variance of the mean
montecarlovar <- function(data, # observed data
                          params, # model parameters
                          resample, # Resampling?
                          thresh = 0.7, # Threshold for resampling
                          move, # Particle Move?
                          num_particles = 300, # number of particles
                          rsmpl_steps = 5, # number of resampling steps,
                          stepSize = 1, # MCMC step size,
                          num_models = 100 # Number of models
                          ) {
  t <- length(data)
  K <- num_models
  means <- matrix(0,K,t)
  for (i in 1:K) {
    res <- SequentialImportanceSampling(data,
                                        params,
                                        resample,
                                        thresh,
                                        move,
                                        num_particles,
                                        rsmpl_steps,
                                        stepSize
                                        )
    mu <- filterMean(res$X,res$W)
    means[i,] <- mu
  }
  montecarlovar <- numeric(t)
  for (n in 1:t) {
    montecarlovar[n] <- var(means[,n])
  }
  return(montecarlovar)
}

