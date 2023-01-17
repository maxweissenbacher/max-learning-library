library(tidyverse)
library(geometry)

# Gaussian kernel
gaussian_kernel <- function(sd = 1.) {
  function(x1,x2) dnorm(x1-x2,sd=sd)
}

# Exp-Sine-Squared kernel
ExpSineSquared_kernel <- function(p,l) {
  function(x1,x2) exp(-2*(sin(abs(x1-x2)*pi/p)**2)/l**2)
}

# Rational quadratic kernel
RationalQuadratic_kernel <- function(alpha,l) {
  function(x1,x2) (1+(abs(x1-x2)**2)/(2*alpha*(l**2)))**(-alpha)
}

# Kernel smoothing for general kernels
smoother <- function(X, Y, kernel) {
  function(x) modify(x, ~ dot(Y,kernel(X, . ))/sum(kernel(X, . )))
}

# Computing the Gram Matrix for a kernel
gram_matrix <- function(X, kernel) {
  n <- length(X)
  K <- matrix(0,n,n)
  for (i in 1:n) {
    for (j in 1:n) {
      K[i,j] <- kernel(X[i],X[j])
    }
  }
  K
}

# Kernel ridge regression for general kernels
kernelRidgeRegression <- function(X, Y, kernel, lambda) {
  K <- gram_matrix(X,kernel)
  A <- K + lambda*diag(length(X))
  alpha <- solve(A, Y)
  function(x) modify(x, ~ dot(alpha,kernel(X, . )))
}
