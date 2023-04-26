import numpy as np
from scipy.optimize import minimize

# Define the log-likelihood function
def log_likelihood(theta, x):
    p = f(x, theta)
    p[p == 0] = 1e-10  # Replace zero probabilities with a small positive number
    return np.sum(np.log(p))


# Define the probability density function for a Gaussian distribution
def f(x, theta):
    mu = theta[0]  # mean
    sigma = theta[1]  # standard deviation
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Generate some data from the probability distribution
x = np.random.normal(loc=0, scale=1, size=100)

# Define the initial guess for the parameters
theta0 = np.array([0, 1])

# Minimize the negative log-likelihood function
res = minimize(lambda theta: -log_likelihood(theta, x), theta0)

# Print the estimated parameters
print(res.x)