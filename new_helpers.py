#########################
# New helper functions (before merge into main helpers.py)
#########################

import numpy as np
import scipy.stats as stats
from scipy.stats import genpareto
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn


def find_gpd_mle(data, threshold = 0.001, find_std = True, return_nll = True):
    """
    Function to find the MLE (and standard error) of the GPD model for a given data set
    """
    def neg_log_likelihood(params, data):
        scale, shape = params 
        if shape == 0:
            return np.inf  # Avoid division by zero
        return -np.sum(genpareto.logpdf(data, shape, loc=threshold, scale=scale))
    
    #Initial parameter guesses
    initial_guess = (1, 0.5)

    # Perform maximum likelihood estimation
    result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='Nelder-Mead')
    minimized_nll = result.fun
    scale_mle, shape_mle = result.x
    # Check if the optimization was successful
    if not result.success:
        raise ValueError('MLE optimization failed: ' + result.message)

    if find_std:
        # Find standard errors
        # Compute Hessian matrix
        def hessian(func, params, args):
            def get_gradient(func, params, args, epsilon=1e-5):
                grad = []
                for i in range(len(params)):
                    params_plus = params.copy()
                    params_plus[i] += epsilon
                    grad.append((func(params_plus, *args) - func(params, *args)) / epsilon)
                return grad
            grad = get_gradient(func, params, args)
            hess = []
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += 1e-4 * params_plus[i]
                grad_plus = get_gradient(func, params_plus, args)
                hess.append([(grad_plus[j] - grad[j]) / (1e-4 * params_plus[i]) for j in range(len(grad))])
            return np.array(hess)

        hess = hessian(neg_log_likelihood, result.x, (data,))

        # Compute standard errors
        covariance_matrix = np.linalg.inv(hess)
        scale_se, shape_se = np.sqrt(np.diagonal(covariance_matrix))
        
        if return_nll:
            return covariance_matrix, scale_mle, shape_mle, scale_se, shape_se, minimized_nll
        else:
            return covariance_matrix, scale_mle, shape_mle, scale_se, shape_se
    else:
        if return_nll:
            return scale_mle, shape_mle, minimized_nll
        else:
            return scale_mle, shape_mle


def goodness_of_fit(observed, expected, method = 'chi2', nbins = 20, log_scale = False):
    """
    Function to perform goodness of fit tests for the GPD model
    """
    if method == 'chi2':
        # we now need to bin the data
        bins = np.linspace(0, 1, 21) # 20 bins
        observed_counts, _ = np.histogram(observed, bins=bins)
        expected_counts, _ = np.histogram(expected, bins=bins)
        # perform the chi-squared test
        chi2, p = stats.chisquare(observed_counts, expected_counts, ddof=20-2)
        return chi2, p
    elif method == 'ks':
        # perform the Kolmogorov-Smirnov test
        if log_scale:
            observed = np.log(observed)
            expected = np.log(expected)
        D, p = stats.ks_2samp(observed, expected)
        return D, p
    elif method == 'AD':
        # perform the Anderson-Darling test
        if log_scale:
            observed = np.log(observed)
            expected = np.log(expected)
        A, crit, sig = stats.anderson_ksamp([observed, expected])
        return A, crit, sig
    else:
        raise ValueError('method should be either "chi2" or "ks" or "AD"')
    
def find_gpd_bounds(x, data, scale_mle, shape_mle, scale_se, shape_se, covariance_matrix, n_simulations = 5000, n_bootstrap = 1000):
    """
    Function to find the lower and upper bounds within 95% confidence intervals for the GPD model
    """
    # Define the GPD distribution
    dist = mvn( mean=[scale_mle,shape_se], cov=covariance_matrix )

    # intervals to generate the bounds
    scale_lower = scale_mle - 1.96 * scale_se 
    scale_upper = scale_mle + 1.96 * scale_se
    shape_lower = shape_mle - 1.96 * shape_se
    shape_upper = shape_mle + 1.96 * shape_se

    # initialize the arrays to store the bounds
    distance_above = 0
    distance_below = 0
    
    intensities = np.linspace(np.min(data[x]), np.max(data[x]), n_bootstrap)
    # find the pdf of the mle gpd
    gpd_mle = genpareto.pdf(intensities, shape_mle, loc=0.001, scale=scale_mle)

    for _ in range(n_simulations):
        scale_samples = np.random.uniform(scale_lower, scale_upper)
        shape_samples = np.random.uniform(shape_lower, shape_upper)
        # discard if the cdf is less than 0.025 or greater than 0.975
        if dist.cdf([scale_samples, shape_samples]) < 0.025 or dist.cdf([scale_samples, shape_samples]) > 0.975:
            continue
        else:
            gpd_vals = genpareto.pdf(intensities, shape_samples, loc=0.001, scale=scale_samples)

            # find the furthest line above and below the mle gpd
            if np.max(gpd_vals - gpd_mle) > distance_above:
                distance_above = np.max(gpd_vals - gpd_mle)
                scale_above = scale_samples
                shape_above = shape_samples
            elif np.max(gpd_mle - gpd_vals) > distance_below:
                distance_below = np.max(gpd_mle - gpd_vals)
                scale_below = scale_samples
                shape_below = shape_samples
            else:
                continue

    return scale_above, scale_below, shape_above, shape_below