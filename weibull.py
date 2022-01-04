from scipy.stats import weibull_min
from scipy.optimize import minimize
from ctc.simulation import CTCCircuitSimulator

import numpy as np

import matplotlib.pyplot as plt


def evaluate(c, scale, emp_cdf, x_values):
    cdf = weibull_min.cdf(x_values, c=c, scale=scale)
    errors = np.array([emp_cdf[i] - cdf[i] for i in range(len(emp_cdf))])
    return np.linalg.norm(errors)


def fit_weibull(emp_cdf, x_values):
    result = minimize(evaluate, args=[emp_cdf, x_values], x0=np.array([1.5, 1.]), bounds=[(1., 2.), (0.00001, 10)])
    print(result.message)
    return result.x


if __name__ == '__main__':
    iterations = range(1, 22)
    sim = CTCCircuitSimulator(size=2, k_value=0, ctc_recipe="brun")
    probs = sim.test_convergence(0.5, 1, 21, 1, cloning="no_cloning")
    print("x =", fit_weibull(probs, iterations))
