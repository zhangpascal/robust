import numpy as np
import matplotlib.pyplot as plt


def Huber_score_function(x, c): 
    return np.where((np.abs(x)-c) <= 0, x ,c*np.sign(x))

def Huber_weights_function(x, c):
    return  np.where(x!=0,Huber_score_function(x, c)/x , 1)

def Tukey_score_function(x, c):
    return np.where((np.abs(x)-c) <= 0, x*(1-(x/c)**2)**2 ,0)

def Tukey_weights_function(x, c):
    non_zero = np.array(x!=0)
    return  np.where(x!=0, Tukey_score_function(x, c)/x, 1)


def m_estimator_loc(x, mu, sigma, estimator , c, tol, max_iter):
    
    flag = False
    i=0
    
    match estimator:
        case "Huber":
            weights_function = Huber_weights_function
        case "Tukey":
            weights_function = Tukey_weights_function
        case _:
            weights_function = lambda x, _: np.ones(x.size)

    while not flag and i < max_iter:

        res = x - mu
        
        weights = weights_function(res/sigma, c)
        mu_new = np.sum(weights * x )/ np.sum(weights)

        if np.abs(mu_new - mu) < tol:
            flag = True
            
        mu = mu_new
        
        i += 1

    return mu

def m_estimator_scale(x, mu, sigma, estimator, c, b, tol, max_iter):
    
    flag = False
    i=0
    
    match estimator:
        case "Huber":
            weights_function = Huber_weights_function
        case "Tukey":
            weights_function = Tukey_weights_function
        case _:
            weights_function = lambda x, _: np.ones(x.size)

    while not flag and i < max_iter:

        res = x - mu
        
        weights = weights_function(res/sigma, c)
        sigma_new = np.sqrt(np.sum(weights*res**2)/(weights.size*b))

        if np.abs(sigma_new - sigma) < tol:
            flag = True
            
        sigma = sigma_new
        
        i += 1

    return sigma

def m_estimator(x, mu, sigma, estimator = "", c=1, b=1, tol=1e-6, max_iter=1000):
    mu_est = m_estimator_loc(x, mu, sigma, estimator, c, tol, max_iter)
    sigma_est = m_estimator_scale(x, mu, sigma, estimator, c, b, tol, max_iter)
    
    return mu_est, sigma_est
    

n = 10000
x = np.random.normal(2,4,n)

mu0 = np.median(x)
sigma0 = 1.4826*np.median(np.abs(x-np.median(x)))

mu_mest, sigma_mest = m_estimator(x, mu0, sigma0, estimator="Huber",c=6)


print(mu_mest, sigma_mest)

