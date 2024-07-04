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

def m_estimator(X, y, beta_init, sigma_init, estimator = "", c=1, tol=1e-6, max_iter=1000):
    n, p = X.shape
    
    beta = beta_init
    res = y - X@beta
    i=0
    
    match estimator:
        case "Huber":
            weights_function = Huber_weights_function
        case "Tukey":
            weights_function = Tukey_weights_function
        case _:
            weights_function = lambda x, _: np.ones(n)
    
    
    while i < max_iter:
        i += 1

        weights = weights_function(res/sigma_init, c)
        W = np.diag(weights)
        
        beta_new = np.linalg.pinv(X.T@W@X)@X.T@W@y

        if np.linalg.norm(beta_new-beta, ord= 2) / np.linalg.norm(beta, ord= 2) < tol:
            break
            
        beta = beta_new
        res = y - X@beta
        
        
    print(i)
    return beta_new

def s_estimator(X, y, beta_init, sigma_init, estimator = "", c=1, b=1, tol=1e-6, max_iter=1000):
    
    n, p = X.shape
    
    beta = beta_init
    res = y - X@beta
    sigma = sigma_init
    i=0
    
    match estimator:
        case "Huber":
            weights_function = Huber_weights_function
        case "Tukey":
            weights_function = Tukey_weights_function
        case _:
            weights_function = lambda x, _: np.ones(n)
    
    
    while i < max_iter:
        i += 1

        weights = weights_function(res/sigma, c)
        W = np.diag(weights)
        
        beta_new = np.linalg.pinv(X.T@W@X)@X.T@W@y

        if np.linalg.norm(beta_new-beta, ord= 2) / np.linalg.norm(beta, ord= 2) < tol:
            break
            
        sigma = np.sqrt(np.sum(weights*res**2)/(n*b))
        beta = beta_new
        res = y - X@beta
        
        
    print(i)
    return beta_new, sigma

def mm_estimator(X, y, beta_init, sigma_init, estimator = "", c=1, b=1, tol=1e-6, max_iter=1000):
    beta = beta_init
    i=0
    
    while i < max_iter:
        i += 1
        _, sigma = s_estimator(X, y, beta, sigma_init, estimator, c, b, tol, max_iter)
        beta_new = m_estimator(X, y, beta, sigma_init, estimator, c, tol, max_iter)
        
        if np.linalg.norm(beta_new-beta, ord= 2) / np.linalg.norm(beta, ord= 2) < tol:
            break
        
        beta = beta_new
    
    return beta_new
        
def mm_estimator2(X, y, beta_init, sigma_init, estimator = "", c=1, b=1, tol=1e-6, max_iter=1000):
    
    n, p = X.shape
    
    beta = beta_init
    res = y - X@beta
    sigma = sigma_init
    X_plus = np.linalg.pinv(X.T@X)@X.T
    i=0
    
    match estimator:
        case "Huber":
            score_function = Huber_score_function
        case "Tukey":
            score_function = Tukey_score_function
        case _:
            score_function = lambda x, _: x
    
    
    while i < max_iter:
        i += 1
        
        res_pseudo = score_function(res/sigma, c)*sigma
        
        sigma = np.linalg.norm(res_pseudo, ord= 2)/(np.sqrt(2*n*c))
        
        res_pseudo = score_function(res/sigma, c)*sigma
        
        beta_new = beta + X_plus@res_pseudo

        if np.linalg.norm(beta_new-beta, ord= 2) / np.linalg.norm(beta, ord= 2) < tol:
            break
            
        beta = beta_new
        res = y - X@beta
    
    print(i)
        
    return beta

n = 10000
p= 10
X = np.random.normal(0, 1, (n,p+1))
X[:,0] = 1

noise = np.random.normal(0, 4, n)
beta = np.array([i for i in range(p+1)])
y = X@beta + noise

beta_init = np.ones(p+1)
res = y - X@beta_init
sigma_init = 1.4826*np.median(np.abs(res-np.median(res, axis=0)), axis= 0)

beta_est_m = m_estimator(X, y, beta_init, sigma_init, "Tukey", c=10)
beta_est_s, sigma_est_s = s_estimator(X, y, beta_init, sigma_init, "Tukey", c=10)
beta_est_mm = mm_estimator(X, y, beta_init, sigma_init, "Tukey", c=10)
beta_est_mm2 = mm_estimator2(X, y, beta_init, sigma_init, "Tukey", c=10)

print(beta_est_m)
print(beta_est_s)
print(beta_est_mm)
print(beta_est_mm2)
