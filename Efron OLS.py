# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:01:42 2020

@author: Ming Cai
"""

#Efron Naive Bootstrap 
"""
This algorithm is intended to bootstrap sample average 
"""

import numpy as np
import scipy.stats as ss
import time

#Init: True Population Simulation 
print("Initializing")

n = 20
#requires more than 8 observations 

#Build Model 
# parameter beta: y = x*beta + e
beta = np.array([5, 3])

# Generate Data Given Distribution 
meanx = np.array([50, 20])
varx = np.array([[3, -1],[-1, 4]])

x = np.random.multivariate_normal(meanx, varx, size = n)
EXX = varx + meanx.reshape(beta.shape[0], 1) @ meanx.reshape(1, beta.shape[0])

#heteroskedastic e but independent of x
#but since sigma is independent of x 
#residual bootstrap is valid 

"""
sdsigma = 10
meansigma = 10
vare = np.random.normal(meansigma, sdsigma, size = n)
e = np.zeros(n)
for i in range(n):
    e[i] = np.random.normal(0, np.sqrt(np.abs(vare[i])))
    
#theoretical asymptotic variance
var_beta = meansigma*np.linalg.inv(EXX)
"""
#Uniform e iid:
ewidth = 5
e = np.random.uniform(-ewidth,ewidth, size = n)
vare = (2*ewidth)**2/12

var_beta = vare*np.linalg.inv(EXX)


#Get y samples
y = x@beta + e

#Sample statistics 
def OLSb(X, Y):
    return np.linalg.inv(X.transpose()@X)@X.transpose()@Y

beta_hat = OLSb(x, y) #beta hat converges at around n = 100

#estimate variance 
def varb(X, Y):
    be = OLSb(X, Y)
    res = Y - X@be
    O = np.diag(np.diag(res.reshape(n, 1)@res.reshape(1, n)))
    XOX = X.transpose()@O@X 
    invX = np.linalg.inv(x.transpose()@x)
    return invX@XOX@invX

var_beta_hat = n**2/(n-1)*varb(x, y) #for an unbiased estimate
rese = y - x@beta_hat

print("Init completed")

#Bootstrap World
B = 40000
m = n 
# allow for m over n, but Wild bootstrap needs to be fixed 

#Pairwise Bootstrap 
b_bp = np.zeros([B, beta_hat.shape[0]])

print("Calculating Pairwise Bootstrap")
a = time.time()
for i in range(B):
    x_bp = np.zeros([m, 2])
    y_bp = np.zeros(m)
    for j in range(m):
        int_ = np.random.randint(0, n-1)
        x_bp[j] = x[int_]
        y_bp[j] = y[int_]
    b_bp[i] = OLSb(x_bp, y_bp)
    
b = time.time()
print("Pairwise Bootstrap calculation completed, time taken: %.2f s"%(b-a))

sort_b_bp = np.sort(b_bp)
b_bp_mean = np.mean(b_bp, axis = 0)
b_bp_median = np.median(b_bp, axis = 0)
diffp = b_bp - beta_hat
var_bp = diffp.transpose() @ diffp/B*m 

#Wild bootstrap (residual + heteroskedasticity)
b_bw = np.zeros([B, beta_hat.shape[0]])
b_br = np.zeros([B, beta_hat.shape[0]])

print("Calculating Wild/Residual Bootstrap")

a = time.time()
for i in range(B):
    z = np.random.normal(size = m)
    e_bw = np.abs(np.random.choice(rese, size = m))*z
    e_br = np.random.choice(rese, size = m)
    b_bw[i] = beta_hat + OLSb(x, e_bw)
    b_br[i] = beta_hat + OLSb(x, e_br)
    
b = time.time()
print("Wild/Residual Bootstrap calculation completed, time taken: %.2f s"%(b-a))
    
b_bw_mean = np.mean(b_bw, axis = 0)
b_bw_median = np.median(b_bw, axis = 0)
diffw = b_bw - beta_hat
var_bw = diffw.transpose() @ diffw/B*m 

b_br_mean = np.mean(b_br, axis = 0)
b_br_median = np.median(b_br, axis = 0)
diffr = b_br - beta_hat
var_br = diffr.transpose() @ diffr/B*m 

#Confidence Intervals

alpha = 0.05
z_l = ss.norm.ppf(alpha/2)
z_u = ss.norm.ppf(1-alpha/2)
t_l = ss.t.ppf(alpha/2, df = n-1)
t_u = ss.t.ppf(1-alpha/2, df = n-1)

#sort 
b_br_s0 = np.sort(b_br[:,0])
b_bw_s0 = np.sort(b_bw[:,0])
b_bp_s0 = np.sort(b_bp[:,0])

b_br_s1 = np.sort(b_br[:,1])
b_bw_s1 = np.sort(b_bw[:,1])
b_bp_s1 = np.sort(b_bp[:,1])



b_true_l = beta - z_u*np.diag(var_beta)
b_true_u = beta - z_l*np.diag(var_beta)

b_hat_l = beta - t_u*np.diag(var_beta)
b_hat_u = beta - t_l*np.diag(var_beta)

ind_l = np.int(alpha/2*B)-1
ind_u = np.int((1-alpha/2)*B)-1

b_br_l0 = b_br_s0[ind_l]
b_br_u0 = b_br_s0[ind_u]
b_br_l1 = b_br_s1[ind_l]
b_br_u1 = b_br_s1[ind_u]

b_bw_l0 = b_bw_s0[ind_l]
b_bw_u0 = b_bw_s0[ind_u]
b_bw_l1 = b_bw_s1[ind_l]
b_bw_u1 = b_bw_s1[ind_u]

b_bp_l0 = b_bp_s0[ind_l]
b_bp_u0 = b_bp_s0[ind_u]
b_bp_l1 = b_bp_s1[ind_l]
b_bp_u1 = b_bp_s1[ind_u]


#Summary 
def Summarize(displayvar=False):
    print("\nobs = %d"%(n))
    print("alpha = %.4f"%(alpha))
    print("\nAsymptotic Theory:")
    for i in range(beta.shape[0]):
        print("beta %d = %.2f"%(i+1, beta[i]))
    if displayvar == True:
        print("Variance:")
        print(var_beta)
    print("95%% Confidence Interval:")
    print("b1: [%.4f, %.4f]"%(b_true_l[0],b_true_u[0]))
    print("b2: [%.4f, %.4f]"%(b_true_l[1],b_true_u[1]))
    print("\nSample:")
    for i in range(beta.shape[0]):
        print("beta %d = %.2f"%(i+1, beta_hat[i]))
    if displayvar == True:
        print("Variance:")
        print(var_beta_hat)
    print("%.0f %% Confidence Interval:"%(100*(1-alpha)))
    print("b1: [%.4f, %.4f]"%(b_hat_l[0],b_hat_u[0]))
    print("b2: [%.4f, %.4f]"%(b_hat_l[1],b_hat_u[1]))
    """
    print("\nResidual Bootstrap:")
    for i in range(beta.shape[0]):
        print("beta %d = %.2f"%(i+1, b_br_mean[i]))
    if displayvar == True:
        print("Variance:")
        print(var_br)
    print("%.0f %% Confidence Interval:"%(100*(1-alpha)))
    print("b1: [%.4f, %.4f]"%(b_br_l0,b_br_u0))
    print("b2: [%.4f, %.4f]"%(b_br_l1,b_br_u1))
    """
    print("\nWild Bootstrap:")
    for i in range(beta.shape[0]):
        print("beta %d = %.2f"%(i+1, b_bw_mean[i]))
    if displayvar == True:    
        print("Variance:")
        print(var_bw)
    print("%.0f %% Confidence Interval:"%(100*(1-alpha)))
    print("b1: [%.4f, %.4f]"%(b_bw_l0,b_bw_u0))
    print("b2: [%.4f, %.4f]"%(b_bw_l1,b_bw_u1))
    print("\nPairwise Bootstrap:")
    for i in range(beta.shape[0]):
        print("beta %d = %.2f"%(i+1, b_bp_mean[i]))
    if displayvar == True:
        print("Variance:")
        print(var_bp)
    print("%.0f %% Confidence Interval:"%(100*(1-alpha)))
    print("b1: [%.4f, %.4f]"%(b_bp_l0,b_bp_u0))
    print("b2: [%.4f, %.4f]"%(b_bp_l1,b_bp_u1))
    
Summarize()