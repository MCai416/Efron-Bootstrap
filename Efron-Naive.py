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

#Init: True Population Simulation 

n = 10
#requires more than 8 observations 
alpha = 0.05 # Percentile 

#Normal 
normal = True
mean = -5
var = 33**2

x = np.random.normal(mean, np.sqrt(var), n)

#For normal only
z_lower = ss.norm.ppf(alpha/2) 
z_upper = ss.norm.ppf(1-alpha/2)
ci_l = mean + z_lower*np.sqrt(var/n)
ci_u = mean + z_upper*np.sqrt(var/n) 

"""
#Uniform 
xmin = 5
xmax = 20

x = np.random.uniform(xmin, xmax, n)

mean = (xmax+xmin)/2
var = (xmax-xmin)**2/12
"""
"""
#Exponential 
lambda_ = 0.5
mean = 1/lambda_
var = 1/(lambda_**2)

x = np.random.exponential(1/lambda_, n)
"""
#Sort x
sorted_x = np.sort(x)

#Sample statistics 

aver_sam = np.mean(x)
var_sam = np.var(x)

aver_var_sam = var_sam/n

"""
given asymptotics of x is normal, 95% confidence interval is estimated using t dist
"""

# Confidence Intervals

t_lower = ss.t.ppf(alpha/2, df = n-1) #Quantile 
t_upper = ss.t.ppf(1-alpha/2, df = n-1) #Quantile

x_lower = aver_sam - t_upper*np.sqrt(aver_var_sam)
x_upper = aver_sam - t_lower*np.sqrt(aver_var_sam)

#Efron Bootstrap 

m = n #we allow for m over n bootstraps in the future 
B = 20000 #number of draws <= n factorial

aver_b = np.zeros(B)
t_b = np.zeros(B)

for i in range(B):
    x_b = np.random.choice(x, m) # draw function
    aver_b[i] = np.mean(x_b) 
    var_x_b = np.var(x_b)/n
    t_b[i] = (aver_b[i] - aver_sam)/np.sqrt(var_x_b)
    
#Percentile Bootstrap # Simply construct confidence interval from the samples 
sort_aver_b = np.sort(aver_b)

B_lower = np.int(alpha/2*B)-1
B_upper = np.int(B*(1-alpha/2))-1

x_bp_aver = np.mean(aver_b)
x_bp_median = sort_aver_b[np.int(0.5*B)-1]
x_bp_var = np.var(aver_b)

x_bp_l = sort_aver_b[B_lower]
x_bp_u = sort_aver_b[B_upper]


#t Bootstrap # calculate bootstrap t statistic and inference using it 
sort_t_b = np.sort(t_b)
t_bt_l = sort_t_b[B_lower]
t_bt_u = sort_t_b[B_upper]

x_bt_l = aver_sam - t_bt_u*np.sqrt(aver_var_sam)
x_bt_u = aver_sam - t_bt_l*np.sqrt(aver_var_sam)


def report(norm):
    print("\nobs = %d"%(n))
    print("\nAsymptotic Theory:")
    print("True Value: %.2f True Var: %.2f"%(mean, var/n))
    if norm == True:
        print("%.0f %% confidence interval: [%.2f, %.2f]"%(100*(1-alpha),ci_l, ci_u))
    print("\nSample:")
    print("Sample stat: %.2f Var: %.2f"%(aver_sam, aver_var_sam))
    print("Asymptotic t(alpha/2): %.2f t(1-alpha/2): %.2f"%(t_lower,t_upper))
    print("%.0f %% confidence interval: [%.2f, %.2f]"%(100*(1-alpha),x_lower, x_upper))
    print("\nPercentile Bootstrap:")
    print("Mean: %.2f Median: %.2f Var: %.2f"%(x_bp_aver, x_bp_median, x_bp_var))
    print("%.0f %% confidence interval: [%.2f, %.2f]"%(100*(1-alpha), x_bp_l, x_bp_u))
    print("\nt Bootstrap:")
    print("Bootstrap t(alpha/2): %.2f, t(1-alpha/2): %.2f"%(t_bt_l, t_bt_u))
    print("%.0f %% confidence interval: [%.2f, %.2f]"%(100*(1-alpha), x_bt_l, x_bt_u))

report(normal)
    