# Resampling Methods by the Bootstrap 

1)Efron-Naive looks at percentile bootstrap and t-stat bootstrap 
  Results displayed on "Naive Average.JPG"
  
2)Efron OLS looks at residual, wild and pairwise bootstrap under linear regression (they are the same for nonlinear regressions)
  Results displayed on "OLS.JPG"

3)Parametric Bootstrap in the time domain: ARMA(p,q)/Sieve Bootstrap. Algorithm by Franke and Kreiss (1992) (Bootstrapping Stationary Autoregressive Moving-Average Models) 

Overall Bootstrap performs just as well as the estimators using asymptotic theory 

The reason why bootstrap didn't perform better than asymptotic theory is because of information matrix. The estimators are the best given the specifications. But it is very surprising that bootstrap can perform just as well as asymptotic estimators. 

ARMA(p,q) Bootstrap is comptutationally costly. Under the algorithm of Franke and Kreiss (1992), each step requires 1) drawing boostrap samples, 2) setting up sequence, 3) least squares estimation. It takes significantly longer if sample size increases. 

Thus, for statistics that have complicated distribution, bootstrap is a good and simple way to provide distribution of the statistics. 

Plans for the future: 

  1) Nonparametric Moving Block (Circular and Stationary) 
  
  2) Order statistics (non regular case (m=o(n) bootstrap)) 
  
  3) Subsampling (not really bootstrap but a good comparison) 
