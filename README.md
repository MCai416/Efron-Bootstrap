Resampling Methods by the Bootstrap 

1)Efron-Naive looks at percentile bootstrap and t-stat bootstrap 
  Results displayed on "Naive Average.JPG"
  
2)Efron OLS looks at residual, wild and pairwise bootstrap under linear regression (they are the same for nonlinear regressions)
  Results displayed on "OLS.JPG"

Overall Bootstrap performs just as well as the estimators using asymptotic theory 
The reason why bootstrap didn't perform better than asymptotic theory is because of information matrix. The estimators are the best given the specifications. But it is very surprising that bootstrap can perform just as well as asymptotic estimators. 

Plans for the future: 
  1) Time series 
    AR(p) bootstrap, (-m+n residuals)
    Nonparametric Moving Block (Circular and Stationary) 
  2) Order statistics (non regular case (m-over-n bootstrap))
  3) Subsampling (not really bootstrap but a good comparison) 
