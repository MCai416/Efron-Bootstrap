Resampling Methods by the Bootstrap 

1)Efron-Naive looks at percentile bootstrap and t-stat bootstrap 
  Result: small sample perform poorly, mainly due to the samples drawn. Cannot perform better than the sample estimate 
  Results displayed on "Naive Average.JPG"
  
2)Efron OLS looks at residual, wild and pairwise bootstrap under linear regression (they are the same for nonlinear regressions)
  Setuup: heteroskedastic 
  Result: performs exceptionally well under small sample, and consistently having a smaller confidence interval than the sample bootstrap,   approaches to the true value faster
  Results displayed on "OLS.JPG"

Plans for the future: 
  1) Time series 
    AR(p) bootstrap, (-m+n residuals)
    Nonparametric Moving Block (Circular and Stationary) 
  2) Order statistics (non regular case (m-over-n bootstrap))
  3) Subsampling (not really bootstrap but a good comparison) 
