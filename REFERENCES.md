# References

Methods used in this project and where each is applied.

- Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill. Alpha = IC × volatility × score; measured information coefficients (`portfolio_construction.py`, `scoring/scoreboard.py`).
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28-43. View blending (`portfolio_construction.py`).
- He, G. & Litterman, R. (1999). The Intuition Behind Black-Litterman Model Portfolios. Goldman Sachs. Tau and view-uncertainty convention.
- Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model. Confidence-proportional view uncertainty.
- Ledoit, O. & Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. Covariance shrinkage (pods and firm VaR).
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th ed. EWMA volatility (λ = 0.94), 95% VaR convention (`_market_math.py`, managers).
- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and Error Correction. *Econometrica*, 55(2), 251-276. Pair screening (`_pairs_screening.py`).
- Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies*, 19(3), 797-827. Entry/exit convention, formation window (`stat_arb_pod`).
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357-384. Two-state regime model (`regime_hmm_analyst.py`).
- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day. Order identification cadence (`arima_forecast_analyst.py`).
- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica*, 59(2), 347-370. EGARCH.
- Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779-1801. GJR-GARCH.
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. *Review of Financial Studies*, 22(11), 4463-4492. Variance-risk-premium signal (`heston_vol_surface_analyst.py`).
- Fama, E. F. & French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33(1), 3-56. Factor loadings.
- Fama, E. F. & MacBeth, J. D. (1973). Risk, Return, and Equilibrium: Empirical Tests. *Journal of Political Economy*, 81(3), 607-636. Cross-sectional premia.
- Ehsani, S. & Linnainmaa, J. T. (2022). Factor Momentum and the Momentum Factor. *Journal of Finance*, 77(3), 1877-1919. 12-1 factor-momentum premium blend.
- Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703-708. HAC t-statistics.
- Kelly, J. L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal*, 35(4), 917-926. Fractional-Kelly overlay.
- Sharpe, W. F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*, 21(1), 49-58. Performance metrics and trailing-Sharpe weighting.
- Freund, Y. & Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. *Journal of Computer and System Sciences*, 55(1), 119-139. The Hedge weighting strategy (`scoring/weighting.py`).
