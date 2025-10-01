# Baseball Pitching
This is an exploratory project into pitching analytics.

## In progress

1. Pitch Locations -- create a simple KL Divergence or Energy score metric to compare pitchers in terms of their pitch location distributions. The goal is to first approximate the pitch location distribution (conditional on pitch type) using a non-parametric methods (normalizing flows) or parametric methods (Bivariate Gaussian Mixture Models). Given the approximate distributions, one can then compare pitchers on a distributional level. Project motivated by Baseball Savant's "Statcast Pitch Arsenal". 

## Ideas

1. Model Calibration ties physics and statistics through various inference and prediction tasks. This project will explore model calibration methods to estimate / verify the Induced Vertical Break metric. I will inlcude differential equations to estimate the break accouting for gravity. Calibration will be used to estimate the pitcher impact (i.e. the induced vertical break). 
