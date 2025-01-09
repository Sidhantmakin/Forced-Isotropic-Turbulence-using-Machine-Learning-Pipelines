# Forced-Isotropic-Turbulence-using-Machine-Learning-Pipelines

Objective: 
The main objective of this project is to evaluate and model forced isotropic turbulence data using machine learning (ML) pipelines. There are three main characteristics of the dataset:
•	Time (time): The turbulence simulation's time series.
•	Energy (energy): Turbulent kinetic energy is a measurement of the energy present in turbulence.
•	Taylor Microscale Reynolds Number (Re_lambda): The strength of turbulence is described by the dimensionless Taylor Microscale Reynolds Number (Re_lambda).

The aim is to predict Reynolds number (Re_lambda) based on the given time and energy features using various ML techniques. 

Dataset Description: 
The dataset used in this project contains the following columns:
•	time: A continuous numerical feature representing the simulation time.
•	energy: Turbulent kinetic energy, a continuous feature.
•	Re_lambda: Taylor microscale Reynolds number, the target variable for prediction.
Data source: https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic

The data is organized as a time series and comes from forced isotropic turbulence simulations 

Challenges Addressed:
This study addressed several issues pertaining to forced isotropic turbulence modelling and data preparation. A crucial component was feature selection, and several methods were attempted to precisely determine the most significant predictors, including mutual information regression, recursive feature elimination (RFE), correlation matrices, and random forest feature importance. To guarantee appropriate model convergence and performance, `StandardScaler` was used to handle the numerical range of characteristics that needed standardization. Furthermore, because the objective was to estimate Reynolds number trends rather than carry out direct time-series forecasting, the time-based character of the dataset made it difficult to strike a balance between temporal information and feature selection. The goal of these actions was to increase the analysis's machine learning models' accuracy and resilience.
