# Generating Probabilistic Atmospheric River Forecasts from Deterministic NWP Systems With Neural Networks and Convolutional Neural Networks
## Forecasts are from 0-5 Days Lead Time

This repository provides Python accompanying the paper

Post arxiv link here (when available). 

Code for processing and implementation All Neural Network models is available.

This study demonstrates how neural networks can be used for post-processing of deterministic weather forecasts for a distributional regression framework. 


### ABSTRACT (currently in flux)
Dynamical ensemble predictions are computationally expensive to create. Deep Learning (DL) post-processing methods are examined to obtain reliable and accurate probabilistic forecasts from deterministic numerical weather prediction. Using a 34-year reforecast of North American West Coast integrated vapor transport (IVT) as a case study, the statistically derived 0-120 hour probabilistic forecasts for IVT under atmospheric river (AR) conditions are tested. These predictions are compared to the Global Ensemble Forecast System (GEFS) dynamic model and the GEFS calibrated with a neural network. Additionally, the DL methods are tested against an established, but more rigid, statistical-dynamical ensemble method (the Analog Ensemble). The findings show, using continuous ranked probability skill score and Brier skill score as verification metrics, that the DL methods compete with or beat the calibrated GEFS system at lead times from 0-48 hours and again from 72-120 hours for AR vapor transport events. Additionally, the DL methods create well calibrated and reliable probabilistic forecasts. Lastly, the implications of varying the length of the training dataset are examined and show that the DL methods learn relatively quickly and ~10 years of hindcast data are required compete with the GEFS ensemble.

