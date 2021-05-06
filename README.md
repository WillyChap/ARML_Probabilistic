# Generating Probabilistic Atmospheric River Forecasts from Deterministic NWP Systems With Neural Networks and Convolutional Neural Networks
## Forecasts are from 0-5 Days Lead Time

**Contact:** wchapman@ucsd.edu

**Authors:** William E. Chapman, Luca Delle Monache, Stefano Alessandrini, Aneesh C. Subramanian, F. Martin Ralph, Shang-Ping Xie , Sebastian Lerch, Negin Hayatbini

**Affiliations:** Scripps Institution of Oceanography, La Jolla, California; National Center for Atmospheric Research, Boulder, Colorado; University of Colorado Boulder, Boulder, Colorado; Institute for Stochastics, Karlsruhe Institute of Technology, Karlsruhe, Germany



This repository provides Python accompanying the paper

Post arxiv link here (when available). 

Code for processing and implementation All Neural Network models is available.

This study demonstrates how neural networks can be used for post-processing of deterministic weather forecasts to create a distributional (i.e. probabilistic) regressional framework using a CRPS loss function. 

***Data Availability***:
--
**NOTE**: It took a massive amount of data to run/store results of this study, the authors are happy to provide keys to this data if you contact us. Additionally, the West-WRF model output will be made publically available very shortly from the Center for Western Weather and Water Extremes. Additionally, the intermediate files are in the process of being stored at the UCSD library archive, currently these are available upon request as they are too large to store on git. 

West-WRF simulations are archived at the Center for Western Weather and Water Extremes and on the National Center for Atmospheric Research servers are readily available upon request. GEFS data can be retrieved through the TIGGE archive (https://www.ecmwf.int/en/research/projects/tigge). MERRA2 data can be retrieved at https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/data_access/.![image](https://user-images.githubusercontent.com/14932329/117337174-e6a3a380-ae51-11eb-8dd1-f0b209318984.png)


#### Analog Ensemble
Code for the Analog Ensemble was graciously provided by Stefano Alessandrini. It is contained within the folder ./Coastal_Locations/AnEn. This model has a very particular structure and we do not provide the code to run the model at every point, as the data is very large. This is available upon request. We have included one coastal location and operating script to assist in running this model (for future use). 

### ABSTRACT (currently in flux)
Dynamical ensemble predictions are computationally expensive to create. Deep Learning (DL) post-processing methods are examined to obtain reliable and accurate probabilistic forecasts from deterministic numerical weather prediction. Using a 34-year reforecast of North American West Coast integrated vapor transport (IVT) as a case study, the statistically derived 0-120 hour probabilistic forecasts for IVT under atmospheric river (AR) conditions are tested. These predictions are compared to the Global Ensemble Forecast System (GEFS) dynamic model and the GEFS calibrated with a neural network. Additionally, the DL methods are tested against an established, but more rigid, statistical-dynamical ensemble method (the Analog Ensemble). The findings show, using continuous ranked probability skill score and Brier skill score as verification metrics, that the DL methods compete with or beat the calibrated GEFS system at lead times from 0-48 hours and again from 72-120 hours for AR vapor transport events. Additionally, the DL methods create well calibrated and reliable probabilistic forecasts. Lastly, the implications of varying the length of the training dataset are examined and show that the DL methods learn relatively quickly and ~10 years of hindcast data are required compete with the GEFS ensemble.

