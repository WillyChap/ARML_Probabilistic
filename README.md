# Probabilistic Predictions from Deterministic Atmospheric River Forecasts with Deep Learning
## Forecasts are from 0-5 Days Lead Time

**Contact:** wchapman@ucsd.edu

**Authors:** William E. Chapman, Luca Delle Monache, Stefano Alessandrini, Aneesh C. Subramanian, F. Martin Ralph, Shang-Ping Xie , Sebastian Lerch, Negin Hayatbini

**Affiliations:** Scripps Institution of Oceanography, La Jolla, California; National Center for Atmospheric Research, Boulder, Colorado; University of Colorado Boulder, Boulder, Colorado; Institute for Stochastics, Karlsruhe Institute of Technology, Karlsruhe, Germany



This repository provides Python code to accompany the paper:

***Post arxiv link here (when available).*** 

Code use and file structure:

Code for processing and implementation all neural betwork models is available in /Coastal_Points/python_scripts and accompanying jupyter notebooks for verification are contained in /Coastal_Points/Generate_Figures. All utility files, which contain python functionality are in /Coastal_Points/python_scripts/utils*.py. The U-NET build file is /Coastal_Points/python_scripts/UNET_b.py.

This study demonstrates how neural networks can be used for post-processing of deterministic weather forecasts to create a distributional (i.e. probabilistic) regressional framework using a CRPS loss function. 

***Data Availability***:
--
**NOTE**: It took a massive amount of data to run/store results of this study, the authors are happy to provide keys to this data if you contact us. Additionally, the West-WRF model output will be made publically available very shortly from the Center for Western Weather and Water Extremes. Additionally, the intermediate files are in the process of being stored at the UCSD library archive, currently these are available upon request as they are too large to store on git. 

West-WRF simulations are archived at the Center for Western Weather and Water Extremes and on the National Center for Atmospheric Research servers are readily available upon request. GEFS data can be retrieved through the TIGGE archive (https://www.ecmwf.int/en/research/projects/tigge). MERRA2 data can be retrieved at https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/data_access/ .

#### Analog Ensemble
Fortran code for the Analog Ensemble was graciously provided by co-author Stefano Alessandrini. It is contained within the folder ./Coastal_Locations/AnEn. This model has a very particular structure and is run locally at every model point. We do not provide the code to run the model at every point, as the data is very large. This is available upon request. We have included one coastal location and operating script to assist in running this model those interested in using this model, it should provide a good template. 

### ABSTRACT (currently in flux)
Deep Learning (DL) post-processing methods are examined to obtain reliable and accurate probabilistic forecasts from single-member numerical weather predictions of integrated vapor transport (IVT). Using a 34-year reforecast of North American West Coast IVT, the dynamically/statistically derived 0-120 hour probabilistic forecasts for IVT under atmospheric river (AR) conditions are tested. These predictions are compared to the Global Ensemble Forecast System (GEFS) dynamic model and the GEFS calibrated with a neural network. Additionally, the DL methods are tested against an established, but more rigid, statistical-dynamical ensemble method (the Analog Ensemble). The findings show, using continuous ranked probability skill score and Brier skill score as verification metrics, that the DL methods compete with or outperform the calibrated GEFS system at lead times from 0-48 hours and again from 72-120 hours for AR vapor transport events. Additionally, the DL methods generate reliable and skillful probabilistic forecasts. The implications of varying the length of the training dataset are examined and show that the DL methods learn relatively quickly and ~10 years of hindcast data are required compete with the GEFS ensemble.

