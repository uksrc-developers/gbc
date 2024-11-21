# Gradient Boosting Classifier for LoTSS based on HEALPix

This repository contains the code relevant to running the GBC as part of [MULTIWAVE demonstrator case](https://confluence.skatelescope.org/display/SRCSC/MULTIWAVE). It is adapted to run on small datasets based on **HEALPix**.

## Introduction 

This code aims to predict which sources from the LOFAR Two-Metre Sky Survey (LoTSS) can be automatically cross-matched with optical and infrared sources using the Likelihood Ratio (LR) technique or otherwise require visual analysis. A Gradient Boosting Classifier (GBC) was trained to assist this process. 
Cross-matching using the LR technique involves assessing the probability that an optical source is a true counterpart of a radio source. The LR method is successful for small and compact sources but often fails with extended, multi-component, or blended sources.  
The model is trained on LoTSS DR1 data, and includes features based on PyBDSF sources and Gaussians. PyBDSF is the source detection tool used in LoTSS. Each PyBDSF source can consist of one or more Gaussians. The features used consist of source size and flux density, the number of Gaussians for each PyBDSF source and LR Information for the source and its nearest neighbors. 

**Paper reference**:  https://doi.org/10.1093/mnras/stac1888 

## Hardware and Software

###

The code runs on a CPU. For LoTSS DR2 the recommended hardware requirements were:
* 1 CPUs
* 16 GB RAM
* 10 GB hard disk space

### Set up an environment

It is important to have installed the following python packages versions:
```
joblib=1.4.2
scikit-learn=0.23.1 (ideally install 0.23.1 but also works with scikit-learn=0.23.2)
python=3.8.19 or python=3.8.18 
joblib=1.4.2
pandas=1.2.4
scipy=1.9.3
numpy=1.20.2 or numpy=1.20.3
astropy=4.2.1 or astropy=5.2
```

Set the environment (for example in conda):

`conda env create -f environment.yml `

Or install individual packages with conda:

`conda create -n gbc python=3.8.18 joblib=1.4.2 scipy=1.9.3 `

Some packages require conda-forge:

`conda install -c conda-forge scikit-learn=0.23.2 numpy=1.20.3 astropy=5.2.1 pandas=1.2.4 `


## Directory Structure

```
Project
│
├── README.md                       <- Overview and instructions
├── Licence                         <- Licence
├── environment.yml                 <- Python packages to create a conda environment
│
├── mw-gbc
|   └── build_features_and_make_predictions.py  <- Code to output the predictions
├── data (untracked)  
│   └── hp_outputs
│       ├── radio_333.fits          <- PyBDSF radio source catalogue for the central HEALPix (hp) 
│       ├── radio_333_nn.fits       <- PyBDSF radio source for central hp and nearest neighbours (nn)
│       ├── gaussian_333_nn.fit     <- PyBDSF Gaussian catalogue for the central hp and nn
│       ├── radio_333_lr.fits       <- LR source catalogue for the central hp
│       ├── radio_333_nn_lr.fits    <- LR ratio source catalogue for the central hp and nn
│       └── gaussian_333_nn_lr.fits <- LR Gaussian catalogue for the central hpealpix and nn
└── models (untracked)
    └── gbc
        └── GradientBoostingClassifier_A1_31504_18F_TT1234_B1_exp3.joblib <- model trained on LoTSS-DR1

```

## Inputs and Outputs

### Inputs 
* The code inputs 6 HEALpix files that are stored on /project/data/hp_outputs
  The HEALPix number is an arbitraty number that corresponds to a region of the sky. Please see [MW-HEALPix repository](https://github.com/uksrc-developers/MW-HEALPix) 
* A GBC model trained on LoTSS-DR1 that is stored on /project/models/gbc
* A LR thresholds (LR_thresh) value. The values used for LoTSS DR2 were:
  * 13h
    * LR_thresh = 0.309 (n, dec>=32.375) 
    * LR_thresh = 0.328 (s, dec<=32.375) 
  * 0h 
    * LR_thresh = 0.394

### Outputs
The output of the code is a binary classification: 
* Cross-match by LR (automatic cross-matching) - 1
* Manual inspection needed (requires visual analysis or further methods) - 0

The code creates 2 output files:
* features.fits - list of features used to train the model which are necessary to output the predictions
* pred_thresholds.csv - predictions for different thresholds (we used tlv=0.20 for DR1 and DR2)

To be used by its own, the threshold value (tlv) for binary decisions sugested is 0.20. This was optimised to adapt the model to the imbalanced datasets of LoTSS using LoTSS DR1. This threshold separates sources that can be cross-matched by LR from those requiring human inspection. 


## Running MW-GBC 

### Download the data and model

Download the [project folder](https://lofar-surveys.org/public/uksrc/project.zip) (untracked data and model) from the LOFAR-surveys website.

### Run the code
    
`python build_features_and_make_predictions.py <HEALPix number> <LR_thresh>`

Example: HEALPix 333 which is situated on the n13h

`python build_features_and_make_predictions.py 333 0.309`

## Future work

* The code is adapted to input 6 HEALPix files, but ideally would input 8 HEALPix files that include the central HP for the Gaussian catalogues as well:
    * gaussian_333.fits
    * gaussian_333_lr.fits
* The code was initially trained on LoTSS DR1 and applied to LoTSS DR2. It can be used for future LoTSS data releases that follow as long as the same PyBDSF and LR methods are used. Otherwise the model should be retrained.
* Applying the code to datasets with difference distributions between the 2 classes of sources may require adapting the threshold value (tlv) of 0.20.



## External links

The relevant Jira tickets are as follows:

* [Splitting the GBC input files into the separate sky areas](https://jira.skatelescope.org/browse/TEAL-594)



## List of developers and Collaborators 
Alegre, L.

