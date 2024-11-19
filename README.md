# Gradient Boosting Classifier for LoTSS 

This repository contains part of the code used in the [MULTIWAVE demonstrator case](https://confluence.skatelescope.org/display/SRCSC/MULTIWAVE). 

**Description**: Predict which sources of LoTSS can be cross-matched by LR or require visual analysis using small datasets based on HEALPix.

**Paper reference**:  https://doi.org/10.1093/mnras/stac1888 

## Code aims

This code aims to predict which sources from the LOFAR Two-Metre Sky Survey (LoTSS) can be automatically cross-matched with optical and infrared sources using the Likelihood Ratio (LR) technique or otherwise require visual analysis. A Gradient Boosting Classifier (GBC) was trained to assist this process. 

Cross-matching using the LR technique involves assessing the probability that an optical source is a true counterpart of a radio source. The LR method is successful for small and compact sources but often fails with extended, multi-component, or blended sources.  

The model is trained on LoTSS DR1 data, and includes features based on PyBDSF sources and Gaussians. PyBDSF is the source detection tool used in LoTSS. Each PyBDSF source can consist of one or more Gaussians. The features used consist of source size and flux density, the number of Gaussians for each PyBDSF source and LR Information for the source and its nearest neighbors. 

The output of the code is a binary classification: 
* Cross-match by LR (automatic cross-matching).
* Manual inspection needed (requires visual analysis or further methods). 

To be used by its own, the threshold value (tlv) for binary decisions sugested is 0.20. This was optimised to adapt the model to the imbalanced datasets of LoTSS using LoTSS DR1. This threshold separates sources that can be cross-matched by LR from those requiring human inspection. The code also accounts for the varying LR threshold values across different areas of the sky used to specific regions of the survey. 

## Usage 

Download the [project folder](https://lofar-surveys.org/public/uksrc/project.zip) (untracked data and model) from the LOFAR-surveys website.

### Structure of the project

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

The code requires the LR thresholds values used for LoTSS DR2: 
* 13h 
  * LR_thresh = 0.309 (n, dec>=32.375) 
  * LR_thresh = 0.328 (s, dec<=32.375) 
* 0h 
  * LR_thresh = 0.394 

The code outputs: 
* features.fits - list of features used to train the model which are necessary to output the predictions
* pred_thresholds.csv - predictions for different thresholds (we used tlv=0.20 for DR1 and DR2)


### Environment
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


### Run the code

`python build_features_and_make_predictions.py <HEALPix number> <LR_thresh>`

Example: HEALPix 333 which is situated on the n13h

`python build_features_and_make_predictions.py 333 0.309`


### Notes 

* This code is adapted to input HEALPix files, so the original catalogues need to be split into HEALPix for batch processing before running the code.
* The code was initially trained on LoTSS DR1 and can also be applied to LoTSS DR2 or future LoTSS data releases that follow as long as the same PyBDSF and LR methods are used. 
* Adjustments are required for different regions of the sky, as LR thresholds vary across datasets.
* Applying the code to different dataset distributions may require adapting the threshold value (tlv) of 0.20.
