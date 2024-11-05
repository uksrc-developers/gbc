## This branch is being used to adapt the code to HP

Currently inputs radio and gaussian information. 
E.g python script_name HP_number (e.g. 333)

## Gradient Boosting Classifier for LoTSS 

This repository contains part of the code used in the [MULTIWAVE demonstrator case](https://confluence.skatelescope.org/display/SRCSC/MULTIWAVE). 

**Description**: Predict which sources of LoTSS can be cross-matched by LR or require visual analysis using small datasets based on healpixs.

**Paper reference**:  https://doi.org/10.1093/mnras/stac1888 

### Code aims:

This code aims to predict which sources from the LOFAR Two-Metre Sky Survey (LoTSS) can be automatically cross-matched with optical and infrared sources using the Likelihood Ratio (LR) technique or otherwise require visual analysis. A Gradient Boosting Classifier (GBC) was trained to assist this process. 

Cross-matching using the LR technique involves assessing the probability that an optical source is a true counterpart of a radio source. The LR method is successful for small and compact sources but often fails with extended, multi-component, or blended sources.  

The model is trained on LoTSS DR1 data, and includes features based on PyBDSF sources and Gaussians. PyBDSF is the source detection tool used in LoTSS. Each PyBDSF source can consist of one or more Gaussians. The features used consist of source size and flux density, the number of Gaussians for each PyBDSF source and LR Information for the source and its nearest neighbors. 

The output of the code is a binary classification: 
* Cross-match by LR (automatic cross-matching).
* Manual inspection needed (requires visual analysis or further methods). 

To be used by its own, the threshold value (tlv) for binary decisions sugested is 0.20. This was optimised to adapt the model to the imbalanced datasets of LoTSS using LoTSS DR1. This threshold separates sources that can be cross-matched by LR from those requiring human inspection. The code also accounts for the varying LR threshold values across different areas of the sky used to specific regions of the survey. 

### Usage 

E.g. Input data (LoTSS DR2): 

a) The original full catalogues for sources and Gaussians can be found at https://lofar-surveys.org/dr2_release.html: 
* LoTSS_DR2_v110_masked.srl.fits 
* LoTSS_DR2_v110.gaus.fits 

b) LR for sources and Gaussians can be found internally on Herts cluster: 
* LoTSS_DR2_v100.gaus_13h.lr-full.fits 
* LoTSS_DR2_v100.srl_13h.lr-full.sorted_step3_flux4.fits 

The code requires the LR thresholds values used for LoTSS DR2: 
* 13h 
  * LR_thresh = 0.309 (n, dec>=32.375) 
  * LR_thresh = 0.328 (s, dec<=32.375) 
* 0h 
  * LR_thresh = 0.394 

The code outputs: 
* features.fits - list of features used to train the model which are necessary to output the predictions
* pred_thresholds.csv - predictions for different thresholds (we used tlv=0.20 for DR1)


### How to run the rode

1. Set the environment (for example in conda):

` conda env create -f environment.yml `

2. Download the model. Currently download from [here](https://github.com/laraalegre/LOFARMachineLearningClassifier/tree/main/models).

3. Define the paths for the necessary directories (data, project directory, models, results) in the JSON file provided.

4. Run the script:

` python build_features_and_make_predictions.py <healpix number> `

The [healpix number](LOFAR/DR2/healpix_batch.py) corresponds the area of the sky to be processed. 

This will create an interim folder where the features will be stored and a results folder with the predictions.

Notes: 

* This code is adapted to input healpix files, so the original catalogues need to be split into healpixs for batch processing before running the code.
* The tlv information is contained in the JSON file. This will be different for different healpixs - the values need to be defined before running the code.


### Additional Notes 

* The code was initially trained on LoTSS DR1 and can also be applied to LoTSS DR2 or future LoTSS data releases that follow as long as the same PyBDSF and LR methods are used. 
* Adjustments are required for different regions of the sky, as LR thresholds vary across datasets.
* Applying the code to different dataset distributions may require adapting the threshold value (tlv) of 0.20.
