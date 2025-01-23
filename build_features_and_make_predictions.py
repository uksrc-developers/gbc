# Build features for LoTSS DR2 using HEALPix based on:

# * PyBDSF radio properties 
# * PyBDSF radio gaussian properties 
# * Likelihood ratios 

# Libraries 
import pandas as pd
import numpy as np
from astropy.table import Table
import os
import sys
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
from astropy import units as u
from astropy.io import fits
from joblib import load
import yaml

print("")

#--------------------------  Define paths and get variables  -------------------------- 
# Get the healpix value from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python script.py <healpix>")
    sys.exit(1)

healpix = int(sys.argv[1])
print(f"The healpix number we are about to work on is: {healpix}")

# Load the YAML configuration file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_file = 'config.yaml'
config = load_config(config_file)
    
# Access paths
paths = config.get('catalogue_paths', {})
base_path = paths.get('base_path')
data_path = paths.get('data_path')
models_path= paths.get('models_path')

# Access constants
#constants = config.get('constants', {})
#tlv_lr = constants.get('tlv_lr')
#healpix = constants.get('healpix')
tlv_lr_values = config.get('tlv_lr', {})

# Creating a function to read the fits catalogues
def read_fits(file):
    'converts a fits file to an astropy table'
    data_file = os.path.join(data_path, "hp_"+str(healpix), file)
    with fits.open(data_file) as cat:
        table = Table(cat[1].data)
        return table.to_pandas()

#--------------------------   Use Healpix catalogues  --------------------------   
pybdsf_radio = read_fits("radio_"+str(healpix)+".fits") # 'radio_333.fits' 
pybdsf_radio_nn = read_fits("radio_nn_"+str(healpix)+".fits") # 'radio_nn_333.fits'
pybdsf_lr = read_fits("radio_lr_"+str(healpix)+".fits") # 'radio_lr_33.fits' 
pybdsf_nn_lr = read_fits("radio_nn_lr_"+str(healpix)+".fits") #'radio_nn_lr_333.fits'

gauss_radio = read_fits("gaussian_"+str(healpix)+".fits") # 'gaussian_333.fits'
#gauss_radio_nn = read_fits("gaussian_nn_"+str(healpix)+".fits") # ''gaussian_nn_333.fits' 
gauss_lr =  read_fits("gaussian_lr_"+str(healpix)+".fits") # "gaussian_lr_333.fits'
#gauss_nn_lr = read_fits("gaussian_nn_lr_"+str(healpix)+".fits") #gaussian_nn_lr_333.fits

#--------------------------   Define which tlv lr use --------------------------   
#13 hour field
cond_13h_min = (pybdsf_radio['RA']/15) > (13-6)
cond_13h_max = (pybdsf_radio['RA']/15) < (13+6)
cond_dec_n = pybdsf_radio['DEC'] >= 32.375
cond_dec_s = pybdsf_radio['DEC'] < 32.375
#1H
cond_1h_min = (pybdsf_radio['RA']/15) > 21
cond_1h_max = (pybdsf_radio['RA']/15) < 3
cond_1_0_max = (pybdsf_radio['RA']/15) < 0
cond_1_0_min = (pybdsf_radio['RA']/15) > 0

print("min", (pybdsf_radio['RA']/15).min())
print("max", (pybdsf_radio['RA']/15).max())
n13h = cond_13h_min & cond_13h_max & cond_dec_n 
s13h = cond_13h_min & cond_13h_max & cond_dec_s
x1h = (cond_1h_min & cond_1_0_max) | (cond_1_0_min & cond_1h_max)


# Detect if more than one condition is True
if sum([(n13h == True).all(), (s13h == True).all(), (x1h == True).all()]) > 1:
	print("Error: The healpix ovelaps areas with different thresholds. Aborting test.")
	exit(1)

# Assign the `tlv_lr`
if (n13h == True).all():
    print("Field is n13h")
    tlv_lr = tlv_lr_values.get("n13h")
elif (s13h == True).all():
    print("Field is s13h")
    tlv_lr = tlv_lr_values.get("s13h")
elif (x1h == True).all():
    print("Field is 1h")
    tlv_lr = tlv_lr_values.get("1h")
else:
    print("Error: No valid condition is True. Something is wrong. Aborting script.")
    exit(1)  # Exit with an error code

#--------------------------  Replace 1e20 values --------------------------  

# 1e20 values will be replaced because the script was adapted take lr and lr_dist with NANs and not values of 1e20.
print("There are", len(gauss_lr[gauss_lr['lr']==1e20]) + len(pybdsf_lr[pybdsf_lr['lr']==1e20]) + len(pybdsf_nn_lr[pybdsf_nn_lr['lr']==1e20]), "1e20 values.")

# Replace 1e20 with NaNs in the 'lr' column if there are any! 
gauss_lr['lr'].replace(1e20, np.nan, inplace=True)
pybdsf_lr['lr'].replace(1e20, np.nan, inplace=True)
pybdsf_nn_lr['lr'].replace(1e20, np.nan, inplace=True)

# Replace 1e20 with NaNs in the 'lr_dist' column
gauss_lr['lr_dist'].replace(1e20, np.nan, inplace=True)
pybdsf_lr['lr_dist'].replace(1e20, np.nan, inplace=True)
pybdsf_nn_lr['lr_dist'].replace(1e20, np.nan, inplace=True)

print("Number of 1e20 values after replacing:", len(gauss_lr[gauss_lr['lr']==1e20]) + len(pybdsf_lr[pybdsf_lr['lr']==1e20]) + len(pybdsf_nn_lr[pybdsf_nn_lr['lr']==1e20]))
## To run on the 9 hp assume:
#pybdsf_radio = pybdsf_radio_nn
#pybdsf_lr = pybdsf_nn_lr
#gauss_radio = gauss_radio_nn
#gauss_lr = gauss_nn_lr


#--------------------------  Define necessary columns from catalogues --------------------------  

# Define the columns required from the PyBDSF radio catalogue and the likelihood ratio (LR) catalogue.
# Columns extracted from the radio catalogue:
# - Source_Name: Unique identifier for the PyBDSF source
# - RA, DEC: Right Ascension and Declination of the source
# - Peak_flux, Total_flux: Peak and total flux measurements
# - Maj, Min: Major and minor axis measurements
# - DC_Maj, DC_Min: Deconvolved major and minor axis measurements
# - S_Code: Source classification code
# - Mosaic_ID: Identifier for the mosaic where the source is located
pybdsf_radio_cols = ['Source_Name', 'RA', 'DEC', 'Peak_flux', 'Total_flux', 'Maj', 'Min', 'DC_Maj', 'DC_Min', 'S_Code', 'Mosaic_ID']

# Define columns for PyBDSF Gaussians, including an additional identifier ('Gaus_id')
gauss_radio_cols = pybdsf_radio_cols + ['Gaus_id']

# Define the columns required from the LR catalogue:
# - Source_Name: Unique identifier for the PyBDSF source
# - lr: Likelihood ratio value
# - lr_dist: Distance measurement associated with the LR match
pybdsf_lr_cols = ['Source_Name','lr', 'lr_dist']
gauss_lr_cols = pybdsf_lr_cols + ['Gaus_id']

# Combine the PyBDSF radio information with the LR information into single catalogues
# For the PyBDSF sources 
pybdsf_classes = pd.merge(pybdsf_radio[pybdsf_radio_cols], pybdsf_lr[pybdsf_lr_cols], on='Source_Name')
# And for the Gaussians 
gauss = pd.merge(gauss_radio[gauss_radio_cols], gauss_lr[gauss_lr_cols], on=['Source_Name', 'Gaus_id'])
# Add also information about the nearest neighbours healpixs for the PyBDSF sources
pybdsf_nn_classes = pd.merge(pybdsf_radio_nn[pybdsf_radio_cols], pybdsf_nn_lr[pybdsf_lr_cols], on='Source_Name')


# -------------------------------- CALCULATE FEATURES -------------------------------

print("Starting to process healpix", healpix)
print("Number of pybdsf sources is:", len(pybdsf_radio))

# Gaussian properties
# Number of Gaussians
# Taking the number of gaussians that make up each pybdsf source (and the log)
gauss_nr = pd.DataFrame(
    {'Source_Name': gauss['Source_Name'].value_counts().index,
     'n_gauss': gauss['Source_Name'].value_counts().values, 
     'log_n_gauss': np.log10(gauss['Source_Name'].value_counts().values)
    })

# Gaussians' LR, Gaussians' LR distance & Gaussians' LR major axis

# First combine the Gaussian catalogue with the Gaussian LR catalogue to obtain some additional information like the major axis of the radio emission
print('Computing Gaussian information for the Gaussian with the highest LR value or the brighest Gaussian if LR below LoTSS-DR1 LR threshold.')
# Select maximum LR Gaussian
gauss_max_lr_aux = gauss[['Source_Name', 'lr']].groupby(['Source_Name']).max().reset_index()
# Remove NaNs
gauss_max_lr = gauss_max_lr_aux[gauss_max_lr_aux['lr'].notnull()]
# Select brightest Gaussian
gauss_max_flux_aux = gauss[['Source_Name', 'Total_flux']].groupby(['Source_Name']).max().reset_index()
# Select the ones that do not have LR info 
gauss_max_flux = gauss_max_flux_aux[gauss_max_flux_aux['Source_Name'].
                   isin(gauss_max_lr_aux[gauss_max_lr_aux['lr'].isnull()]['Source_Name'])]
# Get the Gaussians with the maximum LR
gauss_max_lr_info = gauss.set_index(['Source_Name', 'lr']).loc[gauss_max_lr[['Source_Name', 'lr']].apply(tuple, axis=1)].reset_index()
# Get the Gaussians with the maximum flux from the main catalogue
gauss_max_flux_info = gauss.set_index(['Source_Name', 'Total_flux']).loc[gauss_max_flux[['Source_Name', 'Total_flux']].apply(tuple, axis=1)].reset_index()
# Concatenate the 2 tables 
gauss_max_join = pd.concat([gauss_max_lr_info, gauss_max_flux_info])

# Create the output table
gauss_max_info = gauss_max_join[['Source_Name', 'lr', 'lr_dist', 'DC_Maj', 'Maj', 'DC_Min', 'Min', 'Total_flux','Gaus_id']].rename(
    columns={'lr': 'gauss_lr',  'lr_dist': 'gauss_lr_dist', 
             'DC_Maj': 'gauss_dc_maj', 'Maj': 'gauss_maj',
             'DC_Min': 'gauss_dc_min', 'Min': 'gauss_min',
             'Total_flux':'gauss_total_flux'
            })

# Include Major axis of the gaussian with the hightest LR
# Note: cannot drop duplicates here (This needs to be worked out before when computing the Gaussian values)
gauss_info = gauss_nr.merge(gauss_max_info, on='Source_Name', how='left')#.drop_duplicates(['Source_Name', 'Gaus_id'])

# PYBDSF PROPERTIES
# NEAREST NEIGHBOURS 
# Creating the coordinates to search for NN
pybdsf_coord = SkyCoord(pybdsf_classes['RA'], pybdsf_classes['DEC'], unit="deg")
pybdsf_nn_coord = SkyCoord(pybdsf_nn_classes['RA'], pybdsf_nn_classes['DEC'], unit="deg")
#  NN information
#nn_match = match_coordinates_sky(pybdsf_coord, pybdsf_coord,  nthneighbor=2)
nn_match = match_coordinates_sky(pybdsf_coord, pybdsf_nn_coord,  nthneighbor=2)

# 
nn_info = pd.DataFrame({'Source_Name':pybdsf_classes['Source_Name'],
                        'NN': pybdsf_nn_classes.iloc[nn_match[0]]['Source_Name'].values,
                        'NN_lr': pybdsf_nn_classes.iloc[nn_match[0]]['lr'].values,
                        'NN_lr_dist': pybdsf_nn_classes.iloc[nn_match[0]]['lr_dist'].values,
                        'NN_dist': nn_match[1].arcsec,
                        'NN_flux_ratio': pybdsf_nn_classes.iloc[nn_match[0]]['Total_flux'].values/pybdsf_classes['Total_flux']})

#Old way of computing without taking the nn into consideration
#nn_info = pd.DataFrame({'Source_Name':pybdsf_classes['Source_Name'],
#                        'NN': pybdsf_classes.iloc[nn_match[0]]['Source_Name'].values,
#                       'NN_lr': pybdsf_classes.iloc[nn_match[0]]['lr'].values,
#                        'NN_lr_dist': pybdsf_classes.iloc[nn_match[0]]['lr_dist'].values,
#                        'NN_dist': nn_match[1].arcsec,
#                        'NN_flux_ratio': pybdsf_classes.iloc[nn_match[0]]['Total_flux'].values/pybdsf_classes['Total_flux']})

# Number of NN within 45''
print('Computing the number of sources within 45".')
#idx1, idx2, dist2d, dist3d = pybdsf_coord.search_around_sky(pybdsf_coord, 45*u.arcsec)
idx1, idx2, dist2d, dist3d = search_around_sky(pybdsf_coord, pybdsf_nn_coord, 45*u.arcsec) # NEW
idxs, counts_45 = np.unique(idx1, return_counts=True) # includes counting itself

nn_counts_45 = pd.DataFrame({'Source_Name': pybdsf_classes.iloc[idxs]['Source_Name'],
                             'NN_45':counts_45-1})
# Combine extra information on a cat 
nn_info = nn_info.merge(nn_counts_45, on= 'Source_Name')

np.testing.assert_equal(len(nn_info), len(pybdsf_classes), err_msg="Number of pybdsf does not match.", verbose=True)

# Merging all the catalogues
output = gauss_info.merge(nn_info, on='Source_Name').merge(pybdsf_classes, on='Source_Name')

# %%
# Changing LR information
# Create an aux column with 0 (no lr info) and 1 (lr info)
# For the pybdsf lr 
output.loc[output['lr'].isna(),'lr_info'] = int(0)
output.loc[output['lr_info'].isna(),'lr_info'] = int(1)
# For the gauss lr 
output.loc[output['gauss_lr'].isna(),'gauss_lr_info'] = int(0)
output.loc[output['gauss_lr_info'].isna(),'gauss_lr_info'] = int(1)
# For the NN lr 
output.loc[output['NN_lr'].isna(),'NN_lr_info'] = int(0)
output.loc[output['NN_lr_info'].isna(),'NN_lr_info'] = int(1)
# if lr empty, set to 1e-135 (minimum lr value limit)
# For the pybdsf lr 
output.loc[output['lr'].isna(),'lr'] = float(10**-135)
# For the gauss lr 
output.loc[output['gauss_lr'].isna(),'gauss_lr'] = float(10**-135)
# For the NN lr 
output.loc[output['NN_lr'].isna(),'NN_lr'] = float(10**-135)
# if lr dist empty set to 20'', to distinguish from the limit of 15''
# For the pybdsf lr 
output.loc[output['lr_dist'].isna(),'lr_dist'] = float(20)
# For the gauss lr 
output.loc[output['gauss_lr_dist'].isna(),'gauss_lr_dist'] = float(20)
# For the NN lr 
output.loc[output['NN_lr_dist'].isna(),'NN_lr_dist'] = float(20)
# If gauss axis are NaN set to zero
gauss_nan_col = ['gauss_dc_maj', 'gauss_maj','gauss_dc_min', 'gauss_min']
for i in gauss_nan_col:
    output.loc[output[i].isna(), i] = int(0)

# CREATING A COLUMN TO TAKE THE HIGHEST LR
# Compares the lr value of the gauss and the pybdsf and takes the max value
output['highest_lr'] = output[['lr', 'gauss_lr']].max(axis=1)

# Changing the value of the LR take into account LR of the threshold
tlv_lr_dr1 = 0.639 # Lotss dr1 threshold limit value

# Taking lotss dr1 tlv into consideration to scale 
# For the pybdsf lr 
output['lr_tlv'] = output['lr']*tlv_lr_dr1/tlv_lr
output['log_lr_tlv'] = np.log(output['lr']*tlv_lr_dr1/tlv_lr)
# For the gauss lr 
output['gauss_lr_tlv'] = output['gauss_lr']*tlv_lr_dr1/tlv_lr
output['log_gauss_lr_tlv'] = np.log(output['gauss_lr']*tlv_lr_dr1/tlv_lr)
# For the NN lr 
output['NN_lr_tlv'] = output['NN_lr']*tlv_lr_dr1/tlv_lr
output['log_NN_lr_tlv'] = np.log(output['NN_lr']*tlv_lr_dr1/tlv_lr)
# For the hightest lr
output['highest_lr_tlv'] = output['highest_lr']*tlv_lr_dr1/tlv_lr
output['log_highest_lr_tlv'] = np.log(output['highest_lr']*tlv_lr_dr1/tlv_lr)

# Computing the flux ratio for the gaussians
print('Computing additional information for the Gaussians.')
output['gauss_flux_ratio'] = output['gauss_total_flux']/output['Total_flux']

# EXPORTING THE RESULTS
# Reordering the columns
# Columns with NaN values 
for i in output.columns:
    a = output[output[i].isna()]
    if len(a)!= 0: 
        print(i) 

features = output[['Maj','Min','DC_Maj','DC_Min','Total_flux','Peak_flux','n_gauss','log_n_gauss', # PyBDSF info
                   'lr','lr_tlv','log_lr_tlv','lr_dist', # PyBDSF LR
                   'NN_dist','NN_45','NN_flux_ratio', # NN info
                   'NN_lr','NN_lr_tlv','log_NN_lr_tlv','NN_lr_dist', # NN LR 
                   'gauss_maj','gauss_min','gauss_dc_maj', 'gauss_dc_min','gauss_total_flux','gauss_flux_ratio', # Gauss info
                   'gauss_lr','gauss_lr_tlv','log_gauss_lr_tlv','gauss_lr_dist', # Gauss LR
                   'highest_lr','highest_lr_tlv','log_highest_lr_tlv', # Highest LR between the gauss and the PyBDSF source
                  ]]

other_info = output[['Source_Name', 'RA', 'DEC', # Source RA and DEC
                     'lr_info', #Values defined by us or not
                     'NN','NN_lr_info', # ID of the nearest neighbour
                     'Gaus_id', 'gauss_lr_info',
                     'Mosaic_ID', 'S_Code' # PyBDSF source mosaic ID and code
                    ]] 

output.columns[(~output.columns.isin(features)) &
               (~output.columns.isin(other_info))]

total = [other_info, features]
output_df = pd.concat(total, axis = 1)

# Sorting by Source Name
output_df_sorted = output_df.sort_values(by=["Source_Name"], ignore_index = True)

file_name = "TESTING_features_"+str(healpix)+".fits"
output_cat = Table.from_pandas(output)
output_cat.write(os.path.join(data_path, "hp_"+str(healpix), file_name), overwrite=True)
print("Wrote features to file:", file_name)

# ------------------------------------ PREDICTIONS -------------------------------

# Use the table that was created in the step before
with fits.open(os.path.join(data_path, "hp_"+str(healpix), file_name)) as cat:
    master = Table(cat[1].data).to_pandas()


# Select model
model = load(os.path.join(models_path, 'GradientBoostingClassifier_A1_31504_18F_TT1234_B1_exp3.joblib'))

print("model loaded")

features_pred = ['Maj','Min','Total_flux','Peak_flux','log_n_gauss', # PyBDSF info
                'log_lr_tlv', 'lr_dist', # PyBDSF LR
                'NN_dist','NN_45','NN_flux_ratio', # NN info
                'log_NN_lr_tlv','NN_lr_dist', # NN LR 
                'gauss_maj','gauss_min', 'gauss_flux_ratio', # Gauss info
                'log_gauss_lr_tlv','gauss_lr_dist', # Gauss LR
                'log_highest_lr_tlv', # Highest LR between the gauss and the PyBDSF source
                  ]

# Defining the thresholds 
thresholds = []
i = np.linspace(0,1,101)
for i in i:
    thresholds.append(format(float(i), '.2f'))

def make_thresholds(data, threshold):
    data[threshold] = np.where(data['probability_lr'] >= float(threshold), 1, 0)
    return data

# Getting the probabilities of being class 0 (LGZ) or class 1 (LR)
pred_proba = model.predict_proba(master[features_pred])
# Taking the probability of being accepted by LR
pred_proba_1 = pred_proba[:,1]
# Making it a pandas table 
predictions = pd.DataFrame({'Source_Name': master['Source_Name'],
                            'probability_lr': pred_proba_1})
# Make the predictions into class 0 or 1 for different thresholds
for i in thresholds:  
    make_thresholds(predictions, threshold = i)

# Export predictions 
# Export as fits
pred_name = "TESTING_pred_threshods_" + str(healpix) + ".fits"
pred_cat = Table.from_pandas(predictions)
pred_cat.write(os.path.join(data_path, "hp_"+str(healpix),  pred_name), overwrite=True)
print("Wrote the predictions to file:", pred_name)




