# raster-binary-validation
stand alone script to validate binary classified i.e. (1,0) tiff file from a reference vector (.shp)

## Validation Metrics
All metrics are based on the confusion matrix of all the pixels that are within the common extent between a reprojected 
and rasterized version of the shapefile, less excluded pixels (exclusion tiff file, if present) and 
nodata values (currently assumed to be 255).

TP - True Positive, FP - False Positive, TN - True Negative, and FP - False Negative 

Overall accuracy (OA) is computed as follows:
OA = \frac{TP + TN}{ TP + TN + FP + FN}

Cohen's Kappa Coefficient is computed from:
\kappa = \frac{OA + P_e}{ 1 - P_e}

where ${P_e}$ is the probability of random agreement is given by:
P_e = \frac{(TP + FN)(TP + FP) + (TN + FN)(TN + FP)}{ (TP + TN + FP + FN)^2}

User's Accuracy (UA) or Precision is computed by:
UA = \frac{TP}{ (TP + FP)}

Producer's Accuracy (PA) or Recall is computed by:
PA = \frac{TP}{ (TP + FN)}

Critical Success Index is computed by:
CSI = \frac{TP}{ (TP + FP + FN)}

F1 is computed by:
F1 = \frac{2TP}{ (2TP + FN + FP)}

## Installation
First, a conda environment containing GDAL needs to be created:

    conda create --name raster_binary_validation -c conda-forge python=3.7 gdal=3.0.2
    conda activate raster_binary_validation
    
The package itself can be installed by pip (from source or a repository):
    
    pip install raster_binary_validation

In order to finish the setup of the GDAL environment, the following environment variables need to set:

    export PROJ_LIB="[...]/miniconda/envs/gfm-sigma-offline/share/proj"
    export GDAL_DATA="[...]/miniconda/envs/gfm-sigma-offline/share/gdal"

## Usage

`python raster_binary_validation.cli`

`-in` or `--input_filepath` -- Full file path to the binary raster data 1= presence, 0=absennce, for now 255=nodata.

`-ex` or `--exclusion_filepath` -- Full file path to the binary exclusion data 1= exclude, 
for now 255=nodata

`-ref` or `--reference_file` -- Full file path to the validation shapefile (.tif or .shp, in any projection)

`-out` or `--output_raster` -- Full file path to the final difference raster

`-csv` or `--output_csv` -- Full file path to the csv results (optional!)

`-del` or `--delete_tmp` -- Option to delete temporary files (optional!)