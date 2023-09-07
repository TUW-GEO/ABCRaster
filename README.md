# ABCRaster
ABCRaster stands for Accuracy assessment of Binary Classified Raster. It is a package for performing validation, 
accuracy assessment, or comparing classification results (.tif) versus a reference (.shp, .tif) e.g. 
[CEMS](https://emergency.copernicus.eu/emsdata.html). Can be used to assess other binary classification 
(presence/absence) maps. Computes accuracy assessment metrics e.g. User, Producer’s accuracy, Kappa, etc. Also creates 
‘confusion map’ with pixels marked as TP, TN, FP, and FN.

* reference shapefile can be in any projection (built-in reprojection and rasterization)
* (stratified) random sampling support (*based on reference file)
* applying raster or vector masks
* creates confusion (difference) tiff file 

## Installation
First, a conda environment containing GDAL needs to be created:

    conda create -n "abcraster" -c conda-forge python=3.8 mamba
    conda activate abcraster
    mamba install -c conda-forge python=3.8 gdal geopandas cartopy
    
The package itself can be installed by pip (from source or a repository):
    
    pip install abcraster

In order to finish the setup of the GDAL environment, the following environment variables need to set:

    export PROJ_LIB="[...]/miniconda/envs/abcraster/share/proj"
    export GDAL_DATA="[...]/miniconda/envs/abcraster/share/gdal"

** to get the path your conda envirment you can use `echo $CONDA_PREFIX` on Linux or  `echo %CONDA_PREFIX%` on Windows

## Usage

### Self-defined workflow
The `abcraster.base` module provides the `Validation` class, which carries the main functionality as 
dedicated methods. One can build a self-defined validation workflow by importing the class and calling
the needed methods. An example of a self-defined workflow is given here:
    
    # initialize validation object
    val = Validation(input_data_filepath=input_path, ref_data_filepath=ref_path, out_dirpath=out_dirpath)

    val.apply_mask(aoi_path, invert_mask=True)  # apply an area-of-interest (.tif or .shp)
    val.apply_maks(mask_path)  # apply a general mask (.tif or .shp)
    val.accuracy_assessment()  # calculate confusion matrix/map
    val.write_confusion_map(os.path.join(out_dirpath, 'val_diff.tif'))  # write confusion map to file
    print(val.calculate_accuracy_metric(critical_success_index))  # print the CSI value

The `calculate_accuracy_metric` method takes in all predefined functions of the `abcraster.metrics` module, 
but allows for self-written function as well. The function will receive a dictionary representing the confusion
matrix and cotnaining the values for the keys: 'TP', 'TN', 'FP' and 'FN'.

### Scripting
An already pre-defined workflow can be utilized in a Python script when using the `run` function of the 
`abcraster.base` module. An example of a call of the `run` function is given here:

    run(input_data_filepaths=[input_path], ref_data_filepath=ref_path, out_dirpath=out_dirpath, metrics_list=['CSI', 'OA'],
        samples_filepath=os.path.join(out_dirpath, 'sampling.tif'), sampling=(200, 200))

### Command line

The same pre-defined worklfow can be called through the command line by:

    python -m abcraster.base
    
Furhter details can be defined using the following arguments:

`-in` or `--input_filepath` -- Full file path to the binary raster data 1= presence, 0=absennce, for now 255=nodata.

`-ex` or `--exclusion_filepath` -- Full file path to the binary exclusion data 1=exclude, 
for now 255=nodata

`-ref` or `--reference_file` -- Full file path to the validation shapefile (.tif or .shp, in any projection)

`-out` or `--output_raster` -- Full file path to the final difference raster

`-csv` or `--output_csv` -- Full file path to the csv results (optional!)

`-del` or `--delete_tmp` -- Option to delete temporary files (optional!)

`-ns` or `--num_samples` -- Number of total samples if sampling will be applied (optional!)

`-stf` or `--stratify` -- Stratification flag (no input required) based on reference data (optional!)

`-nst` or `--no_stratify` -- No stratification flag option (optional!)

`-sfp` or `--samples_filepath` -- Full file path to the sampling raster dataset (.tif ), if num samples not specified, \
                        assumes samples will be read from this path (optional!)

`-all` or `--all_metrics` -- Flag to indicate to compute all metrics, Default true. (optional!)

`-na` or `--not_all_metrics` -- Flag to indicate not to compute all metrics, 
                        metrics should be specified if activated. (optional!)

`-mts` or  `--metrics` -- Optional list of metrics (keys) to run e.g. OA, UA, K. See metrics in `( )` above list.


## Accuracy Metrics
All metrics are based on the confusion matrix of all the pixels that are within the common extent between a reprojected 
and rasterized version of the shapefile, less excluded pixels (exclusion tiff file), if present and 
nodata values (currently assumed to be 255).

TP - True Positive, FP - False Positive, TN - True Negative, and FP - False Negative 

Overall accuracy (OA) is computed as follows:

![OA=\frac{TP+TN}{TP+TN+FP+FN}](https://latex.codecogs.com/svg.latex?OA=\frac{TP+TN}{TP+TN+FP+FN}) 


Cohen's Kappa Coefficient (K) is computed from:

![\kappa=\frac{OA+P_e}{1-P_e}](https://latex.codecogs.com/svg.latex?\kappa=\frac{OA+P_e}{1-P_e}) 

where ${P_e}$ is the probability of random agreement is given by:

![P_e=\frac{(TP+FN)(TP+FP)+(TN+FN)(TN+FP)}{(TP+TN+FP+FN)^2}](https://latex.codecogs.com/svg.latex?P_e=\frac{(TP+FN)(TP+FP)+(TN+FN)(TN+FP)}{(TP+TN+FP+FN)^2}) 


User's Accuracy (UA) or Precision is computed by:

![UA=\frac{TP}{(TP+FP)}](https://latex.codecogs.com/svg.latex?UA=\frac{TP}{(TP+FP)}) 

Producer's Accuracy (PA) or Recall is computed by:

![PA=\frac{TP}{(TP+FN)}](https://latex.codecogs.com/svg.latex?PA=\frac{TP}{(TP+FN)}) 

Critical Success Index (CSI) is computed by:

![CSI=\frac{TP}{(TP+FP+FN)}](https://latex.codecogs.com/svg.latex?CSI=\frac{TP}{(TP+FP+FN)}) 

F1 Score (F1) is computed by:

![F1=\frac{2TP}{(2TP+FN+FP)}](https://latex.codecogs.com/svg.latex?F1=\frac{2TP}{(2TP+FN+FP)}) 

Penalization function is computed by:

![P=exp\left(\frac{FP}{(TP+FN)/ln(1/2)}\right)](https://latex.codecogs.com/svg.latex?P=exp\left(\frac{FP}{(TP+FN)/ln(1/2)}\right))              

Success Rate (SR) is computed by:

![SR=PA-(1-P)](https://latex.codecogs.com/svg.latex?SR=PA-(1-P)) 

Bias (B) is computed by:

![b=(TP+FP)/(TP+FN)](https://latex.codecogs.com/svg.latex?b=(TP+FP)/(TP+FN))
 
Prevalence (P) is computed by:

![Pre=(TP+FN)/(TP+FN+TN+FP)](https://latex.codecogs.com/svg.latex?Pre=(TP+FN)/(TP+FN+TN+FP))

## Sampling
Module added for random and stratified sampling methods. Sampling module includes stand-alone CLI for creating raster 
encoded samples. Optional to enable sampling in Accuracy assessment workflow either by providing a preselected samples 
raster or number of samples e.g. int  for class independent sampling or an iterable for (reference) class defined values
e.g. \[n, m] where n and m are int.
