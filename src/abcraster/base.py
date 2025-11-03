import geopandas as gpd
import rioxarray
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from abcraster.sampling import gen_random_sample
from abcraster.metrics import metrics
from abcraster.input import rasterize_to_rioxarray, ensure_path


class Validation:
    """Class to perform a validation of binary classification results."""

    def __init__(self, input_data_filepath: Path, ref_data_filepath: Path, out_dirpath: Path):
        """
        Loads and harmonizes the classification result and reference data.

        Parameters
        ----------
        input_data_filepath: Path or str
            Path of binary classified raster tiff file.
        ref_data_filepath: Path or str
            Path of reference data.
        out_dirpath: Path or str
            Path of the output directory.
        ras_data_nodata: int, optional
            No data value of the classification result (default: 255).
        ref_data_nodata: int, optional
            No data value of the reference data (default: 255).
        """

        input_data_filepath = ensure_path(input_data_filepath)
        ref_data_filepath = ensure_path(ref_data_filepath)
        out_dirpath = ensure_path(out_dirpath)

        if ref_data_filepath.suffix == '.shp':
            self.input_ds = rioxarray.open_rasterio(input_data_filepath)
            ref_vec_data = gpd.read_file(ref_data_filepath)
            if ref_vec_data.crs != self.input_ds.rio.crs:
                ref_vec_data = ref_vec_data.to_crs(self.input_ds.rio.crs)
            self.ref_ds = rasterize_to_rioxarray(vec_gpf=ref_vec_data, riox_arr=self.input_ds)

        elif ref_data_filepath.suffix == '.tif':
            self.input_ds = rioxarray.open_rasterio(input_data_filepath)
            self.ref_ds = rioxarray.open_rasterio(ref_data_filepath)
            self.ref_ds = self.ref_ds.rio.reproject_match(self.input_ds)

        else:
            raise ValueError("Input file with extension " + ref_data_filepath.suffix + " is not supported.")

        # define further attributes
        self.input_path = input_data_filepath
        self.samples = None
        self.confusion_matrix = None
        self.confusion_map = None
        self.out_dirpath = out_dirpath

    def accuracy_assessment(self):
        """Runs validation on aligned numpy arrays."""

        # calculating difference between classification and reference
        res = 1 + (2 * self.input_ds.values) - self.ref_ds.values
        res[self.input_ds.values == self.input_ds.rio.nodata] = 255
        res[self.ref_ds.values == self.ref_ds.rio.nodata] = 255
        self.confusion_map = res

        # apply sampling
        if self.samples is not None:
            res_dcopy = np.array([x for x in res])
            res_dcopy[self.samples == 255] = 255
            tp = np.sum(res_dcopy == 2)
            tn = np.sum(res_dcopy == 1)
            fn = np.sum(res_dcopy == 0)
            fp = np.sum(res_dcopy == 3)
        else:
            # define confusion matrix
            tp = np.sum(res == 2)
            tn = np.sum(res == 1)
            fn = np.sum(res == 0)
            fp = np.sum(res == 3)
        self.confusion_matrix = np.array([[tp, fp], [fn, tn]])

    def define_sampling(self, sampling, samples_filepath=None):
        """
        Defines samples for the validation.

        Parameters
        ----------
        sampling: list, tuple or int, optional
            stratified sampling = list/tuple of number samples, matching iterable index to class encoding
            non-stratified sampling = integer number of class-independent samples
            None = this implies samples are loaded from samples_filepath (default: None),
            *sampling is superseded if samples_filepath is None
        samples_filepath: str, optional
            Path of the output samples file, which is writen if the path is passed (default: None).
        """

        # performs sampling
        self.samples = gen_random_sample(sampling, self.input_ds.values, self.ref_ds.values, nodata=255)

        # write output
        if samples_filepath is not None:
            samples_filepath = ensure_path(samples_filepath)
            self.write_output_file(self.samples, samples_filepath)

    def load_sampling(self, samples_filepath):
        """Loads the samples from a raster file."""

        self.samples = rioxarray.open_rasterio(samples_filepath).values

    def apply_mask(self, mask_path: Path, invert_mask=False):
        """
        Apply a raster or vector mask to the input data.

        Parameters
        ----------
        mask_path: Path
            Path of the mask to be applied.
        invert_mask: bool, optional
            Option to invert the passed mask (default: False).
        """

        # load mask layer
        mask_path = ensure_path(mask_path)
        if mask_path.suffix == '.shp':
            mask_vec_data = gpd.read_file(mask_path)
            if mask_vec_data.crs != self.input_ds.rio.crs:
                mask_vec_data = mask_vec_data.to_crs(self.input_ds.rio.crs)
            ex_mask = rasterize_to_rioxarray(vec_gpf=mask_vec_data, riox_arr=self.input_ds)
        elif mask_path.suffix == '.tif':
            ex_mask = rioxarray.open_rasterio(mask_path)
            ex_mask = ex_mask.rio.reproject_match(self.input_ds)
        else:
            raise ValueError("Input file with extension " + mask_path.suffix + " is not supported.")

        # apply mask
        if invert_mask:
            ex_mask = ex_mask != 1
        else:
            ex_mask = ex_mask == 1
        if self.confusion_map is not None:
            self.confusion_map[ex_mask.values] = 255
        self.input_ds = self.input_ds.where(~ex_mask, self.input_ds.rio.nodata)

    def calculate_accuracy_metric(self, metric_func):
        """
        Runs a function to calculate a certain accuracy metric.

        Parameters
        ----------
        metric_func: callable
            Function, which takes in a dictionary with the keys TP (true positives), FP (false positives), TN (true
            negatives) and FN (false negatives), and returns an accuracy metric.

        Returns
        -------
        accuracy_metric: float
            Resulting metric of the used function.
        """

        # create dictionary
        conf = dict()
        conf['TP'] = self.confusion_matrix[0, 0]
        conf['FP'] = self.confusion_matrix[0, 1]
        conf['FN'] = self.confusion_matrix[1, 0]
        conf['TN'] = self.confusion_matrix[1, 1]

        # run function
        return metric_func(conf)

    def write_output_file(self, out_arr, out_fpath):
        if isinstance(out_arr, np.ndarray):
            out_ds = self.input_ds.copy(deep=True)
            out_ds.values = out_arr
            out_ds.rio.to_raster(out_fpath)
        elif isinstance(out_arr, xr.DataArray):
            out_arr.rio.to_raster(out_fpath)

    def write_valid_array(self, valid_filepath):
        """
        Writes binary array, which indicates the valid pixels of the validation effort.

        Parameters
        ----------
        valid_filepath: str
            Path of the output file.
        """

        valid = np.logical_and(self.ref_ds.values != 255, self.input_ds.values != 255).astype(np.uint8)
        self.write_output_file(valid, valid_filepath)

    def write_confusion_map(self, out_filepath):
        """
        Writes confusion map.

        Parameters
        ----------
        out_filepath: str
            Path of the output file.
        """

        self.write_output_file(self.confusion_map, out_filepath)


def run(input_data_filepaths, ref_data_filepath: Path, out_dirpath: Path, metrics_list: list,
        samples_filepath: Path = None, sampling=None, diff_ras_out_filename='val.tif', out_csv_filename='val.csv',
        ex_filepath=None, aoi_filepath=None):
    """
    Runs a validation workflow.

    Parameters
    ----------
    input_data_filepaths: list[Path], Path
        Paths of binary classified raster tiff files.
    ref_data_filepath: Path
        Path of reference data.
    out_dirpath: Path
        Path of the output directory.
    metrics_list: list, str
        List of metric (function) keys defined in abcraster.metrics.metrics dictionary.
    samples_filepath: Path, optional
        Path of sampling raster or None if no sampling should be performed (default: None).
    sampling: list, tuple or int, optional
        stratified sampling = list/tuple of number samples, matching iterable index to class encoding
        non-stratified sampling = integer number of class-independent samples
        None = this implies samples are loaded from samples_filepath (default: None),
        *sampling is superseded if samples_filepath is None
    diff_ras_out_filename: str, optional
        Output path of the difference layer file (default: 'val.tif').
    out_csv_filename: str, optional
        Output path of the validation measures as csv file. If set to None, no csv file is written (default: 'val.csv').
    ex_filepath: Path, optional
        Path of the exclusion layer which is not applied if set to None (default: None).
    aoi_filepath: Path, optional
        Path of the AOI layer which is not applied if set to None (default: None).

    Returns
    -------
    df: Pandas dataframe
        Dataframe containing the resulting validation measures. df is printed and written to csv TODO: fix output logic
    """

    input_data_filepaths = [ensure_path(idf) for idf in input_data_filepaths]
    ref_data_filepath = ensure_path(ref_data_filepath)
    out_dirpath = ensure_path(out_dirpath)
    num_inputs = len(input_data_filepaths)
    results = []
    cols = ['input file', 'reference file']

    for m in metrics_list:
        metric = metrics[m]
        cols += [metric.__doc__]

    for i in range(num_inputs):
        input_data_filepath = input_data_filepaths[i]

        # initialize validation object
        v = Validation(input_data_filepath, ref_data_filepath=ref_data_filepath, out_dirpath=out_dirpath)

        # apply exclusion mask
        if ex_filepath is not None:
            v.apply_mask(ex_filepath)

        # apply aoi mask
        if aoi_filepath is not None:
            v.apply_mask(aoi_filepath, invert_mask=True)

        # handle sampling
        if samples_filepath is not None:  # logic, could be integrated in the object/class
            if sampling is None:
                v.load_sampling(samples_filepath)
            else:
                v.define_sampling(sampling, samples_filepath=samples_filepath)
                sampling = None  # set to none, to use after 1st sampling file created

        # run accuracy estimation
        v.accuracy_assessment()
        if num_inputs > 1:
            # overrride output file name
            # naming the output files based on input and reference
            diff_ras_out_filename = out_dirpath / '{}--{}.tif'.format(input_data_filepath.stem,
                                                                      ref_data_filepath.stem)

        v.write_confusion_map(out_dirpath / diff_ras_out_filename)

        result = [input_data_filepath.name, ref_data_filepath.name]

        # computes the selected metrics
        for m in metrics_list:
            metric = metrics[m]
            result += [v.calculate_accuracy_metric(metric)]
        results += [result]

    if num_inputs == 1:  # changed output format for multiple results
        df = pd.DataFrame(results[0], cols)
        print(df)
    else:
        df = pd.DataFrame(results, columns=cols)

    df.to_csv(out_dirpath / out_csv_filename)

    return df
