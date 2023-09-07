import os
import argparse
from osgeo import ogr
import numpy as np
import pandas as pd
from geospade.raster import RasterGeometry
from geospade.crs import SpatialRef
from veranda.raster.native.geotiff import GeoTiffFile
from abcraster.input import rasterize, raster_reproject, raster_intersect, raster_read_from_polygon, update_filepath
from abcraster.sampling import gen_random_sample
from abcraster.metrics import metrics


class Validation:
    """Class to perform a validation of binary classification results."""

    def __init__(self, input_data_filepath, ref_data_filepath, out_dirpath, reproj_add_str='reproj',
                 rasterized_add_str='rasterized', delete_tmp_files=False, ras_data_nodata=255,
                 ref_data_nodata=255):
        """
        Loads and harmonizes the classification resultan d reference data.

        Parameters
        ----------
        input_data_filepath: str
            Path of binary classified raster tiff file.
        ref_data_filepath: str
            Path of reference data.
        out_dirpath: str
            Path of the output directory.
        reproj_add_str: str, optional
            String which will be added to the filename of vector or raster files after reprojecting (default: 'reproj').
        rasterized_add_str: str, optional
            String which is added to the filename of a vector file after being rasterized (default: 'rasterized').
        delete_tmp_files: bool, optional
            Option to delete all temporary files (default: False).
        ras_data_nodata: int, optional
            No data value of the classification result (default: 255).
        ref_data_nodata: int, optional
            No data value of the reference data (default: 255).
        """

        ref_file_ext = os.path.splitext(os.path.basename(ref_data_filepath))[1]

        if ref_file_ext == '.shp':
            # load classification result
            with GeoTiffFile(input_data_filepath) as input_ds:
                self.input_data = input_ds.read()[1]
                self.gt = input_ds.geotrans
                self.sref = input_ds.sref_wkt

            # rasterize vector-based reference data
            v_rasterized_path =  update_filepath(ref_data_filepath, add_str=rasterized_add_str, new_ext='tif',
                                                 new_root=out_dirpath)
            self.ref_data = rasterize(vec_path=ref_data_filepath, out_ras_path=v_rasterized_path,
                                      ras_path=input_data_filepath)

            # delete temporary files if requested
            if delete_tmp_files:
                os.remove(v_rasterized_path)

        elif ref_file_ext == '.tif':
            # calculate details of harmonized raster datasets
            with GeoTiffFile(input_data_filepath) as input_ds, GeoTiffFile(ref_data_filepath) as ref_ds:
                input_gt = input_ds.geotrans
                input_sref = input_ds.sref_wkt
                ref_gt = ref_ds.geotrans
                ref_sref = ref_ds.sref_wkt

                if ref_sref != input_sref:
                    ref_data_filepath = raster_reproject(ref_ds.filepath, input_ds.sref_wkt, int(input_ds.geotrans[1]),
                                                         out_dirpath, reproj_add_str)

                if ref_gt != input_gt:
                    input_geom = RasterGeometry(n_rows=input_ds.raster_shape[0], n_cols=input_ds.raster_shape[1],
                                                sref=SpatialRef(input_ds.sref_wkt, sref_type='wkt'),
                                                geotrans=input_ds.geotrans)
                    ref_geom = RasterGeometry(n_rows=ref_ds.raster_shape[0], n_cols=ref_ds.raster_shape[1],
                                              sref=SpatialRef(ref_ds.sref_wkt, sref_type='wkt'),
                                              geotrans=ref_ds.geotrans)
                    intersect_geom = raster_intersect(input_geom, ref_geom)
                else:
                    intersect_geom = None

            if intersect_geom is None:
                self.input_data = input_ds.read()[1]
                self.gt = input_ds.geotrans
                self.sref = input_ds.sref_wkt
                self.ref_data = ref_ds.read()[1]
            else:  # geocoding needs to be adapted
                self.input_data = raster_read_from_polygon(input_data_filepath, intersect_geom)
                self.gt = intersect_geom.geotrans
                self.sref = intersect_geom.sref.wkt
                self.ref_data = raster_read_from_polygon(ref_data_filepath, intersect_geom)

        else:
            raise ValueError("Input file with extension " + ref_file_ext + " is not supported.")

        # define further attributes
        self.input_path = input_data_filepath
        self.samples = None
        self.confusion_matrix = None
        self.confusion_map = None
        self.input_nodata = ras_data_nodata
        self.ref_nodata = ref_data_nodata
        self.out_dirpath = out_dirpath
        self.reproj_add_str = reproj_add_str
        self.rasterized_add_str = rasterized_add_str

    def accuracy_assessment(self):
        """Runs validation on aligned numpy arrays."""

        # calculating difference between classification and reference
        res = 1 + (2 * self.input_data) - self.ref_data
        res[self.input_data == self.input_nodata] = 255
        res[self.ref_data == self.ref_nodata] = 255
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
        self.samples = gen_random_sample(sampling, self.input_data, self.ref_data, nodata=255)

        # write output
        if samples_filepath is not None:
            with GeoTiffFile(samples_filepath, mode='w', geotrans=self.gt, sref_wkt=self.sref, nodatavals=255,
                             overwrite=True) as src:
                src.write(np.expand_dims(self.samples, axis=0))

    def load_sampling(self, samples_filepath):
        """Loads the samples from a raster file."""

        with GeoTiffFile(samples_filepath) as src:
            self.samples = src.read()[1]

    def apply_mask(self, mask_path, invert_mask=False):
        """
        Apply a raster or vector mask to the input data.

        Parameters
        ----------
        mask_path: str
            Path of the mask to be applied.
        invert_mask: bool, optional
            Option to invert the passed mask (default: False).
        """

        # load mask layer
        mask_ext = os.path.splitext(os.path.basename(mask_path))[1]
        if mask_ext == '.shp':
            v_rasterized_path = update_filepath(mask_path, add_str=self.rasterized_add_str, new_ext='tif',
                                                new_root=self.out_dirpath)
            ex_mask = rasterize(vec_path=mask_path, out_ras_path=v_rasterized_path, ras_path=self.input_path)
        elif mask_ext == '.tif':
            with GeoTiffFile(mask_path) as mask_ds:
                mask_gt = mask_ds.geotrans
                mask_sref = mask_ds.sref_wkt

                if self.sref != mask_sref:
                    mask_path = raster_reproject(mask_path, self.sref, self.out_dirpath, self.reproj_add_str)

                if self.gt != mask_gt:
                    geom = RasterGeometry(n_rows=self.input_data.shape[0], n_cols=self.input_data.shape[1],
                                          sref=SpatialRef(self.sref, sref_type='wkt'), geotrans=self.gt)
                    ex_mask = raster_read_from_polygon(mask_path, geom)
                else:
                    ex_mask = mask_ds.read()[1]

        else:
            raise ValueError("Input file with extension " + mask_ext + " is not supported.")

        # apply mask
        if invert_mask:
            ex_mask = ex_mask != 1
        else:
            ex_mask = ex_mask == 1
        if self.confusion_map is not None:
            self.confusion_map[ex_mask] = 255
        self.input_data[ex_mask] = 255

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

    def write_valid_array(self, valid_filepath):
        """
        Writes binary array, which indicates the valid pixels of the validation effort.

        Parameters
        ----------
        valid_filepath: str
            Path of the output file.
        """

        valid = np.logical_and(self.ref_data != 255, self.input_data != 255)
        with GeoTiffFile(valid_filepath, mode='w', geotrans=self.gt, sref_wkt=self.sref, nodatavals=255,
                         overwrite=True) as src:
            src.write(np.expand_dims(valid, axis=0))

    def write_confusion_map(self, out_filepath):
        """
        Writes confusion map.

        Parameters
        ----------
        out_filepath: str
            Path of the output file.
        """

        with GeoTiffFile(out_filepath, mode='w', geotrans=self.gt, sref_wkt=self.sref, nodatavals=255,
                         overwrite=True) as src:
            src.write(np.expand_dims(self.confusion_map, axis=0))


def delete_shapefile(shp_path):
    """ Deletes all files from which belong to a shapefile. """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_path):
        driver.DeleteDataSource(shp_path)


def run(input_data_filepaths, ref_data_filepath, out_dirpath, metrics_list, samples_filepath=None, sampling=None,
        diff_ras_out_filename='val.tif', reproj_add_str='reproj', rasterized_add_str='rasterized',
        out_csv_filename='val.csv', ex_filepath=None, delete_tmp_files=False):
    """
    Runs a validation workflow.

    Parameters
    ----------
    input_data_filepaths: list, str
        Paths of binary classified raster tiff files.
    ref_data_filepath: str
        Path of reference data.
    out_dirpath: str
        Path of the output directory.
    metrics_list: list, str
        List of metric (function) keys defined in abcraster.metrics.metrics dictionary.
    samples_filepath: str, optional
        Path of sampling raster or None if no sampling should be performed (default: None).
    sampling: list, tuple or int, optional
        stratified sampling = list/tuple of number samples, matching iterable index to class encoding
        non-stratified sampling = integer number of class-independent samples
        None = this implies samples are loaded from samples_filepath (default: None),
        *sampling is superseded if samples_filepath is None
    diff_ras_out_filename: str, optional
        Output path of the difference layer file (default: 'val.tif').
    reproj_add_str: str, optional
            String which will be added to the filename of vector or raster files after reprojecting (default: 'reproj').
    rasterized_add_str: str, optional
        String which is added to the filename of a vector file after being rasterized (default: 'rasterized').
    out_csv_filename: str, optional
        Output path of the validation measures as csv file. If set to None, no csv file is written (default: 'val.csv').
    ex_filepath: str, optional
        Path of the exclusion layer which is not applied if set to None (default: None).
    delete_tmp_files: bool, optional
        Option to delete all temporary files (default: False).
    Returns
    -------
    df: Pandas dataframe
        Dataframe containing the resulting validation measures. df is printed and written to csv TODO: fix output logic
    """

    ref_data_filepath_current = ref_data_filepath  # temporary place holder
    delete_tmp_files_current = False  # by default retain temporary files to reuse for subsequent runs
    num_inputs = len(input_data_filepaths)

    results = []
    cols = ['input file', 'reference file']

    for m in metrics_list:
        metric = metrics[m]
        cols += [metric.__doc__]

    for i in range(num_inputs):
        input_data_filepath = input_data_filepaths[i]

        input_base_filename = os.path.basename(input_data_filepath)
        ref_base_filename = os.path.basename(ref_data_filepath)

        # initialize validation object
        v = Validation(input_data_filepath, ref_data_filepath=ref_data_filepath_current, out_dirpath=out_dirpath,
                       reproj_add_str=reproj_add_str, rasterized_add_str=rasterized_add_str,
                       delete_tmp_files=delete_tmp_files_current, ref_data_nodata=255)

        # apply exclusion mask
        if ex_filepath is not None:
            v.apply_mask(ex_filepath)

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
            diff_ras_out_filename = os.path.join(out_dirpath, '{}--{}.tif'.format(input_base_filename.split('.')[0],
                                                                                ref_base_filename.split('.')[0]))

        v.write_confusion_map(os.path.join(out_dirpath, diff_ras_out_filename))

        result = [input_base_filename, ref_base_filename]

        # computes the selected metrics
        for m in metrics_list:
            metric = metrics[m]
            result += [v.calculate_accuracy_metric(metric)]
        results += [result]

        ref_file_ext = os.path.splitext(os.path.basename(ref_data_filepath))[1]
        if ref_file_ext == '.shp':
            v_rasterized_path = update_filepath(ref_data_filepath, add_str=rasterized_add_str, new_ext='tif',
                                                new_root=out_dirpath)
            ref_data_filepath_current = v_rasterized_path  # use the temporary rasterized reference

        if i == num_inputs - 1:  # if last run use delete flag from cli
            if delete_tmp_files:
                v_rasterized_path = update_filepath(ref_data_filepath, add_str=rasterized_add_str, new_ext='tif',
                                                    new_root=out_dirpath)
                os.remove(v_rasterized_path)

    if num_inputs == 1:  # changed output format for multiple results
        df = pd.DataFrame(results[0], cols)
        print(df)
    else:
        df = pd.DataFrame(results, columns=cols)

    df.to_csv(os.path.join(out_dirpath, out_csv_filename))

    return df


def command_line_interface():
    """ Command line interface to perform a validation based on vector-based reference data. """

    # define parser
    parser = argparse.ArgumentParser(
        description="Simple Binary Validation Workflow. Initially designed to validate floods")
    parser.add_argument("-in", "--input_filepaths",
                        help="Full file path to the binary raster data 1= presence, 0=absence, for now 255=nodata.",
                        required=True, nargs="+", type=str)
    parser.add_argument("-ref", "--reference_filepath",
                        help="Full file path to the validation dataset (.tif or .shp, in any projection)",
                        required=True, type=str)
    parser.add_argument("-ex", "--exclusion_filepath",
                        help="Full file path to the binary exclusion data 1= exclude, for now 255=nodata.",
                        required=False, type=str)
    parser.add_argument("-ns", "--num_samples",
                        help="number of total samples if sampling will be applied.", required=False, type=int)
    parser.add_argument("-stf", "--stratify", help="Stratification based on reference data", required=False,
                        default=True, action="store_true")
    parser.add_argument("-nst", "--no_stratify", dest='stratify', action='store_false',
                        help="No Stratification option")
    parser.add_argument("-sfp", "--samples_filepath",
                        help="Full file path to the sampling raster dataset (.tif ), if num samples not specified, \
                        assumes samples will be read from this path",
                        required=False, type=str)
    parser.add_argument("-out", "--output_raster",
                        help="Full file path to the final difference raster", required=True, type=str)
    parser.add_argument("-csv", "--output_csv",
                        help="Full file path to the csv results", required=False, type=str)
    parser.add_argument("-del", "--delete_tmp",
                        help="Option to delete temporary files.", required=False, type=bool)
    parser.add_argument("-all", "--all_metrics", help="Option to compute all metrics.",
                        default=True, action="store_true")
    parser.add_argument('-na', "--not_all_metrics", dest='all_metrics', action='store_false',
                        help="Option not to compute all metrics, metrics should be specified if activated.")
    parser.add_argument("-mts", "--metrics", nargs="+", required=False, type=str,
                        help="Option to list metrics to run.")

    # collect inputs
    args = parser.parse_args()
    input_raster_filepaths = args.input_filepaths
    exclusion_filepath = args.exclusion_filepath
    validation_filepath = args.reference_filepath
    output_raster_filepath = args.output_raster
    output_csv_filepath = args.output_csv
    delete_tmp = args.delete_tmp
    strat = args.stratify
    sampling = args.num_samples
    samples_filepath = args.samples_filepath

    if args.all_metrics:
        metrics_list = metrics.keys()  # all metrics as defined in metrics dictionary
    else:
        metrics_list = args.metrics

        if len(metrics_list) == 0:
            raise RuntimeError('Metric keys list not found!')

        for m in metrics_list:
            if m not in metrics:
                raise RuntimeError('Metric key ({}) not found!'.format(m))

    if len(input_raster_filepaths) > 1 and output_csv_filepath is None:
        raise RuntimeError('Output CSV file is required for multiple input files')

    if args.metrics is not None and len(args.metrics) > 0 and args.all_metrics:
        print("WARNING: Specific metrics specified but all metrics selected. Proceeding with all metrics")

    if sampling is not None:
        if strat:
            sampling = [sampling]
        else:
            sampling = int(sampling)

        if samples_filepath is None:
            print("WARNING: Number of samples specified but no filepath to output samples raster specified! \
            Ignoring number of samples. Proceeding without sampling.")  # warning or run time error?
            sampling = None

    # define output names
    out_dirpath, out_raster_filename = os.path.split(output_raster_filepath)

    # set default option
    if delete_tmp is None:
        delete_tmp = False

    run(input_data_filepaths=input_raster_filepaths, ref_data_filepath=validation_filepath, out_dirpath=out_dirpath,
        diff_ras_out_filename=out_raster_filename, out_csv_filename=output_csv_filepath, ex_filepath=exclusion_filepath,
        delete_tmp_files=delete_tmp, sampling=sampling, samples_filepath=samples_filepath, metrics_list=metrics_list)


if __name__ == '__main__':
    command_line_interface()
