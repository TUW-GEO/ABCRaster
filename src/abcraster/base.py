# Copyright (c) 2022, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import math
from osgeo import ogr
import numpy as np
import pandas as pd
from veranda.io.geotiff import GeoTiffFile
from abcraster.input import rasterize
from abcraster.sampling import gen_random_sample
from abcraster.metrics import metrics


class Validation:
    """Class to perform a validation of binary classification results."""
    def __init__(self, ras_data_filepath, ref_data_filepath, out_dirpath, v_reprojected_filename='reproj_tmp.shp',
                 v_rasterized_filename='rasterized_ref.tif', ex_filepath=None, delete_tmp_files=False,
                 ras_data_nodata=255, ref_data_nodata=255):
        """
        Loads and harmonizes the classification resultan d reference data.

        Parameters
        ----------
        ras_data_filepath: str
            Path of binary classified raster tiff file.
        ref_data_filepath: str
            Path of reference data.
        out_dirpath: str
            Path of the output directory.
        v_reprojected_filename: str, optional
            Output path of the reprojected vector layer file (default: 'reproj_tmp.shp').
        v_rasterized_filename: str, optional
            Output path of the rasterized reference data (default: 'rasterized_val.tif').
        ex_filepath: str, optional
            Path of the exclusion layer which is not applied if set to None (default: None).
        delete_tmp_files: bool, optional
            Option to delete all temporary files (default: False).
        ras_data_nodata: int, optional
            No data value of the classification result (default: 255).
        ref_data_nodata: int, optional
            No data value of the reference data (default: 255).
        """

        # load classification result
        with GeoTiffFile(ras_data_filepath, auto_decode=False) as src:
            self.input_data = src.read(return_tags=False)
            self.gt = src.geotransform
            self.sref = src.spatialref

        # load exclusion mask
        if ex_filepath is None:
            self.ex_mask = None
        else:
            with GeoTiffFile(ex_filepath, auto_decode=False) as src:
                self.ex_mask = src.read(return_tags=False)
                ex_gt = src.geotransform
                ex_sref = src.spatialref

                if ex_gt != self.gt or ex_sref != self.sref:
                    raise RuntimeError("Grid/projection of input and reference data are not the same!")

        # handle reference data input
        ref_file_ext = os.path.splitext(os.path.basename(ref_data_filepath))[1]
        if ref_file_ext == '.shp':
            vec_ds = ogr.Open(ref_data_filepath)
            v_rasterized_path = os.path.join(out_dirpath, v_rasterized_filename)
            v_reprojected_path = os.path.join(out_dirpath, v_reprojected_filename)
            self.ref_data = rasterize(vec_ds, v_rasterized_path, self.input_data, self.gt, self.sref,
                                      v_reprojected_filepath=v_reprojected_path)

            # delete temporary files if requested
            if delete_tmp_files:
                os.remove(v_rasterized_path)
                delete_shapefile(v_reprojected_path)
        elif ref_file_ext == '.tif':
            with GeoTiffFile(ref_data_filepath, auto_decode=False) as src:
                self.ref_data = src.read(return_tags=False)
                ref_gt = src.geotransform
                ref_sref = src.spatialref

                if ref_gt != self.gt or ref_sref != self.sref:
                    raise RuntimeError("Grid/projection of input and reference data are not the same!")
        else:
            raise ValueError("Input file with extension " + ref_file_ext + " is not supported.")

        # define further attributes
        self.samples = None
        self.confusion_matrix = None
        self.confusion_map = None
        self.input_nodata = ras_data_nodata
        self.ref_nodata = ref_data_nodata

    def accuracy_assessment(self):
        """Runs validation on aligned numpy arrays."""

        # calculating difference between classification and reference
        res = 1 + (2 * self.input_data) - self.ref_data
        res[self.input_data == self.input_nodata] = 255
        res[self.ref_data == self.ref_nodata] = 255

        # applying exclusion, setting exclusion pixels as no data
        if self.ex_mask is not None:
            res[self.ex_mask == 1] = 255
            self.input_data[self.ex_mask == 1] = 255

        # store confusion map
        self.confusion_map = res

        # apply sampling
        if self.samples is not None:
            res[self.samples == 255] = 255

        # define confusion matrix
        tp = np.sum(res == 2)
        tn = np.sum(res == 1)
        fn = np.sum(res == 0)
        fp = np.sum(res == 3)
        self.confusion_matrix = np.array([[tp, fp], [fn, tn]])

    def define_sampling(self, sampling, opt_stratification=False, samples_filepath=None):
        """
        Defines samples for the validation.

        Parameters
        ----------
        sampling: list, tuple or int, optional
            stratified sampling = list/tuple of number samples, matching iterable index to class encoding
            non-stratified sampling = integer number of class-independent samples
            None = this implies samples are loaded from samples_filepath (default: None),
            *sampling is superseded if samples_filepath is None
        opt_stratification: bool, optional
            Should a stratification be applied (default: False)?
        samples_filepath: str, optional
            Path of the output samples file, which is writen if the path is passed (default: None).
        """

        # performs sampling
        self.samples = gen_random_sample(sampling, self.input_data, self.ref_data, exclusion=self.ex_data, nodata=255)

        # write output
        if samples_filepath is not None:
            with GeoTiffFile(samples_filepath, mode='w', count=1, geotransform=self.gt, spatialref=self.sref) as src:
                src.write(self.samples, band=1, nodata=[255])

    def load_sampling(self, samples_filepath):
        """Loads the samples from a raster file."""

        with GeoTiffFile(samples_filepath, auto_decode=False) as src:
            self.samples = src.read(return_tags=False)

    def calculate_accuracy_metric(self, metric_func):
        """
        Runs a function to calculate a certain accuracy metric.

        Parameters
        ----------
        metric_func: function
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
        with GeoTiffFile(valid_filepath, mode='w', count=1, geotransform=self.gt, spatialref=self.sref) as src:
            src.write(valid, band=1, nodata=[255])

    def write_confusion_map(self, out_filepath):
        """
        Writes confusion map.

        Parameters
        ----------
        out_filepath: str
            Path of the output file.
        """

        with GeoTiffFile(out_filepath, mode='w', count=1, geotransform=self.gt, spatialref=self.sref) as src:
            src.write(self.confusion_map, band=1, nodata=[255])


def delete_shapefile(shp_path):
    """ Deletes all files from which belong to a shapefile. """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_path):
        driver.DeleteDataSource(shp_path)


def run(ras_data_filepath, ref_data_filepath, out_dirpath, samples_filepath=None, sampling=None,
        diff_ras_out_filename='val.tif', v_reprojected_filename='reproj_tmp.shp',
        v_rasterized_filename='rasterized_ref.tif', out_csv_filename='val.csv', ex_filepath=None,
        delete_tmp_files=False, metrics_list=[]):

    v = Validation(ras_data_filepath, ref_data_filepath, out_dirpath, v_reprojected_filename,
                 v_rasterized_filename, ex_filepath, delete_tmp_files,
                 ras_data_nodata=255, ref_data_nodata=255)

    if samples_filepath is not None:
        if sampling is None:
            v.load_sampling(samples_filepath)
        else:
            v.define_sampling(sampling, samples_filepath=samples_filepath)

    v.accuracy_assessment()

    v.write_confusion_map(diff_ras_out_filename)

    input_base_filename = os.path.basename(ref_data_filepath)

    result = [input_base_filename]
    cols = ['file']

    for m in metrics_list:
        metric = metrics(m)
        result += [v.calculate_accuracy_metric(metric)]
        cols += [metric.__doc__]

    df = pd.DataFrame(result, cols)

    df.to_csv(out_csv_filename)

    print(df)
