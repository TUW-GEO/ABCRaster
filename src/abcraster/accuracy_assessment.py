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


def run(ras_data_filepath, ref_data_filepath, out_dirpath, sample_filepath=None, sampling=None,
        diff_ras_out_filename='val.tif', v_reprojected_filename='reproj_tmp.shp',
        v_rasterized_filename='rasterized_ref.tif', out_csv_filename='val.csv', ex_filepath=None,
        delete_tmp_files=False):
    """
    Runs the validation with vector data input (presence = 1, absence=0).

    Parameters
    ----------
    ras_data_filepath: str
        Path of binary classified raster tiff file.
    ref_data_filepath: str
        Path of reference data.
    out_dirpath: str
        Path of the output directory.
    sample_filepath: str, optional
        Path of sampling raster or None if no sampling should be performed (default: None).
    sampling: list, tuple or int, optional
        stratified sampling = list/tuple of number samples, matching iterable index to class encoding
        non-stratified sampling = integer number of class-independent samples
        None = this implies samples are loaded from sample_filepath (default: None)
    diff_ras_out_filename: str, optional
        Output path of the difference layer file (default: 'val.tif').
    v_reprojected_filename: str, optional
        Output path of the reprojected vector layer file (default: 'reproj_tmp.shp').
    v_rasterized_filename: str, optional
        Output path of the rasterized reference data (default: 'rasterized_val.tif').
    out_csv_filename: str, optional
        Output path of the validation measures as csv file. If set to None, no csv file is written (default: 'val.csv').
    ex_filepath: str, optional
        Path of the exclusion layer which is not applied if set to None (default: None).
    delete_tmp_files: bool, optional
        Option to delete all temporary files (default: False).

    Returns
    -------
    val_measures: dict
        Dictionary containing the resulting validation measures.
    """

    print('Load classification result.')
    with GeoTiffFile(ras_data_filepath, auto_decode=False) as src:
        input_data = src.read(return_tags=False)
        gt = src.geotransform
        sref = src.spatialref

    if ex_filepath is None:
        ex_data = None
    else:
        print('Load exclusion layer.')
        with GeoTiffFile(ex_filepath, auto_decode=False) as src:
            ex_data = src.read(return_tags=False)

    # handle reference data input
    ref_file_ext = os.path.splitext(os.path.basename(ref_data_filepath))[1]
    if ref_file_ext == '.shp':
        print('Load and rasterize vector reference data.')
        vec_ds = ogr.Open(ref_data_filepath)
        v_rasterized_path = os.path.join(out_dirpath, v_rasterized_filename)
        v_reprojected_path = os.path.join(out_dirpath, v_reprojected_filename)

        ref_data = rasterize(vec_ds, v_rasterized_path, input_data, gt, sref,
                             v_reprojected_filepath=v_reprojected_path)
        print('Done ... rasterizing')

        # delete temporary files if requested
        if delete_tmp_files:
            os.remove(v_rasterized_path)
            delete_shapefile(v_reprojected_path)
    elif ref_file_ext == '.tif':
        print('Load raster reference data.')
        with GeoTiffFile(ref_data_filepath, auto_decode=False) as src:
            ref_data = src.read(return_tags=False)  # TODO: add projecttion check and reprojection procedure
    else:
        raise ValueError("Input file with extension " + ref_file_ext + " is not supported.")

    # sampling logic
    if sample_filepath is None:
        # no sampling
        samples = None
    else:
        if sampling is None:
            with GeoTiffFile(sample_filepath, auto_decode=False) as src:
                # assumes there is a sampling raster existing, then reads it
                samples = src.read(return_tags=False)
            samples = samples == 1
        else:
            # performs sampling
            samples = gen_random_sample(sampling, input_data, ref_data, nodata=255)
            with GeoTiffFile(sample_filepath, mode='w', count=1, geotransform=gt, spatialref=sref) as src:
                src.write(samples.astype(np.uint8), band=1, nodata=255)

    print('Start validation')
    res, idx, UA, PA, Ce, Oe, CSI, F1, SR, K, A = accuracy_assessment(input_data, ref_data, mask=ex_data,
                                                                      samples=samples, data_nodata=255, ref_nodata=255)

    # write difference map
    res = res.astype(np.uint8)
    res[~idx] = 255
    diff_ras_out_path = os.path.join(out_dirpath, diff_ras_out_filename)
    with GeoTiffFile(diff_ras_out_path, mode='w', count=1, geotransform=gt, spatialref=sref) as geotiff:
        geotiff.write(res, band=1, nodata=[255])

    # write csv summary
    input_base_filename = os.path.basename(ref_data_filepath)
    if out_csv_filename is not None:
        out_csv_path = os.path.join(out_dirpath, out_csv_filename)
        dat = [[input_base_filename, UA, PA, Ce, Oe, CSI, F1, SR, K, A]]
        df = pd.DataFrame(dat,
                          columns=['file', "User's Accuracy/Precision", "Producer's Accuracy/Recall",
                                   'Commission Error', 'Omission Error', 'Critical Success Index', 'F1', 'Success Rate',
                                   'Kappa', 'Accuracy'])
        df.to_csv(out_csv_path)

    # return validation measures as dictionary
    val_measures = {'file': input_base_filename, "User's Accuracy/Precision": UA, "Producer's Accuracy/Recall": PA,
                    'Commission Error': Ce, 'Omission Error': Oe, 'Critical Success Index': CSI, 'F1': F1,
                    'Success Rate': SR, 'Kappa': K, 'Accuracy': A}

    print('End validation')
    return val_measures


def accuracy_assessment(data, ref_data, mask=None, samples=None, data_nodata=255, ref_nodata=255):
    """
    Runs validation on aligned numpy arrays.

    Parameters
    ----------
    data: numpy.array
        Binary classification result which will be validated.
    ref_data: numpy.array
        Binary reference data array.
    mask: numpy.array, optional
        Binary mask to be applied on both input arrays (default: None).
    samples: numpy.array, optional
        Boolean array showing the pixels which should be considered for accuracy assessment (default: None).
    data_nodata: int, optional
        No data value of the classification result (default: 255).
    ref_nodata: int, optional
        No data value of the reference data (default: 255).

    Returns
    -------
    res: numpy.array
        Array which includes the differences of reference data and binary result.
    valid: numpy.array
        Array which includes the pixels which have valid data
    UA: float
        User's accuracy/Precision
    PA: float
        Producer's accuracy/Recall
    Ce: float
        Comission error
    Oe: float
        Omission error
    CSI: float
        Critical success index
    F1: float
        F1-score
    SR: float
        Success rate
    K: float
        Kappa coefficient
    A: float
        Accuracy
    """

    res = 1 + (2 * data) - ref_data
    res[data == data_nodata] = 255
    res[ref_data == ref_nodata] = 255

    if mask is not None:
        res[mask == 1] = 255
        data[mask == 1] = 255  # applying exclusion, setting exclusion pixels as no data
    valid = np.logical_and(ref_data != 255, data != 255)  # index removing no data from comparison raster

    ras_result = res

    if samples is not None:
        res[~samples] = 255

    TP = np.sum(res == 2)
    TN = np.sum(res == 1)
    FN = np.sum(res == 0)
    FP = np.sum(res == 3)
    print(np.array([[TP, FP], [FN, TN]]))

    # calculating metrics
    Po = A = (TP + TN) / (TP + TN + FP + FN)
    Pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / (TP + TN + FP + FN) ** 2
    K = (Po - Pe) / (1 - Pe)
    UA = TP / (TP + FP)
    PA = TP / (TP + FN)  # accuracy:PP2 as defined in ACube4Floods 5.1
    CSI = TP / (TP + FP + FN)
    F1 = (2 * TP) / (2 * TP + FN + FP)
    Ce = FP / (FP + TP)  # inverse of precision
    Oe = FN / (FN + TP)  # inverse of recall
    P = math.exp(FP / ((TP + FN) / math.log(0.5)))  # penalization as defined in ACube4Floods 5.1
    SR = PA - (1 - P)  # Success rate as defined in ACube4Floods 5.1

    print("User's Accuracy/Precision: %f" % UA)
    print("Producer's Accuracy/Recall/PP2: %f" % PA)
    print("Critical Success In: %f" % CSI)
    print("F1: %f" % F1)
    print("commission error: %f" % Ce)
    print("omission error: %f" % Oe)
    print("total accuracy: %f" % A)
    print("kappa: %f" % K)
    print("Penalization Function: %f" % P)
    print("Success Rate: %f" % SR)

    return ras_result, valid, UA, PA, Ce, Oe, CSI, F1, SR, K, A


def delete_shapefile(shp_path):
    """ Deletes all files from which belong to a shapefile. """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_path):
        driver.DeleteDataSource(shp_path)
