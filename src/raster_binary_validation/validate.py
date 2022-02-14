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


import math
from osgeo import ogr
import numpy as np
import pandas as pd
from veranda.io.geotiff import GeoTiffFile
from raster_binary_validation.input import rasterize


def run(ras_data_filepath, v_val_data_filepath, diff_ras_out_filepath='val.tif',
        v_reprojected_filepath='reproj_tmp.shp', v_rasterized_filepath='rasterized_val.tif',
        out_csv_filepath='val.csv', ex_filepath=None):
    """
    Runs the validation with vector data input (presence = 1, absence=0).

    Parameters
    ----------
    ras_data_filepath: str
        Path of classification result.
    v_val_data_filepath: str
        Path of reference data.
    diff_ras_out_filepath: str, optional
        Output path of the difference layer file (default: 'val.tif').
    v_reprojected_filepath: str, optional
        Output path of the reprojected vector layer file (default: 'reproj_tmp.shp').
    v_rasterized_filepath: str, optional
        Output path of the rasterized reference data (default: 'rasterized_val.tif').
    out_csv_filepath: str, optional
        Output path of the validation measures as csv file (default: 'val.csv').
    ex_filepath: str, optional
        Path of the exclusion layer which is not applied if set to None (default: None).
    """

    vec_ds = ogr.Open(v_val_data_filepath)
    with GeoTiffFile(ras_data_filepath, auto_decode=False) as src:
        flood_data = src.read(return_tags=False)
        gt = src.geotransform
        sref = src.spatialref

    if ex_filepath is None:
        ex_data = None
    else:
        with GeoTiffFile(ex_filepath, auto_decode=False) as src:
            ex_data = src.read(return_tags=False)

    print('rasterizing')
    val_data = rasterize(vec_ds, v_rasterized_filepath, flood_data, gt, sref,
                         v_reprojected_filepath=v_reprojected_filepath)
    print('done ... rasterizing')

    print('start validation')
    res, idx, UA, PA, Ce, Oe, CSI, F1, SR, K, A = validate(flood_data, val_data, mask=ex_data, data_nodata=255,
                                                           val_nodata=255)

    # save results
    res = res.astype(np.uint8)
    res[~idx] = 255
    with GeoTiffFile(diff_ras_out_filepath, mode='w', count=1, geotransform=gt, spatialref=sref) as geotiff:
        geotiff.write(res, band=1, nodata=[255])

    #
    dat = [['result 1', UA, PA, Ce, Oe, CSI, F1, SR, K, A]]
    df = pd.DataFrame(dat,
                      columns=['file', "User's Accuracy/Precision", "Producer's Accuracy/Recall", 'Commission Error',
                               'Omission Error', 'Critical Success Index', 'F1', 'Success Rate', 'Kappa', 'Accuracy'])
    df.to_csv(out_csv_filepath)
    print('end validation')


def validate(data, val_data, mask=None, data_nodata=255, val_nodata=255):
    """
    Runs validation on aligned numpy arrays.

    Parameters
    ----------
    data: numpy.array
        Binary classification result which will be validated.
    val_data: numpy.array
        Binary reference data array.
    mask: numpy.array
        Binary mask to be applied on both input arrays.
    data_nodata: int, optional
        No data value of the classification result (default: 255).
    val_nodata: int, optional
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

    res = 1 + (2 * data) - val_data
    res[data == data_nodata] = 255
    res[val_data == val_nodata] = 255

    if mask is not None:
        res[mask == 1] = 255
        data[mask == 1] = 255  # applying exclusion, setting exclusion pixels as no data
    valid = np.logical_and(val_data != 255, data != 255)  # index removing no data from comparison

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

    print("User's Accuracy/Precision: %f" % (UA))
    print("Producer's Accuracy/Recall/PP2: %f" % (PA))
    print("Critical Success In: %f" % (CSI))
    print("F1: %f" % (F1))
    print("commission error: %f" % (Ce))
    print("omission error: %f" % (Oe))
    print("total accuracy: %f" % (A))
    print("kappa: %f" % (K))
    print("Penalization Function: %f" % (P))
    print("Success Rate: %f" % (SR))

    return res, valid, UA, PA, Ce, Oe, CSI, F1, SR, K, A
