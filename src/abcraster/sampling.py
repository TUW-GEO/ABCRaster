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

import os, argparse
from veranda.io.geotiff import GeoTiffFile
import numpy as np


def gen_random_sample(num_samples, data, ref, nodata=255):
    """
    Creates a numpy array mask of randomly selected samples given a reference and input classified data.

    Parameters
    ----------
    num_samples: list, tuple or int
        number of samples where iterable index matches class encoding,
        for non-stratified sampling pass a singleton
    data: numpy.array
        (binary) classified data, assumes uint encoded
    ref: numpy.array
        reference data, assumes same data format and projection as data
    nodata: int, optional
        nodata value, assumes the same for both reference and input data

    Returns
    -------
    samples: numpy.array
        Boolean array containing pixels which are considered as sample (=True) and those which are not considered
        (=False).
    """

    # preform checks
    assert(data.shape == ref.shape)

    # initialize samples
    samples = ~np.ones_like(data, dtype=bool)
    nodata_mask = (ref == nodata) | (data == nodata)

    if isinstance(num_samples, list) or isinstance(num_samples, tuple):  # stratified sampling
        # define number of samples per class
        num_class = np.max(ref[ref != nodata]) + 1
        if len(num_samples) == 1:
            num_samples = num_samples * num_class
        elif len(num_samples) != num_class:
            raise ValueError("Dimension of samples do not correspond to the number of classes.")

        # select samples
        for class_id in range(num_class):
            class_sel = random_conditional_selection(arr=ref, num=num_samples[class_id], apriori_mask=nodata_mask,
                                                     cond=class_id)
            samples[class_sel] = True

    elif isinstance(num_samples, int):  # non-stratified sampling
        sel = random_conditional_selection(arr=ref, num=num_samples, apriori_mask=nodata_mask)
        samples[sel] = True

    else:
        raise ValueError("Unknown type for the number of samples variable.")

    return samples


def random_conditional_selection(arr, num, apriori_mask, cond=None):
    """
    Selects indices randomly from an array, only considering a specific value within the array.

    Parameters
    ----------
    arr: numpy.array
        Array which includes integer values.
    num: int
        Number of indices which should be retrieved randomly.
    apriori_mask: numpy.array
        Boolean array showing the pixels which should be excluded a priori.
    cond: int, optional
        Specific integers to which the selection should be limited to or no limitation (=None).

    Returns
    -------
    idx: numpy.array
        Array containing selected indices.
    """

    # get indices
    arr_flt = arr.flatten()
    apriori_mask = apriori_mask.flatten()
    idx_arr = np.arange(arr_flt.size)

    # define masking
    if cond is not None:
        mask = (arr_flt != cond) | (apriori_mask)
    else:
        mask = apriori_mask

    # random selection
    idx_arr = idx_arr[~mask]
    np.random.shuffle(idx_arr)
    idx = idx_arr[:num]

    return np.unravel_index(idx, arr.shape)


def path_wrapper(num_samples, data_path, ref_path, out_path, nodata=255):
    """
    Wraps genrate samples as a function opening the input files
    """

    with GeoTiffFile(data_path, auto_decode=False) as src:
        data = src.read(return_tags=False)
        data_gt = src.geotransform
        data_sref = src.spatialref

    with GeoTiffFile(ref_path, auto_decode=False) as src:
        ref = src.read(return_tags=False)
        ref_gt = src.geotransform
        ref_sref = src.spatialref

    assert(ref_gt == data_gt)
    assert(ref_sref == data_sref)

    samples = gen_random_sample(num_samples, data, ref, nodata=nodata)

    with GeoTiffFile(out_path, mode='w', count=1, geotransform=gt, spatialref=sref) as src:
        src.write(samples, band=1, nodata=nodata)


def command_line_interface():
    """ Command line interface to perform sampling. """

    # define parser
    parser = argparse.ArgumentParser(
        description="Sampling CLI for ABCRaster")
    parser.add_argument("-in", "--input_filepath",
                        help="Full file path to the binary raster data 1= presence, 0=absence, for now 255=nodata.",
                        required=True, type=str)
    parser.add_argument("-ex", "--exclusion_filepath",
                        help="Full file path to the binary exclusion data 1= exclude, for now 255=nodata.",
                        required=False, type=str)
    parser.add_argument("-ref", "--reference_file",
                        help="Full file path to the reference raster dataset (.tif or .shp, in any projection)",
                        required=True, type=str)
    parser.add_argument("-out", "--output_raster",
                        help="Full file path to the sampling raster", required=True, type=str)
    parser.add_argument("-n", "--num_samples",
                        help="number of samples.", required=True, type=int, default=2000)
    parser.add_argument("-nd", "--nodata",
                        help="No data value.", required=False, type=int, default=255)
    parser.add_argument("-stf", "--stratify",
                        help="Stratified.", required=False, type=bolean, default=True)

    # collect inputs
    args = parser.parse_args()
    data_path = args.input_filepath
    exclusion_filepath = args.exclusion_filepath
    ref_path = args.reference_file
    out_path = args.output_raster
    n = args.num_samples
    nodata = args.nodata
    strat = args.stratify

    if strat:
        num_samples = [n/2, n/2]
    else:
        num_samples = [n]

    path_wrapper(num_samples, data_path, ref_path, out_path, nodata=nodata)


if __name__ == '__main__':
    command_line_interface()
