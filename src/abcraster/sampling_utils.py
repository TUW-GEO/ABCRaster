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

def gen_random_sample(n, data, ref, num_class=2, nodata=255, strat=True):
    """
        Creates a numpy array mask of randomly selected samples given a reference and input classified data.

        Parameters
        ----------
        n: int
            number of samples, should be divisible by num_class
        data: numpy Array
            (binary) classified data, assumes uint encoded
        ref: numpy Array
            reference data, assumes same data format and projection as data
        num_class: int, optional
            number of classes
        nodata: int, optional
            nodata value, assumes the same for both reference and input data
        strat: boolean, optional
            is stratified sampling
    """

    assert(data.shape == ref.shape)
    assert(n % num_class == 0)

    samples = np.zeros_like(data, dtype=np.uint8)
    x_size, y_size = data.shape

    if strat:
        for i in range(num_class):
            cnt = 0
            while cnt < n / num_class:
                x = np.random.random_integers(0, x_size - 1)
                y = np.random.random_integers(0, y_size - 1)

                if data[x, y] != nodata and ref[x, y] == i and samples[x, y] == 0:
                    samples[x, y] = 1
                    cnt += 1
    else:
        cnt = 0
        while cnt < n:
            x = np.random.random_integers(0, x_size - 1)
            y = np.random.random_integers(0, y_size - 1)

            if data[x, y] != nodata and ref[x, y] != nodata and samples[x, y] == 0:
                samples[x, y] = 1
                cnt += 1

    return samples

def path_wrapper(n, data_path, ref_path, out_path, num_class=2, nodata=255, strat=True):
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

    samples = gen_random_sample(n, data, ref, num_class=num_class, nodata=nodata, strat=strat)

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

    path_wrapper(n, data_path, ref_path, out_path, num_class=2, nodata=nodata, strat=strat)


if __name__ == '__main__':
    command_line_interface()