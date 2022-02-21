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


from abcraster.accuracy_assessment import run
import argparse
import os


def command_line_interface():
    """ Command line interface to perform a validation based on vector-based reference data. """

    # define parser
    parser = argparse.ArgumentParser(
        description="Simple Binary Validation Workflow. Initially designed to validate floods")
    parser.add_argument("-in", "--input_filepath",
                        help="Full file path to the binary raster data 1= presence, 0=absence, for now 255=nodata.",
                        required=True, type=str)
    parser.add_argument("-ex", "--exclusion_filepath",
                        help="Full file path to the binary exclusion data 1= exclude, for now 255=nodata.",
                        required=False, type=str)
    parser.add_argument("-ref", "--reference_file",
                        help="Full file path to the validation dataset (.tif or .shp, in any projection)",
                        required=True, type=str)
    parser.add_argument("-out", "--output_raster",
                        help="Full file path to the final difference raster", required=True, type=str)
    parser.add_argument("-csv", "--output_csv",
                        help="Full file path to the csv results", required=False, type=str)
    parser.add_argument("-del", "--delete_tmp",
                        help="Option to delete temporary files.", required=False, type=bool)

    # collect inputs
    args = parser.parse_args()
    input_raster_filepath = args.input_filepath
    exclusion_filepath = args.exclusion_filepath
    validation_filepath = args.reference_file
    output_raster_filepath = args.output_raster
    output_csv_filepath = args.output_csv
    delete_tmp = args.delete_tmp

    # define output names
    out_dirpath, out_raster_filename = os.path.split(output_raster_filepath)
    base = os.path.splitext(out_raster_filename)[0]
    reproj_shp_filepath = base + '_reproj_input_vector.shp'
    rasterized_shp_filepath = base + '_rasterize_input_vector.tif'

    # set default option
    if delete_tmp is None:
        delete_tmp = False

    run(ras_data_filepath=input_raster_filepath, ref_data_filepath=validation_filepath, out_dirpath=out_dirpath,
        diff_ras_out_filename=out_raster_filename, v_reprojected_filename=reproj_shp_filepath,
        v_rasterized_filename=rasterized_shp_filepath, out_csv_filename=output_csv_filepath,
        ex_filepath=exclusion_filepath, delete_tmp_files=delete_tmp)


if __name__ == '__main__':
    command_line_interface()
