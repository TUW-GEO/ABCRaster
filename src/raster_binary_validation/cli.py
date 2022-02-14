from raster_binary_validation.worker import run
import argparse


def command_line_interface():
    """ Command line interface to perform a validation based on vector-based reference data. """

    parser = argparse.ArgumentParser(
        description="Simple Binary Validation Workflow. Initially designed to validate floods")
    parser.add_argument("-in", "--input_filepath",
                        help="Full file path to the binary raster data 1= presence, 0=absennce, for now 255=nodata.",
                        required=True, type=str)
    parser.add_argument("-ex", "--exclusion_filepath",
                        help="Full file path to the binary exclusion data 1= exclude, for now 255=nodata.",
                        required=False, type=str)
    parser.add_argument("-ref", "--reference_shpfile",
                        help="Full file path to the validation shapefile (in any projection)", required=True, type=str)
    parser.add_argument("-out", "--output_raster",
                        help="Full file path to the final difference raster", required=False, type=str)
    parser.add_argument("-csv", "--output_csv",
                        help="Full file path to the csv results", required=False, type=str)

    args = parser.parse_args()
    input_raster_filepath = args.input_filepath
    exclusion_filepath = args.exclusion_filepath
    validation_vector_filepath = args.reference_shpfile
    output_raster_filepath = args.output_raster
    output_csv_filepath = args.output_csv

    if output_raster_filepath is None:
        output_raster_filepath = 'validation-results.tif'
    if output_csv_filepath is None:
        output_csv_filepath = 'validation-results.csv'

    base = output_raster_filepath.split('.')[0]
    reproj_shp_filepath = base + '_reproj_input_vector.shp'
    rasterized_shp_filepath = base + '_rasterize_input_vector.tif'

    run(input_raster_filepath, validation_vector_filepath, diff_ras_out_filepath=output_raster_filepath,
        v_reprojected_filepath=reproj_shp_filepath, v_rasterized_filepath=rasterized_shp_filepath,
        out_csv_filepath=output_csv_filepath, ex_filepath=exclusion_filepath)


if __name__ == '__main__':
    command_line_interface()
