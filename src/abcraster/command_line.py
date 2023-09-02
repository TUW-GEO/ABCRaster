from abcraster.base import run
from abcraster.metrics import metrics
import argparse
import os


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
    base = os.path.splitext(out_raster_filename)[0]
    reproj_shp_filepath = base + '_reproj_input_vector.shp'
    rasterized_shp_filepath = base + '_rasterize_input_vector.tif'

    # set default option
    if delete_tmp is None:
        delete_tmp = False

    run(ras_data_filepaths=input_raster_filepaths, ref_data_filepath=validation_filepath, out_dirpath=out_dirpath,
        diff_ras_out_filename=out_raster_filename, v_reprojected_filename=reproj_shp_filepath,
        v_rasterized_filename=rasterized_shp_filepath, out_csv_filename=output_csv_filepath,
        ex_filepath=exclusion_filepath, delete_tmp_files=delete_tmp,
        sampling=sampling, samples_filepath=samples_filepath, metrics_list=metrics_list)


if __name__ == '__main__':
    command_line_interface()
