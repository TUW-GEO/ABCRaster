import argparse
import rioxarray
import numpy as np
from abcraster.input import ensure_path


def gen_random_sample(num_samples, data, ref, nodata=255, exclusion=None):
    """
    Creates a numpy array mask of randomly selected samples given a reference and input classified data.

    Parameters
    ----------
    num_samples: list, tuple or int
        - number of samples where the index of the iterable matches class encoding e.g. [M, N] --> M samples for class 0
        and N samples for class 1,
        - value in singletons i.e. [N] are treated as the total number of samples for all classes and would be equally
        distributed hence should be divisible by the number of classes. num classes detected from reference data values.
        - for non-stratified sampling pass an int.
    data: np.ndarray
        (binary) classified data, assumes uint encoded.
    ref: np.ndarray
        reference data, assumes same data format and projection as (input) data
    nodata: int, optional
        nodata value, assumes the same for both reference and input data
    exclusion: np.ndarray
        exclusion mask, encoded as 1 or True to be removed from analysis

    Returns
    -------
    samples: np.ndarray
        Integer array, which contains the class id for the sample points and no data for other pixels.
    """

    # performing raster shape check
    if data.shape != ref.shape:
        raise RuntimeError("Dimension of input and reference rasters are not the same.")

    # initialize samples
    samples = ~np.ones_like(data, dtype=bool)
    nodata_mask = (ref == nodata) | (data == nodata)

    # check for exclusion data, then added to nodata mask if exists
    if exclusion is not None:
        nodata_mask = nodata_mask | (exclusion == 1)

    # stratified sampling
    if isinstance(num_samples, list) or isinstance(num_samples, tuple):
        # define number of samples per class
        num_class = np.max(ref[ref != nodata]) + 1
        if len(num_samples) == 1:
            if num_samples % num_class != 0:
                raise ValueError("If num_samples list/tuple is a singleton, it should be divisible by num_class.")
            num_samples = num_samples * num_class
            num_samples = [int(sample/num_class) for sample in num_samples]  # samples evenly distributed per class
        elif len(num_samples) != num_class:
            raise ValueError("Dimension of samples do not correspond to the number of classes.")

        # select samples
        for class_id in range(num_class):
            class_sel = random_conditional_selection(arr=ref, num=num_samples[class_id], apriori_mask=nodata_mask,
                                                     cond=class_id)
            samples[class_sel] = True

    # non-stratified sampling
    elif isinstance(num_samples, int):
        sel = random_conditional_selection(arr=ref, num=num_samples, apriori_mask=nodata_mask)
        samples[sel] = True

    else:
        raise ValueError("Unknown type for the number of samples variable.")

    # apply class allocation
    samples = np.where(samples, ref, 255)
    samples = samples.astype(np.uint8)

    return samples


def random_conditional_selection(arr, num, apriori_mask, cond=None):
    """
    Selects indices randomly from an array, only considering a specific value within the array.

    Parameters
    ----------
    arr: np.ndarray
        Array which includes integer values.
    num: int
        Number of indices which should be retrieved randomly.
    apriori_mask: np.ndarray
        Boolean array showing the pixels which should be excluded a priori.
    cond: int, optional
        Specific integers to which the selection should be limited to or no limitation (=None).

    Returns
    -------
    idx: np.ndarray
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


def main_sampling(num_samples, data_path, ref_path, out_path, nodata=255, ex_path=None):
    """
    Runs the sampling step without the accuracy assessment. Writes the sampling raster to file.

    Parameters
    ----------
    num_samples: list, tuple or int
        - number of samples where the index of the iterable matches class encoding e.g. [M, N] --> M samples for class 0
        and N samples for class 1,
        - value in singletons i.e. [N] are treated as the total number of samples for all classes and would be equally
        distributed hence should be divisible by the number of classes. num classes detected from reference data values.
        - for non-stratified sampling pass an int.
    data_path: str or Path
        file path to classified raster data (*.tif), uint encoded
    ref_path: str or Path
        file path to reference raster data (*.tif), unit encoded
    out_path: str or Path
        file path to output sampling raster data (*.tif), unit encoded
    nodata: int (default == 255)
        no data value.
    ex_path: str (default == None)
        file path to exclusion raster data (*.tif), assumed to be encoded with 1 or True values as pixels to be excluded

    Returns
    -------
    None
    """

    data_path = ensure_path(data_path)
    ref_path = ensure_path(ref_path)
    out_path = ensure_path(out_path)

    input_data = rioxarray.open_rasterio(data_path)
    ref_data = rioxarray.open_rasterio(ref_path)
    ref_data = ref_data.rio.reproject_match(input_data)
    if ex_path is not None:
        ex_data = rioxarray.open_rasterio(ex_path).values
    else:
        ex_data = None

    samples = gen_random_sample(num_samples, input_data.values, ref_data.values, nodata=nodata,
                                exclusion=ex_data)
    out_ds = input_data.copy(deep=True)
    out_ds.values = samples
    out_ds.rio.to_raster(out_path)


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
                        required=False, type=str, default='None')
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
                        help="Stratification based on reference data.", required=False, type=bool, default=True)

    # collect inputs
    args = parser.parse_args()
    data_path = args.input_filepath
    exclusion_filepath = args.exclusion_filepath
    if exclusion_filepath == 'None':
        exclusion_filepath = None
    ref_path = args.reference_file
    out_path = args.output_raster
    n = args.num_samples
    nodata = args.nodata
    strat = args.stratify

    if strat:
        num_samples = [int(n/2), int(n/2)]  # use case is binary classified data
    else:
        num_samples = n

    main_sampling(num_samples, data_path, ref_path, out_path, nodata=nodata, ex_path=exclusion_filepath)


if __name__ == '__main__':
    command_line_interface()
