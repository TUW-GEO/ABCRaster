import rasterio


def write_raster(arr, out_filepath, sref, gt, nodata=255):
    profile = {'driver': 'GTiff', 'height': arr.shape[0], 'width': arr.shape[1], 'count': 1, 'dtype': arr.dtype,
               'crs': sref, 'transform': gt, 'nodata': nodata}
    with rasterio.open(out_filepath, 'w', **profile) as dst:
        dst.write(arr, 1)
