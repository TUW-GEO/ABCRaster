from osgeo import gdal


def save_file(outpath, data, nodata=255, gt=None, sref=None):
    """
    Writes array to GeoTiff file.

    Parameters
    ----------
    outpath: str
        Path of the output file.
    data: numpy.array
        Array to be written to disk.
    nodata: int, optional
        No data value (default: 255).
    gt: tuple, optional
        GDAL geotransform defintion.
    sref: osr.SpatialReference
        Spatial reference object.
    """

    rows, cols = data.shape
    driver = gdal.GetDriverByName('GTiff')
    outDS = driver.Create(outpath, cols, rows, 1, gdal.GDT_Byte, ["COMPRESS=LZW"])
    band = outDS.GetRasterBand(1)
    outDS.SetGeoTransform(gt)
    outDS.SetProjection(sref)
    band.SetNoDataValue(nodata)
    band.WriteArray(data)
    band.FlushCache()
    band = None  # dereference band to avoid gotcha described previously
    outDS = None  # save, close