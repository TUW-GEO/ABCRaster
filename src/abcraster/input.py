from veranda.raster.mosaic.geotiff import GeoTiffFile
from geospade.raster import RasterGeometry
from osgeo import gdal, ogr, osr
import numpy as np
import os


def vec_reproject(layer, out_sref, v_reprojected_filepath='tmp.shp'):
    """
    Reprojects a vector layer to a different projection.

    Parameters
    ----------
    layer: ogr object
        Vector layer which can be initialized by ogr.Open(...).
    out_sref: osr.SpatialReference
        Requested spatial projection.
    v_reprojected_filepath: str, optional
        Path of the temporary projected vector layer (default: tmp.shp).
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')
    in_sref = layer.GetSpatialRef()
    coordTrans = osr.CoordinateTransformation(in_sref, out_sref)

    # create the output layer

    if os.path.exists(v_reprojected_filepath):
        driver.DeleteDataSource(v_reprojected_filepath)
    outDataSet = driver.CreateDataSource(v_reprojected_filepath)
    outLayer = outDataSet.CreateLayer("tmp", geom_type=ogr.wkbMultiPolygon)  # to be changed to match input

    # add fields
    inLayerDefn = layer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = layer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = layer.GetNextFeature()

    out_sref.MorphToESRI()
    out_prj_filepath = v_reprojected_filepath.strip().split('.')[0] + '.prj'
    file = open(out_prj_filepath, 'w')
    file.write(out_sref.ExportToWkt())
    file.close()

    outLayer.SyncToDisk()
    outDataSet = None
    outLayer = None


def raster_reproject(fpath, sref, out_dirpath, reproj_add_str):
    """
    Reprojects first raster dataset to the spatial reference of the second raster dataset.

    Parameters
    ----------
    fpath: str
        Raster file path to be reprojected.
    sref: str
        Aimed spatial reference as WKT.
    out_dirpath: str
        Directory to which the output will be written to.
    reproj_add_str: str
        String which will be added to the filename of vector or raster files after reprojecting.

    Returns
    -------
    out_reproj_path: str
        Path of the reprojected raster file.
    """

    out_reproj_path = update_filepath(fpath, add_str=reproj_add_str, new_root=out_dirpath)

    warp = gdal.Warp(out_reproj_path, fpath, dstSRS=sref)
    warp = None  # Closes the files

    return out_reproj_path


def raster_intersect(geom1, geom2):
    """
    Retrieves the intersecting geometry from two raster datasets.

    Parameters
    ----------
    geom1: RasterGeometry
        Geometry of the first raster file.
    geom2: RasterGeometry
        Geometry of the second raster file.

    Returns
    -------
    intersection: RasterGeometry
        Geometry of the intersection.
    """

    if not geom1.intersects(geom2):
        raise ValueError("Input data does not intersect reference data.")

    intersection = geom1.slice_by_geom(geom2, sref=geom2.sref.wkt)
    intersection = intersection.slice_by_geom(geom1, sref=geom1.sref.wkt)

    return intersection


def raster_read_from_polygon(fpath, geom):
    """
    Reads the area defined by the passed polygon from a raster file.

    Parameters
    ----------
    fpath: str
        Path of the input raster files.
    geom: RasterGeometry
        Polygon to read from the raster file.

    Returns
    -------
    arr: np.ndarray
        Resulting numpy array.
    """

    raster_data = GeoTiffFile.from_filepaths(fpath)
    raster_data.select_polygon(geom.boundary, sref=geom.sref_wkt, inplace=True)
    arr = raster_data.read()[1]
    raster_data.close()

    return arr


def rasterize(vec_ds, out_ras_path, ras_data, gt, sref, v_reprojected_filepath='tmp.shp', bg_absence=True):
    """
    Transforms a vector to a raster layer.

    Parameters
    ----------
    vec_ds: ogr vector layer
        Vector layer to be rasterized.
    out_ras_path: str
        Path of the output raster layer.
    ras_data: numpy.array
        Exemplary raster array.
    gt: tuple
        GDAL geotransform definition of output raster.
    sref: osr.SpatialReference
        Spatial projection of the output raster layer.
    v_reprojected_filepath: str
        Path of the temporary reprojected vector layer.
    bg_absence: bool, optional
        Option to limit comparison to the vector layer extent (default: True).

    Returns
    -------
    out_ras_data: numpy.array
        Resulting raster array.
    """

    rasterYSize, rasterXSize = ras_data.shape
    gtiff_driver = gdal.GetDriverByName("GTiff")
    out_ds = gtiff_driver.Create(out_ras_path, rasterXSize, rasterYSize, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(sref)
    vec_layer = vec_ds.GetLayer()

    out_sref = osr.SpatialReference()
    out_sref.ImportFromWkt(sref)

    in_sref = vec_layer.GetSpatialRef()

    if in_sref != out_sref:
        print('reprojecting...')
        vec_reproject(vec_layer, out_sref, v_reprojected_filepath)
        print('done... reprojecting')

        # returning layer object and using that to rasterize results it segmentation fault.
        driver = ogr.GetDriverByName('ESRI Shapefile')
        reProjDataSet = driver.Open(v_reprojected_filepath)
        vec_layer = reProjDataSet.GetLayer()

    # burn polygons as presence,
    # TODO: add attribute filtering for presence and absence
    # TODO: add elif to get attribute.
    gdal.RasterizeLayer(out_ds, [1], vec_layer, burn_values=[1])
    outBand = out_ds.GetRasterBand(1)
    outBand.SetNoDataValue(255)
    out_ras_data = outBand.ReadAsArray()

    output = np.empty_like(out_ras_data, dtype=np.uint8)
    maxRow, maxCol = output.shape
    output[:] = 255  # default nodata
    if bg_absence:
        v_ext = vec_layer.GetExtent()
        row_start, row_end, col_start, col_end = bounding_box2offsets(v_ext, gt)
        # overflow check
        row_end = min([maxRow - 1, row_end])
        col_end = min([maxCol - 1, col_end])
        row_start = max([0, row_start])
        col_start = max([0, col_start])
        output[row_start:row_end, col_start:col_end] = 0
    output[out_ras_data == 1] = 1
    outBand.WriteArray(output)
    out_ras_data = output

    outBand = None
    out_ds = None
    return out_ras_data


def bounding_box2offsets(bbox, geot):
    """ Converts GDAL'S geotransform definition to bounding box. """
    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
    return [row1, row2, col1, col2]


def update_filepath(fpath, add_str=None, new_ext=None, new_root=None):
    """
    Updates the filename in a given path by adding a string at the end and optionally update the file extension.

    Parameters
    ----------
    fpath: str
        Input file path.
    add_str: str, optional
        String to be added to the filename (default: None).
    new_ext: str, optional
        New file extension (default: None).
    new_root: str, optional
        New directory for the file (default: None).

    Returns
    -------
    updated_filepath: str
        Updated file path.
    """

    orig_dirpath, orig_fname = os.path.split(fpath)
    orig_name, orig_ext = os.path.splitext(orig_fname)
    orig_ext = orig_ext.replace('.', '')

    add_str = '' if add_str is None else '_' + add_str
    ext = new_ext.replace('.', '') if new_ext is not None else orig_ext
    dir_path = new_root if new_root is not None else orig_dirpath

    return os.path.join(dir_path, orig_name + add_str + '.' + ext)
