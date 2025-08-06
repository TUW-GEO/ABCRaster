from osgeo import gdal, ogr, osr
from rasterio import features
import geopandas as gpd
import numpy as np
import rasterio
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


def raster_reproject(fpath, sref, res, out_dirpath, reproj_add_str):
    """
    Reprojects first raster dataset to the spatial reference of the second raster dataset.

    Parameters
    ----------
    fpath: str
        Raster file path to be reprojected.
    sref: str
        Aimed spatial reference as WKT.
    res: int
        Aimed spatial resolution.
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

    warp = gdal.Warp(out_reproj_path, fpath, dstSRS=sref, resampleAlg='near', xRes=res, yRes=res)
    warp = None  # Closes the files

    return out_reproj_path


def rasterize_by_raster(vec_path, out_ras_path, ras_path, nodata=255, clip2bbox=False):
    """
    Transforms a vector to a raster layer based on an example raster file.

    Parameters
    ----------
    vec_path: str
        Path of the vector layer to be rasterized.
    out_ras_path: str
        Path of the output raster layer.
    ras_path: str
        Path of exemplary raster array.
    nodata: int
        No data value of output raster. (default: 255).
    clip2bbox: boolean
        Assign nodata (255) to area outside the vector bounding box.

    Returns
    -------
    rasterized: numpy.array
        Resulting raster array.
    """

    raster = rasterio.open(ras_path)
    rasterize(vec_path=vec_path, out_ras_path=out_ras_path, out_sref=raster.crs, out_shape=raster.shape,
              out_transform=raster.transform, nodata=255, clip2bbox=False)
    raster.close()

    return rasterized


def rasterize(vec_path, out_ras_path, out_sref, out_shape, out_transform, nodata=255, clip2bbox=False):
    """
    Transforms a vector to a raster layer.

    Parameters
    ----------
    vec_path: str
        Path of the vector layer to be rasterized.
    out_ras_path: str
        Path of the output raster layer.
    out_sref: rasterio.crs
        Output projection.
    out_shape: tuple
        Rows and columns of output raster file.
    out_transform: tuple
        Output geospatial tranform information.
    nodata: int, optional
        No data value of output raster. (default: 255).
    clip2bbox: boolean, optional
        Assign nodata (255) to area outside the vector bounding box.

    Returns
    -------
    rasterized: numpy.array
        Resulting raster array.
    """

    # Read and transform vector layer
    vector = gpd.read_file(vec_path)
    vector = vector.to_crs(out_sref)
    geom = [shapes for shapes in vector.geometry]

    # Rasterize vector using the shape and coordinate system of the raster
    rasterized = features.rasterize(geom,
                                    out_shape=out_shape,
                                    fill=0,
                                    out=None,
                                    transform=out_transform,
                                    all_touched=False,
                                    default_value=1,
                                    dtype = None)

    if clip2bbox:
        temp_raster = np.empty_like(rasterized)
        temp_raster[:] = nodata

        maxRow, maxCol = temp_raster.shape
        v_ext = vector.total_bounds
        row_start, row_end, col_start, col_end = bounding_box2offsets(v_ext, out_transform)

        # overflow check
        row_end = min([maxRow - 1, row_end])
        col_end = min([maxCol - 1, col_end])
        row_start = max([0, row_start])
        col_start = max([0, col_start])

        temp_raster[row_start:row_end, col_start:col_end] = rasterized[row_start:row_end, col_start:col_end]
        rasterized = temp_raster

    # write output
    with rasterio.open(
        out_ras_path, "w",
        driver="GTiff",
        crs=out_sref,
        transform=out_transform,
        dtype=rasterio.uint8,
        count=1,
        width=out_shape[1],
        height=out_shape[0]) as dst:
            dst.write(rasterized, indexes=1)

    return rasterized


def bounding_box2offsets(bbox, geot):
    """ Converts geotransform definition to bounding box. """

    col1 = int((bbox[0] - geot[2]) / geot[0])
    col2 = int((bbox[2] - geot[2]) / geot[0]) + 1
    row1 = int((bbox[3] - geot[5]) / geot[4])
    row2 = int((bbox[1] - geot[5]) / geot[4]) + 1

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
