from veranda.raster.mosaic.geotiff import GeoTiffReader
from geospade.raster import RasterGeometry
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

    raster_data = GeoTiffReader.from_filepaths([fpath])
    raster_data.select_polygon(geom.boundary, sref=geom.sref, inplace=True)
    raster_data.read()
    raster_data.close()

    return raster_data.data_view.to_array().to_numpy()[0, 0, ...]


def rasterize(vec_path, out_ras_path, ras_path):
    """
    Transforms a vector to a raster layer.

    Parameters
    ----------
    vec_path: str
        Path of the vector layer to be rasterized.
    out_ras_path: str
        Path of the output raster layer.
    ras_path: str
        Path of exemplary raster array.

    Returns
    -------
    rasterized: numpy.array
        Resulting raster array.
    """

    # Open example raster
    raster = rasterio.open(ras_path)

    # Read and transform vector layer
    vector = gpd.read_file(vec_path)
    vector = vector.to_crs(raster.crs)
    geom = [shapes for shapes in vector.geometry]

    # Rasterize vector using the shape and coordinate system of the raster
    rasterized = features.rasterize(geom,
                                    out_shape=raster.shape,
                                    fill=0,
                                    out=None,
                                    transform=raster.transform,
                                    all_touched=False,
                                    default_value=1,
                                    dtype = None)

    # write output
    with rasterio.open(
        out_ras_path, "w",
        driver="GTiff",
        crs=raster.crs,
        transform=raster.transform,
        dtype=rasterio.uint8,
        count=1,
        width=raster.width,
        height=raster.height) as dst:
            dst.write(rasterized, indexes=1)

    return rasterized


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
