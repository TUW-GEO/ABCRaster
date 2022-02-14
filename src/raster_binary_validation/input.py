from equi7grid.equi7grid import Equi7Grid
from osgeo import gdal, ogr, osr
import numpy as np
import os


def get_equi7grid_geotags(tile, sres=20, continent='EU'):
    """ Retrieve spatial details of an Equi7grid tile. """
    grid = Equi7Grid(sres).subgrids[continent]
    tile_geotags = grid.tilesys.create_tile(name=tile).get_geotags()
    gt, sref = tile_geotags['geotransform'], tile_geotags['spatialreference']

    return gt, sref


def reproject_vec(layer, out_sref, v_reprojected_filepath='tmp.shp'):
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


def rasterize2equi7(vec_ds, tile, sres, cont, v_reprojected_filepath='tmp.shp', bg_absence=True):
    """
    Transforms a vector to a raster layer and resamples it to teh Equi7grid.

    Parameters
    ----------
    vec_ds: ogr vector layer
        Vector layer to be rasterized.
    tile: str
        Equi7grid tile.
    sres: int
        Spatial sampling of the Equi7grid tile.
    cont: str
        Equi7grid continent code.
    v_reprojected_filepath: str
        Path of the temporary reprojected vector layer.
    bg_absence: bool, optional
        Option to limit comparison to the vector layer extent (default: True).

    Returns
    -------
    out_ras_data: numpy.array
        Resulting raster array.
    """
    # TODO: solve issue with missing output path parameter
    gt, sref = get_equi7grid_geotags(tile, sres=sres, continent=cont)
    t = int(tile[-1]) * 100000
    rasterYSize = rasterXSize = int(t / sres)

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
        reproject_vec(vec_layer, out_sref, v_reprojected_filepath)
        print('done... reprojecting')

        # returning layer object and using that to rasterize results it segmentation fault.
        driver = ogr.GetDriverByName('ESRI Shapefile')
        reProjDataSet = driver.Open(v_reprojected_filepath)
        vec_layer = reProjDataSet.GetLayer()

    # burn polygons as presence,
    # TODO: add elif to get attribute.
    gdal.RasterizeLayer(out_ds, [1], vec_layer, burn_values=[1])
    outBand = out_ds.GetRasterBand(1)
    outBand.SetNoDataValue(255)
    out_ras_data = outBand.ReadAsArray()

    output = np.empty_like(out_ras_data, dtype=np.uint8)
    maxRow, maxCol = output.shape
    output[:] = 255  # default nodata

    if bg_absence:
        # set no data as 0 or absence
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
        reproject_vec(vec_layer, out_sref, v_reprojected_filepath)
        print('done... reprojecting')

        # returning layer object and using that to rasterize results it segmentation fault.
        driver = ogr.GetDriverByName('ESRI Shapefile')
        reProjDataSet = driver.Open(v_reprojected_filepath)
        vec_layer = reProjDataSet.GetLayer()

    # burn polygons as presence,
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