# Copyright (c) 2020, Vienna University of Technology (TU Wien), Department
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


import os, argparse, math
import gdal, ogr, osr
import numpy as np
import pandas as pd


def openReadFile(path):
    """
    Helper function to read the first band of a raster file.
    :param path: str Path to file.
    :return: numpy array, gdal geotransform, osr projection
    """
    gdal.AllRegister()
    inDS = gdal.Open(path)
    inBand = inDS.GetRasterBand(1)
    data = inBand.ReadAsArray()
    gt = inDS.GetGeoTransform()
    sref = inDS.GetProjection()
    inBand = None
    inDS = None
    return data, gt, sref


def saveTiffFile(outpath, data, nodata=255, gt=None, sref=None):
    """
    Helper function to save a single band, byte type raster file to Tiff. Used to save difference/change raster.
    :param outpath: str path to output file
    :param data: numpy array
    :param nodata: no data value
    :param gt: gdal geotransform
    :param sref: osr projection
    :return:
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


def get_equi7grid_geotags(tile, sres=20, continent='EU'):
    from equi7grid.equi7grid import Equi7Grid

    grid = Equi7Grid(sres).subgrids[continent]
    tile_geotags = grid.tilesys.create_tile(name=tile).get_geotags()
    gt, sref = tile_geotags['geotransform'], tile_geotags['spatialreference']

    return gt, sref


def reproject_vec(layer, out_sref, v_reprojected_filepath='tmp.shp'):
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


def rasterize2Equi7(vec_ds, tile, sres, cont, v_reprojected_filepath='tmp.shp', bg_absence=True):
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
        row_start, row_end, col_start, col_end = boundingBoxToOffsets(v_ext, gt)
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
    # need to reproject!
    # https://pcjericks.github.io/py-gdalogr-cookbook/projection.html
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
        row_start, row_end, col_start, col_end = boundingBoxToOffsets(v_ext, gt)
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


def boundingBoxToOffsets(bbox, geot):
    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
    return [row1, row2, col1, col2]


def validate(data, val_data, exclusion=None, flood_nodata=255, val_nodata=255):
    """
    Runs validation on aligned numpy arrays.
    :param data:
    :param val_data:
    :param exclusion:
    :param flood_nodata:
    :param val_nodata:
    :return: tuple Numpy Array- difference raster, Mask - nodata,"User's Accuracy/Precision", "Producer's Accuracy/Recall", 'Commission Error', 'Omission Error', 'Critical Success Index', 'F1', 'Kappa', 'Accuracy'

    """

    res = 1 + (2 * data) - val_data
    res[data == flood_nodata] = 255
    res[val_data == val_nodata] = 255

    if exclusion is not None:
        res[exclusion == 1] = 255
        data[exclusion == 1] = 255  # applying exclusion, setting exclusion pixels as no data
    idx = np.logical_and(val_data != 255, data != 255)  # index removing no data from comparison

    TP = np.sum(res == 2)
    TN = np.sum(res == 1)
    FN = np.sum(res == 0)
    FP = np.sum(res == 3)
    print(np.array([[TP, FP], [FN, TN]]))

    # removing dependency on scikit
    # valList = val_data[idx].ravel()
    # modList = data[idx].ravel()
    # CM = confusion_matrix(valList, modList)
    # print(CM)
    # TN, FP, FN, TP = CM.astype('float').ravel()

    # calculating metrics
    Po = A = (TP + TN) / (TP + TN + FP + FN)
    Pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / (TP + TN + FP + FN) ** 2
    K = (Po - Pe) / (1 - Pe)
    Precision = UA = TP / (TP + FP)
    Recall = PA = TP / (TP + FN)  # accuracy:PP2 as defined in ACube4Floods 5.1
    CSI = TP / (TP + FP + FN)
    F1 = (2 * TP) / (2 * TP + FN + FP)
    Ce = FP / (FP + TP)  # inverse of precision
    Oe = FN / (FN + TP)  # inverse of recall
    P = math.exp(FP / ((TP + FN) / math.log(0.5)))  # penalization as defined in ACube4Floods 5.1
    SR = PA - (1 - P)  # Success rate as defined in ACube4Floods 5.1

    print("User's Accuracy/Precision: %f" % (UA))
    print("Producer's Accuracy/Recall/PP2: %f" % (PA))
    print("Critical Success In: %f" % (CSI))
    print("F1: %f" % (F1))
    print("commission error: %f" % (Ce))
    print("omission error: %f" % (Oe))
    print("total accuracy: %f" % (A))
    print("kappa: %f" % (K))
    print("Penalization Function: %f" % (P))
    print("Success Rate: %f" % (SR))

    return res, idx, UA, PA, Ce, Oe, CSI, F1, SR, K, A


def validate_raster(in_filename, val_filename, results_filename, exclusion_filename=None):
    """
    Validates raster reference file with raster map. Assumes files are aligned, same projection and extent.
    :param floodFile:
    :param val_filename:
    :param results_base_path:
    :return:
    """
    data, gt, sref = openReadFile(in_filename)
    val_data, gt_val, sref_val = openReadFile(val_filename)

    if exclusion_filename is None:
        exclusion_data = None
    else:
        exclusion_data = openReadFile(exclusion_filename)[0]
        # need to add projection check, assume same as validation data

    res, idx, UA, PA, Ce, Oe, CSI, F1, SR, K, A = validate(data, val_data, exclusion=exclusion_data, flood_nodata=255,
                                                           val_nodata=255)

    saveTiffFile(results_filename, res, nodata=255, gt=gt, sref=sref)

    return file_name, UA, PA, Ce, Oe, CSI, F1, SR, K, A


def main(ras_data_filepath, v_val_data_filepath, diff_ras_out_filepath='val.tif',
         v_reprojected_filepath='reproj_tmp.shp', v_rasterized_filepath='rasterized_val.tif',
         out_csv_filepath='val.csv', ex_filepath=None):
    """
    Runs the validation with vector data input (presence = 1, absence=0).
    :param v_val_data_filepath:
    :param ras_data_filepath:
    :param diff_ras_out_filepath:
    :param v_reprojected_filepath:
    :param v_rasterized_filepath:
    :param out_csv_filepath:
    :return:
    """

    vec_ds = ogr.Open(v_val_data_filepath)
    flood_data, gt, sref = openReadFile(ras_data_filepath)

    if ex_filepath is None:
        ex_data = None
    else:
        ex_data = openReadFile(ex_filepath)[0]

    print('rasterizing')
    val_data = rasterize(vec_ds, v_rasterized_filepath, flood_data, gt, sref,
                         v_reprojected_filepath=v_reprojected_filepath)
    print('done ... rasterizing')

    print('start validation')
    res, idx, UA, PA, Ce, Oe, CSI, F1, SR, K, A = validate(flood_data, val_data, exclusion=ex_data, flood_nodata=255,
                                                           val_nodata=255)

    # save results
    res = res.astype(np.uint8)
    res[~idx] = 255
    saveTiffFile(diff_ras_out_filepath, res, nodata=255, gt=gt, sref=sref)

    #
    dat = [['result 1', UA, PA, Ce, Oe, CSI, F1, SR, K, A]]
    df = pd.DataFrame(dat,
                      columns=['file', "User's Accuracy/Precision", "Producer's Accuracy/Recall", 'Commission Error',
                               'Omission Error', 'Critical Success Index', 'F1', 'Success Rate', 'Kappa', 'Accuracy'])
    df.to_csv(out_csv_filepath)
    print('end validation')


if __name__ == '__main__':
    # reprojection needs the gdal environment paths set!
    os.environ['GDAL_DATA'] = r'/home/mtupas/anaconda3/envs/s1_floodmapping/share/gdal'
    os.environ['PROJ_LIB'] = r'/home/mtupas/anaconda3/envs/s1_floodmapping/share/proj'

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

    main(input_raster_filepath, validation_vector_filepath, diff_ras_out_filepath=output_raster_filepath,
         v_reprojected_filepath=reproj_shp_filepath, v_rasterized_filepath=rasterized_shp_filepath,
         out_csv_filepath=output_csv_filepath, ex_filepath=exclusion_filepath)
