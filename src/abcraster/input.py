from odc.geo.geobox import GeoBox
from odc.geo.xr import rasterize
from odc.geo.geom import Geometry
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
import xarray as xr
import numpy as np
import os


def rasterize_to_rioxarray(vec_gpf: gpd.GeoDataFrame, riox_arr: xr.DataArray) -> xr.DataArray:
    bounds = riox_arr.rio.bounds()
    shape = (riox_arr.shape[1], riox_arr.shape[2])
    input_geobox = GeoBox.from_bbox(bounds, riox_arr.rio.crs, shape=shape)
    multipolygon = unary_union(vec_gpf['geometry'].tolist())
    if multipolygon.geom_type == 'Polygon':
        multipolygon = MultiPolygon([multipolygon])
    rasterized = rasterize(
        poly=Geometry(multipolygon, crs=vec_gpf.crs),
        how=input_geobox,
        value_inside=True
    )
    rasterized = rasterized.astype(np.uint8)
    rasterized = rasterized.expand_dims(band=[1], axis=0)

    return rasterized


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
