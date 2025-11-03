from odc.geo.geobox import GeoBox
from odc.geo.xr import rasterize
from odc.geo.geom import Geometry
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
import xarray as xr
import numpy as np
from pathlib import Path


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
    rasterized = rasterized.rio.reproject_match(riox_arr)

    return rasterized


def ensure_path(path_or_str):
    if isinstance(path_or_str, Path):
        return path_or_str
    else:
        return Path(path_or_str)
