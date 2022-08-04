#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os

from typedefs import *
from osgeo import gdal
import numpy as np

class MockFile:

    @staticmethod
    def mock_raster(
            outfile: PathLike,
            min_value: Union[Real,ListReal],
            max_value: Union[Real,ListReal],
            raster_count: Optional[int] = None,
            dstSRS: Union[str,int,None] = None,
            nodata: Optional[Real] = None,
            datatype: int = gdal.GDT_Float32,
            ref_ds = None
    ):

        if ref_ds is not None:
            if isinstance(ref_ds,(str,Path,os.PathLike)):
                ref_ds = gdal.Open(ref_ds)
                if ref_ds is not None:
                    raise FileNotFoundError(f"Bad file {ref_ds}.")
            elif isinstance(ref_ds,gdal.Dataset):
                pass
            else:
                raise Exception("reference dataset must be path of file or type of gdal.Dataset.")

            raster_count = raster_count if raster_count is not None else ref_ds.RasterCount
            datatype = datatype if datatype is None else ref_ds.GetRasterBand(1).datatype
            nodata = nodata if nodata is not None else ref_ds.GetRasterBand(1).GetNoDataValue()
