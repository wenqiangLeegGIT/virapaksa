#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import re

import numpy as np
from typedefs import *
from osgeo import gdal,osr
import gdalnumeric

class MODISBaseAlg:

    def _reproject(
            self,
            hdf_file   :PathLike,
            data_keys  :Union[ListStr,str],
            dstSRS     :Union[int,str],
            dstRes     :Real,
            dstNoData  :Union[Real,None]=None,
            dstDataType:int = gdal.GDT_Float32,
            resampleAlg:int = gdal.GRA_NearestNeighbour,
            calibration:bool = False,
            to_single  :bool = False
    ):
        hdfds = gdal.Open(str(hdf_file))
        assert hdfds is not None,f"Bad file {hdf_file}."
        subdatasets = hdfds.GetSubDatasets()
        if len(subdatasets) == 0:
            raise Exception(f"No sub datasets were found in {hdf_file}.")
        _subs = []
        if isinstance(data_keys,(str,Path,os.PathLike)):
            _subs = [data_keys]
        elif isinstance(data_keys,list):
            _subs = data_keys
        else:
            raise Exception("'sub_datasets' type must be ListStr or str. ")

        srs = osr.SpatialReference()
        if isinstance(dstSRS,str):
            _ogr_err = srs.ImportFromWkt(dstSRS)
            if not _ogr_err:
                raise Exception(f"Invalid WKT. {dstSRS}")
        elif isinstance(dstSRS,int):
            _ogr_err = srs.ImportFromEPSG(dstSRS)
            if not _ogr_err:
                raise Exception(f"Invalid EPSG code. {dstSRS}")
        else:
            raise Exception("dstSRS only support 'EPSG code' and 'wkt'.")
        del srs

        key_datasets = {}
        for sub in subdatasets:
            for key in _subs:
                if re.search(key,sub[0]):
                    key_datasets[key] = sub[0]
        if len(key_datasets) == 0:
            raise Exception(f"{_subs} not found in file {hdf_file}.")
        root = Path(hdf_file)
        outfiles = []
        for key,ds in key_datasets.items():
            out = root.with_suffix(f".{key}.tif")
            if calibration:
                ds = self._calibaration(ds,output_type=dstDataType,dst_nodata=dstNoData)
            gdal.Warp(str(out),ds,dstSRS=dstSRS,xRes=dstRes,yRes=dstRes,resampleAlg=resampleAlg,
                          dstNodata=dstNoData,multithread=True)
            outfiles.append(str(out))

        if to_single:
            out = Path(hdf_file).with_suffix(".tif")
            RasterTool.layer_stack(outfiles,str(out),data_type=dstDataType)
            for i in outfiles:
                os.remove(i)
            return str(out)

        return outfiles


    def _calibaration(self,ds,output_type,dst_nodata):
        ds = gdal.Open(ds)  # gdal.Open(hdf_type) not support 'gdal.GA_Update' mode
        try:
            metadata = ds.GetMetadata()
            scale = float(metadata['scale_factor'])
            offset = float(metadata['add_offset'])
            fill_value = float(metadata['_FillValue'])
            valid_range = [float(i) for i in metadata['valid_range'].split(",")]
        except Exception as e:
            raise Exception("Not enough metadata for calibration."+str(e))
        _ds = gdal.Warp("", ds, format="MEM",outputType=output_type)
        band = _ds.GetRasterBand(1)
        dst_nodata = fill_value if dst_nodata is None else dst_nodata
        arr = band.ReadAsArray().astype(gdalnumeric.flip_code(output_type))
        if valid_range is not None:
            valid_mask = (arr > valid_range[0]) | (arr < valid_range[1]) & (arr != fill_value)
        else:
            valid_mask = np.zeros_like(arr,dtype=np.int) & (arr != fill_value)
        if scale > 1:
            arr = arr / scale + offset
        else:
            arr = arr * scale + offset

        arr[~valid_mask] = dst_nodata
        band.WriteArray(arr)
        band.SetNoDataValue(dst_nodata)
        band.FlushCache()
        del ds
        return _ds

    def _mosaic(self,files:ListPathLike,out_file:PathLike):
        tempvrt = "./tempvrt.vrt"
        vrt = gdal.BuildVRT(tempvrt,files)
        gdal.Translate(out_file,vrt)
        del vrt
        os.remove(tempvrt)


if __name__ == '__main__':

    from base_alg.utils.raster_tool import RasterTool
    # md = MODISBaseAlg()
    # p = Path(r"I:\project\Tongliao\modis\MOD13Q1\2020")
    # h26s = []
    # h27s = []
    # # for f in p.rglob("*.hdf"):
    # #     md._reproject(str(f), '250m 16 days NDVI', "EPSG:4326", 0.0025,dstNoData=0)
    # for f in p.rglob("*.tif"):
    #     #
    #     if "h26v04" in f.name:
    #         h26s.append(str(f))
    #     else:
    #         h27s.append(str(f))
    # h26 = p.parent / "h26.tif"
    # h27 = p.parent / "h27.tif"
    # a = p.parent / "h2627_NDVI_MAX.tif"
    # RasterTool.merge_layers(h26s,str(h26),"MAX")
    # RasterTool.merge_layers(h27s,str(h27),"MAX")
    # md._mosaic([str(h26),str(h27)],str(a))

    # tvdi
    base = [
    r"I:\project\Tongliao\modis\demo\CHN_3100000000_20200711_modis_250M_soildrought.tif",
    r"I:\project\Tongliao\modis\demo\CHN_3100000000_20200727_modis_250M_soildrought.tif",
    r"I:\project\Tongliao\modis\demo\CHN_3100000000_20200812_modis_250M_soildrought.tif",
    r"I:\project\Tongliao\modis\demo\CHN_3100000000_20200828_modis_250M_soildrought.tif",
    r"I:\project\Tongliao\modis\demo\CHN_3100000000_20200913_modis_250M_soildrought.tif"
    ]
    t = r"I:\project\Tongliao\modis\demo\xlh_modis_250M_soildrought.temp.tif"
    tt = r"I:\project\Tongliao\modis\demo\xlh_modis_250M_soildrought.tif"
    shp = r"I:\project\Tongliao\数据\辽河流域\Xiliaohe_150000_boundary.shp"
    # RasterTool.merge_layers(base,t,'MIN')
    # gdal.Warp(tt,t,cutlineDSName=shp,cropToCutline=True)
    ds = gdal.Open(r"H:\HyperRecons\S2B_MSIL2A_20220308T030549_N9999_R075_T49RGP_20220325T025721.tif")
    print()






