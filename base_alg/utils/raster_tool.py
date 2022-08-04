#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import shutil
import tempfile
import warnings

import numpy as np
from osgeo import gdal,ogr
import gdalnumeric

from typedefs import *


class RasterTool:

    @staticmethod
    def get_nodata_mask(infile: Union[PathLike,gdal.Dataset],src_nodata: Optional[Real]=None) -> (Real,np.ndarray):
        """
        获取无效值掩膜矩阵
        如果src_nodata设置为None，则根据infile本身无效值生成无效值掩膜矩阵，此前提条件为infile必须已经正确写入无效值
        如果src_nodata设置为实数，则不论是否infile具有无效值，以src_nodata为依据生成无效值掩膜矩阵
        :param infile: 输入文件
        :param src_nodata: 输入文件无效值
        :return: 布尔矩阵
        """
        if isinstance(infile,(str,Path,os.PathLike)):
            ds = gdal.Open(str(infile))
        else:
            ds = infile
        if ds is None:
            raise Exception(f"Bad file {infile}.")
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        nodata = nodata if src_nodata is None else src_nodata
        if nodata is None:
            raise Exception(f"{infile} must set a 'NoDataValue', otherwise the rank result may be incorrect.")
        if np.isnan(nodata):
            nodata_mask = np.isnan(arr)
        else:
            nodata_mask = arr == nodata
        return nodata,nodata_mask

    @staticmethod
    def create_copy(
            tif_name: PathLike,
            ref_obj: Union[PathLike,gdal.Dataset],
            driver_name: str = "GTiff",
            raster_count: Optional[int] = None,
            out_type: Optional[int] = None,
            dst_nodata: Optional[Real] = None
    ) -> gdal.Dataset:
        """
        根据参考数据创建副本数据
        :param tif_name: 输出副本数据的路径
        :param ref_obj: 参考数据
        :param driver_name: 数据格式驱动，默认GTiff
        :param raster_count: 副本数据的波段数，不指定则和参考数据保持一致
        :param out_type: 副本数据的数据类型，不指定则和参考数据保持一致
        :param dst_nodata: 副本数据的无效值，不指定则和参考数据保持一致，需注意，如果不指定，则参考数据必须已经具备无效值
        :return:
        """
        if isinstance(ref_obj,(str,Path,os.PathLike)):
            ds = gdal.Open(ref_obj)
            if ds is None:
                raise Exception(f"Bad file {ref_obj}.")
        elif isinstance(ref_obj,gdal.Dataset):
            ds = ref_obj
        else:
            raise Exception("Not implemented.")
        driver = gdal.GetDriverByName(driver_name)
        raster_count = ref_obj.RasterCount if raster_count is None else raster_count
        inband = ds.GetRasterBand(1)
        out_type = inband.DataType if out_type is None else out_type
        tif_name = "" if driver_name == "MEM" else tif_name
        outds = driver.Create(
            str(tif_name),
            ds.RasterXSize,
            ds.RasterYSize,
            raster_count,
            out_type
        )
        dst_nodata = inband.GetNoDataValue() if dst_nodata is None else dst_nodata
        outds.SetProjection(ds.GetProjection())
        outds.SetGeoTransform(ds.GetGeoTransform())
        band = outds.GetRasterBand(1)
        band.SetNoDataValue(dst_nodata)
        del band,ds,driver
        return outds

    @staticmethod
    def layer_stack(
            band_imgs: ListPathLike,
            out_img: PathLike,
            driver_name: str = "GTiff",
            data_type: int = gdal.GDT_Float32,
            dst_nodata: Optional[int] = None,
            cutline_shp: Optional[PathLike] = None,
            band_list: Optional[ListInt] = None
    ):
        """
        波段组合
        :param band_imgs: 长度为1个(含)以上的栅格影像列表
        :param out_img: 输出波段组合结果路径
        :param driver_name: 输出波段组合结果文件格式，默认GTiff
        :param data_type: 输出波段组合结果数据类型，默认gdal.GDT_Float32
        :param dst_nodata: 输出波段组合结果数据無效值
        :param cutline_shp: 是否对波段组合结果进行裁剪，输入有效shp文件则进行裁剪
        :param band_list: 波段写入顺序，如[3,2,1]则将band_images中的文件依次写入输出结果的[3,2,1]波段，默认按照band_images中的文件顺序写入相应波段
        :return:
        """
        if len(band_imgs) == 1:
            raise ValueError("Images for layer stack must greater equal than one.")
        if not all([Path(i).exists() and Path(i).is_file() for i in band_imgs]):
            raise FileExistsError("All the input images must exists.")
        if gdal.GetDriverByName(driver_name) is None:
            raise Exception(f"Driver {driver_name} not found.")
        if gdalnumeric.flip_code(data_type) is None:
            raise ValueError("Data type is invalid.")
        if band_list is not None:
            if max(band_list) != len(band_imgs) or min(band_list) == 1:
                raise ValueError("Invalid band number or band index.")
        else:
            band_list = range(1, len(band_imgs)+1)

        dst_arr = None
        imgds = None
        for imgidx, bandidx in enumerate(band_list):
            imgds = gdal.Open(str(band_imgs[imgidx]))
            if imgds is None:
                raise Exception(f"Image {band_imgs[imgidx]} open failed.")
            dst_nodata = imgds.GetRasterBand(1).GetNoDataValue() if dst_nodata is None else dst_nodata
            if dst_nodata is None:
                raise ValueError("All the input imgs must have nodata, or you can manually pass a nodata to 'dst_nodata'.")
            if dst_arr is None:
                dst_arr = imgds.ReadAsArray().astype(gdalnumeric.flip_code(data_type))
            else:
                dst_arr = np.concatenate((dst_arr, imgds.ReadAsArray().astype(gdalnumeric.flip_code(data_type))))

        outds = RasterTool.create_copy("", band_imgs[0], "MEM", dst_arr.shape[0], data_type, dst_nodata=dst_nodata)
        outds.WriteRaster(0, 0, imgds.RasterXSize, imgds.RasterYSize, buf_type=data_type, buf_string=dst_arr.tobytes())
        del imgds
        outds.FlushCache()

        if cutline_shp is not None:
            shpds = ogr.Open(cutline_shp)
            if shpds is None:
                warnings.warn(f"{cutline_shp} open failed. output for full scene.")
                gdal.Warp(str(out_img), outds, format=driver_name, dstNodata=dst_nodata)
            else:
                del shpds
                gdal.Warp(str(out_img), outds, cutlineDSName=cutline_shp, cropToCutline=True, format=driver_name,
                          dstNodata=dst_nodata)
        else:
            gdal.Warp(str(out_img), outds, format=driver_name, dstNodata=dst_nodata)

        del outds
        return

    @staticmethod
    def reclassify(
            infile: PathLike,
            out_file: PathLike,
            thres: ListReal,
            class_value: Optional[ListInt] = None,
            src_nodata: Optional[Real] = None,
            dst_nodata: Real = 0,
            negative_inf: bool = False,
            positive_inf: bool = False
    ):
        """
        对单波段栅格文件进行分级操作。
        关于negative/positive_inf，如果为True 则在thres开始/结束位置插入-np.inf/np.inf。
        eg.
        >> thres = [a,b,c]; (a<b<c)
        >> if negative_inf is True and class_value is None:
               thres = [-np.inf,a,b,c] # 以此thres的分级标准为 1:[-∞,a),2:[a,b),3:[b,c)
        >> if positive_inf is True class_value is None:
        >>     thres = [a,b,c,np.inf] # 以此thres的分级标准为 1:[a,b),2:[b,c),3:[c,+∞)
        :param infile: 输入待分级文件
        :param out_file: 输出分级文件
        :param thres: 分级参数列表
        :param class_value: 分级值, 默认None，分级值从1开始
        :param src_nodata: 输入文件的无效值，if None则自动获取, else，人为指定(适用无NoDataValue的情况)
        :param dst_nodata: 输出分级文件的无效值
        :param negative_inf: 分级参数是否从负无穷开始
        :param positive_inf: 分级参数是否结束于正无穷
        :return:
        """
        ds = gdal.Open(str(infile))
        if ds is None:
            raise FileNotFoundError(f"Bad file {infile}")
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        _, nodata_mask = RasterTool.get_nodata_mask(ds,src_nodata)
        ranks = np.full_like(arr,fill_value=np.nan, dtype=np.float32)
        thres.insert(0, -np.inf) if negative_inf else thres
        thres.append(np.inf) if positive_inf else thres
        if class_value:
            if len(class_value) != len(thres):
                raise Exception("When 'negative/positive_inf' is set True, you must pass 'class_value' with same length"
                                " of new 'thres'.")
            for i in range(1, len(thres)):
                ranks[(arr >= thres[i - 1]) & (arr < thres[i])] = class_value[i-1]
            ranks[arr < thres[0]] = class_value[0]
            ranks[arr >= thres[-1]] = class_value[-1]
        else:
            for i in range(1, len(thres)):
                ranks[(arr >= thres[i - 1]) & (arr < thres[i])] = i
            ranks[arr < thres[0]] = 1
            ranks[arr >= thres[-1]] = len(thres) - 1
        ranks[nodata_mask] = dst_nodata
        ranks = ranks.astype(gdalnumeric.flip_code(gdal.GDT_Int16))
        outds = RasterTool.create_copy(str(out_file),ds,raster_count=1,out_type=gdal.GDT_Int16,dst_nodata=dst_nodata)
        band = outds.GetRasterBand(1)
        band.WriteArray(ranks)

        del ds, outds
        return

    @staticmethod
    def merge_rasters(
            rasters: ListPathLike,
            out_layer: PathLike,
            pattern: str = 'MAX',
            src_nodata: Optional[Real] = None,
            dst_nodata: Optional[Real] = None,
            out_format: int = gdal.GDT_Float32
    ):
        if len(rasters) == 0:
            raise ValueError("Input layers is empty.")
        if pattern.lower() not in ['max','min','mean']:
            raise ValueError(f"Merge pattern must be one of 'MAX', 'MIN' and 'MEAN'. Got {pattern}.")

        if len(rasters) == 1:
            shutil.copy(rasters[0],out_layer)
            return

        out_arr = None
        nodata_mask = None
        ref_ds = None
        _mean_n = None
        for lyr in rasters:
            ds = gdal.Open(lyr)
            if ds is None:
                raise FileNotFoundError(f"Bad file {lyr}.")
            band = ds.GetRasterBand(1)
            ref_ds = ds if ref_ds is None else ref_ds
            lyr_nodata, lyr_nodata_mask = RasterTool.get_nodata_mask(ds,src_nodata)
            if dst_nodata is None:
                if src_nodata is None:
                    dst_nodata = band.GetNoDataValue() if dst_nodata is None else dst_nodata
                    if dst_nodata is None:
                        raise Exception("You must set a dst nodata.")
                else:
                    dst_nodata = src_nodata
            nodata_mask = np.zeros((ds.RasterYSize,ds.RasterXSize)).astype(np.bool_)
            arr = band.ReadAsArray().astype(np.float32)
            nodata_mask = np.bitwise_or(nodata_mask, ~lyr_nodata_mask)
            if pattern.lower() == "max":
                if out_arr is None:
                    out_arr = np.full((ds.RasterYSize,ds.RasterXSize),fill_value=-np.inf,
                                      dtype=gdalnumeric.flip_code(out_format))
                out_arr = np.max([out_arr,arr],axis=0)
            elif pattern.lower() == "min":
                if out_arr is None:
                    out_arr = np.full((ds.RasterYSize, ds.RasterXSize), fill_value=np.inf,
                                      dtype=gdalnumeric.flip_code(out_format))
                out_arr = np.min([out_arr, arr], axis=0)

            elif pattern.lower() == "mean": # applied iterating mean to save memory and compute resource
                if _mean_n is None:
                    _mean_n = np.full_like(arr, fill_value=2)  # auxiliary variable for iterating mean
                if out_arr is None:
                    out_arr = arr.copy()
                else:
                    print(lyr_nodata)
                    out_arr[out_arr==lyr_nodata] = arr[out_arr==lyr_nodata]
                    arr[arr==lyr_nodata] = out_arr[arr==lyr_nodata]
                    out_arr = out_arr + ((arr - out_arr) / _mean_n)
                    _mean_n += 1

        # out_arr[~nodata_mask] = dst_nodata
        outds = RasterTool.create_copy(str(out_layer),ref_ds,raster_count=1,out_type=out_format,dst_nodata=dst_nodata)
        band = outds.GetRasterBand(1)
        band.WriteArray(out_arr)

        del ds, outds
        return


    @staticmethod
    def apply_mask(
            outfile: PathLike,
            infile: PathLike,
            mask_file: PathLike,
            mask_value: Real = 0,
            src_nodata: Optional[Real] = None,
            dst_nodata: Optional[Real] = None,
            np_ma_mask_func = None,
            mask_as_ref: bool = True
    ):
        """
        对输入文件进行掩膜计算
        :param outfile: 输出掩膜文件
        :param infile: 输入待进行掩膜计算的数据
        :param mask_file: 掩膜文件
        :param mask_value: 掩膜文件中需要被掩膜掉的值，默认0
        :param src_nodata: 输入数据的无效值
        :param dst_nodata: 输出数据的无效值
        :param np_ma_mask_func: numpy.ma中的掩膜函数，如np.ma.mask_equal/where/or，当此参数传入时，将忽略mask_value，提供此参数的目的是增加函数的灵活性。
        :param mask_as_ref: 当掩膜文件与输入文件行列数不等时，以掩膜文件为基准，将输入文件行列数warp至mask文件，反之调换输入文件与掩膜文件的。
        :return:
        """
        maskds = gdal.Open(str(mask_file))
        inds = gdal.Open(str(infile))
        if maskds is None:
            raise FileNotFoundError(f"Bad mask file {mask_file}.")
        if inds is None:
             raise FileNotFoundError(f"Bad input file {infile}.")
        if (maskds.RasterXSize != inds.RasterXSize) or (maskds.RasterYSize != inds.RasterYSize):
            warnings.warn("rows or columns of mask file and input file is not equal,therefore input or mask file will be"
                          " applied resample to match the size of input or mask file.")
            fd, align_file = tempfile.mkstemp(suffix=".tif")
            if mask_as_ref:
                _ds = gdal.Warp(align_file,maskds,width=inds.RasterXSize,height=inds.RasterYSize,
                                resampleAlg=gdal.GRA_NearestNeighbour)
                maskds = _ds
            else:
                _ds = gdal.Warp(align_file, inds, width=maskds.RasterXSize, height=maskds.RasterYSize,
                                resampleAlg=gdal.GRA_NearestNeighbour)
                inds = _ds
        _, src_nodata_mask = RasterTool.get_nodata_mask(inds,src_nodata)
        mask = maskds.GetRasterBand(1).ReadAsArray()
        if np_ma_mask_func is not None:
            mask = ~np_ma_mask_func(mask).mask
        else:
            _, mask = RasterTool.get_nodata_mask(maskds,mask_value)
            mask = ~mask
        outds = RasterTool.create_copy(outfile,inds,dst_nodata=dst_nodata)
        for r in range(inds.RasterCount):
            band = inds.GetRasterBand(r+1)
            arr = band.ReadAsArray()
            out_arr = arr * mask
            out_arr[src_nodata_mask|~mask] = dst_nodata
            outband = outds.GetRasterBand(r+1)
            outband.WriteArray(out_arr)
        outds.FlushCache()
        del inds,outds
        return


