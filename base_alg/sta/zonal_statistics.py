#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import shutil

from base_alg.typedefs import *
import os, math,tempfile
from collections import defaultdict
import warnings
import numpy as np
from osgeo import gdal,ogr,osr
import pandas as pd



# 支持中文
gdal.SetConfigOption("SHAPE_ENCODING","GBK")

class ZonalStatistics:
    """
    分区统计功能类
    """
    def __init__(
            self,
            inShp        : PathLike,
            inRaster     : PathLike,
            zoneField    : str,
            statisticType: Union[str, ListStr],
            staFieldName : Union[str,None] = None,
            outputType   : str = 'DICT',
            outputPath   : Union[PathLike, None] = None,
            staNodata    : Union[str,Real] = '-',
            noDataValue  : Union[Real,None] = None,
            **kwargs
    ):
        """
        inShp        :输入的统计shp
        inRaster     :需要统计的栅格
        zonefield    :inShp中的栅格化字段
        statisticType:统计类型，'ALL' 全部统计，'SUM'统计和，['SUM','MEAN'],统计和和平均数
        staFieldName :统计数据索引字段，默认与zonefield相同，用于将统计数据的key替换为staFieldName指定的字段。
        outputType   :输出类型 'DICT',python内置dict数据类型， 'CSV'
        outputPath   :输出路径，outputType为'DICT'时无效
        staNodata    :当区域内无有效数据时，统计结果将以staNodata作为默认结果填入
        noDataValue  :统计栅格的无效值,默认为None，此时将自动获取，因此统计栅格必须已经设置了无效值，如果统计栅格没有设置无效值，则可以传入指定的无效值

        kwargs       :STA_NAME_MAP: Union[str, list],将statisticType参数映射为自定义名称，如统计'MAX' ，则输出文件中字段名称
                      也为'MAX'，此时STA_NAME_MAP传入自定义名称则可以进行字段名称的替换，如传入'max',则统计字段中名称为'max',
                      因此，此参数的类型，长度，顺序均应该与statisticType参数保持严格对应
                     :RASTERIZE_SHP: str, 如果统计n幅影像，需要将inShp栅格化n次，对于feature比较多的shp比较耗费时间，因此，可以将inShp事先
                       栅格化为tif，将tif作为参数传入以提高效率
                     :PRECALC_GEOMETRY_BOUNDARY: bool, 对于较大的影像，可以计算出inShp中Geometry的Envlope信息，提高计算效率
                     :AREA_UNIT 面积统计单位 m2/mu/km2/ha，默认m2

        """

        self.parcelShp = inShp
        self.raster = inRaster
        self.zoneField = zoneField
        self.staFieldName = staFieldName
        self.statisticType = statisticType
        self.outputType = outputType
        self.outputPath = outputPath
        self.staNodata = staNodata
        self.noDataValue = noDataValue
        self.STA_NAME_MAP = kwargs['STA_NAME_MAP'] if 'STA_NAME_MAP' in kwargs else None
        self.RASTERIZED_SHP = kwargs['RASTERIZED_SHP'] if 'RASTERIZED_SHP' in kwargs else None
        self.PRECALC_GEOMETRY_BOUNDARY = kwargs['PRECALC_GEOMETRY_BOUNDARY'] if 'PRECALC_GEOMETRY_BOUNDARY' in kwargs else True
        self.AREA = kwargs['AREA_UNIT'] if 'AREA_UNIT' in kwargs else 'm2'
        assert self.AREA in ['m2','km2','mu','ha'], "Area statistic type only support 'm2','km2','mu','ha'."

        self.bulitinStatisticFunc = { # 'ALL' for all the statistic type
            'SUM'      : self._cbSum,
            'MEAN'     : self._cbMean,
            'MAX'      : self._cbMax,
            'MIN'      : self._cbMin,
            'STD'      : self._cbStd,
            'MEDIAN'   : self._cbMedian,
            'MAJORITY' : self._cbMajority,
            'COUNT'    : self._cbCount,
            'AREA'     : self._cbArea,
        }

        if self.outputType == 'CSV':
            assert self.outputPath is not None, "Must pass a statistic csv file path."
        assert self._checkSRS(), "Spatial Reference of shapefile and raster should be same."

        self.staFieldName = self.zoneField if self.staFieldName is None else self.staFieldName

        ds = gdal.Open(self.raster)
        assert ds is not None,f"Bad raster {self.raster}."
        band = ds.GetRasterBand(1)
        self.noDataValue = band.GetNoDataValue() if self.noDataValue is None else self.noDataValue

        if self.statisticType == 'ALL' or "AREA" in self.statisticType:

            scale = {'m2':1,'km2':1e-6,'mu':1/666.7,'ha':1e-4}
            geos = ds.GetGeoTransform()
            proj = ds.GetProjectionRef()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj)
            if srs.IsGeographic():
                warnings.warn(f"The CRS of input raster {self.raster} is Geographic, this may cause statistics of area not strictly correct.")
                self.pix_area = round(abs(geos[1]*geos[5])*1e10 * scale[self.AREA],4)
            else:
                self.pix_area = round(abs(geos[1] * geos[5]) * scale[self.AREA],4)

        del band ,ds

    def _getUniqeValue(self):
        uq = []
        inds = ogr.Open(self.parcelShp)
        layer = inds.GetLayer(0)
        for i in range(layer.GetFeatureCount()):
            feat = layer.GetFeature(i)
            if self.zoneField == 'fid':
                uq.append(feat.GetFID())
            else:
                uq.append(feat.GetField(self.zoneField))
        return np.array(uq,dtype=np.int)

    def _checkSRS(self):
        rasds = gdal.Open(self.raster)
        shpds = ogr.Open(self.parcelShp)
        rasrs = osr.SpatialReference()
        spsrs = shpds.GetLayer().GetSpatialRef()
        rasrs.ImportFromWkt(rasds.GetProjectionRef())
        # return bool(rasrs.IsSame(spsrs)) # gdal BUG 相同坐标系仍然返回0
        if rasrs.IsGeographic():
            return bool(rasrs.IsSameGeogCS(spsrs))
        else:
            return bool(rasrs.IsSameVertCS(spsrs))

    def _getStaFunc(self):
        cbfunc = {}
        if self.statisticType == 'ALL':
            cbfunc = self.bulitinStatisticFunc
        else:
            if isinstance(self.statisticType,list):
                for _type in self.statisticType:
                    cbfunc[_type] = self.bulitinStatisticFunc[_type]
            else:
                cbfunc[self.statisticType] = self.bulitinStatisticFunc[self.statisticType]

        if self.STA_NAME_MAP is not None:
            if not (type(self.STA_NAME_MAP) is type(self.statisticType)):
                raise Exception("type of 'STA_NAME_MAP' and type of 'statisticType' is not same, plz cheack! ")
            if isinstance(self.STA_NAME_MAP,list) and  isinstance(self.statisticType,list) and len(self.STA_NAME_MAP) != \
                    len(self.statisticType):
                raise Exception("length of 'STA_NAME_MAP' and length of 'statisticType' must be equal, plz cheack! ")
            STA_NAME_MAP = [self.STA_NAME_MAP] if isinstance(self.STA_NAME_MAP,str) else self.STA_NAME_MAP
            _cbfunc = {}
            for idx,(name,func) in enumerate(cbfunc.items()):
                _cbfunc[STA_NAME_MAP[idx]] = cbfunc[name]
            cbfunc = _cbfunc
        return cbfunc

    def _getStaHeader(self):
        header = {}
        shpds = ogr.Open(self.parcelShp)
        if shpds is None:
            raise Exception("Shpfile Open failed.")
        lyr = shpds.GetLayer()
        for idx in range(lyr.GetFeatureCount()):
            feat = lyr.GetFeature(idx)
            if self.zoneField == "fid":
                key = feat.GetFID()
            else:
                key = feat.GetField(self.zoneField)
            name = feat.GetField(self.staFieldName)
            header[key] = name
        return header



    def setCustomizeCallback(self,funcName,funcObj):
        self.bulitinStatisticFunc[funcName] = funcObj
        if funcName not in self.statisticType:
            self.statisticType.append(funcName)

    def _initGeoInfo(self,inRaster):
        """
        输出统计结果为影像时，初始化信息
        :param inRaster:
        :return:
        """
        self.__outTifInfo = {}
        ds = gdal.Open(inRaster)
        self.__outTifInfo['RasterXSize'] = ds.RasterXSize
        self.__outTifInfo['RasterYSize'] = ds.RasterYSize
        self.__outTifInfo['CRS'] = ds.GetProjectionRef()
        self.__outTifInfo['GeoTransform'] = ds.GetGeoTransform()
        self.__outTifInfo['DataType'] = ds.GetRasterBand(1).ReadAsArray(0,0,1,1).dtype

    def execStatistic(self,rasterizedShp:str=None,**kw):

        cbfunc = self._getStaFunc()
        header = self._getStaHeader()
        descriptor = -1
        if rasterizedShp is None:
            descriptor,rasterizedShp = tempfile.mkstemp(suffix='.tif')
            status = Utils.polygon2mask(self.parcelShp,self.raster,rasterizedShp,self.zoneField)
            if status:
                raise Exception("Rasterize faild.")
        self._initGeoInfo(rasterizedShp)

        if self.PRECALC_GEOMETRY_BOUNDARY:
            boundary = Utils.readBoundary(self.parcelShp,self.raster,self.zoneField)

        shp_rsds = gdal.Open(rasterizedShp)
        if shp_rsds is None:
            raise Exception("Open rasterize file failed.")
        uq = self._getUniqeValue()
        ds = gdal.Open(self.raster)
        geos = ds.GetGeoTransform()
        self.pix_area = abs(geos[1]*geos[5])
        result = {}
        for idx in uq:
            if self.PRECALC_GEOMETRY_BOUNDARY:
                xoff, yoff, xsize, ysize = boundary[idx]['xoff'],boundary[idx]['yoff'],boundary[idx]['xsize'],boundary[idx]['ysize']
            else:
                xoff, yoff, xsize, ysize = 0, 0, shp_rsds.RasterXSize,shp_rsds.RasterYSize
            shp_dsarr = shp_rsds.ReadAsArray(xoff, yoff, xsize, ysize)
            dsarr = ds.ReadAsArray(xoff, yoff, xsize, ysize)
            shpmask = np.where(shp_dsarr == idx, True, False)
            if np.isnan(self.noDataValue):
                rasmask = np.where(np.isnan(dsarr),False,True)
            else:
                rasmask = np.where(dsarr==self.noDataValue,False,True)
            mask = shpmask&rasmask
            _arr = dsarr[mask]
            result[header[idx]] = {}
            for type,func in cbfunc.items():
                if len(_arr) == 0:
                    result[header[idx]][type]= self.staNodata
                else:
                    result[header[idx]][type] = func(_arr)
        shp_rsds,ds = None,None
        if descriptor != -1:
            os.close(descriptor)
            os.remove(rasterizedShp)
        return result

    def _cbSum(self,arr,**kw):
        return np.sum(arr)

    def _cbMean(self,arr,**kw):
        return np.mean(arr)

    def _cbMax(self,arr,**kw):
        return np.max(arr)

    def _cbMin(self,arr,**kw):
        return np.min(arr)

    def _cbStd(self,arr,**kw):
        return np.std(arr)

    def _cbMedian(self,arr,**kw):
        return np.median(arr)

    def _cbMajority(self,arr,**kw):
        if np.issubdtype(arr.dtype, np.floating):
            warnings.warn("Data type of array applying Marjoriy statistic should be integer.")
            arr = arr.astype(np.int32)
        return np.argmax(np.bincount(arr))

    def _cbCount(self,arr,**kw):
        return arr.size

    def _cbArea(self,arr,**kw):
        return arr.size * self.pix_area

    def Sta2Pandas(self,statistics:dict):
        field = self.statisticType
        stad = self._getStaFunc()
        if isinstance(self.statisticType,list):
            field = list(stad.keys())
        else:
            field = list(stad.keys())[0] if field != 'ALL' else list(stad.keys())
        data = defaultdict(list)
        index = []
        for k,v in statistics.items():
            for fld in field:
                data[fld].append(v[fld])
            index.append(k)
        pd.DataFrame(data,index=index).to_csv(self.outputPath,index_label=self.staFieldName)


    def run(self,pattern='iterate'):
        if pattern == 'global':
            sta = self.execStatistic(
                rasterizedShp=self.RASTERIZED_SHP,
                callback=self.statisticType
            )
        elif pattern == 'iterate':
            sta = self.execStatisticeEx()
        else:
            raise Exception("pattern must be 'global' or 'iterate'.")
        if self.outputType == 'CSV':
            self.Sta2Pandas(sta)
            return self.outputPath
        elif self.outputType == 'DICT':
            return sta
        else:
            return

    def execStatisticeEx(self):
        _tempdir = Path(self.parcelShp).parent / "_tempshps"
        _tempdir.mkdir(parents=True,exist_ok=True)
        cbfunc = self._getStaFunc()
        header = self._getStaHeader()
        boundary = Utils.readBoundary(self.parcelShp, self.raster, self.zoneField)
        rasds = gdal.Open(self.raster)
        srs = rasds.GetProjectionRef()
        geos = rasds.GetGeoTransform()
        shpds = ogr.Open(self.parcelShp)
        lyr = shpds.GetLayer()
        result = {}
        for feature in lyr:
            if self.zoneField == "fid":
                field = feature.GetFID()
            else:
                field = feature.GetField(self.zoneField)
            print(feature.GetField("CHINA_NAME"))
            tempshp = _tempdir / f"{field}.shp"
            fmtstr = f"ogr2ogr {tempshp} {self.parcelShp} -where {self.zoneField}={field} -lco ENCODING=UTF-8"
            os.system(fmtstr)
            _subds = ogr.Open(str(tempshp))
            _sublyr = _subds.GetLayer()
            minx,maxx,miny,maxy = _sublyr.GetExtent()
            target_ds = gdal.GetDriverByName("MEM").Create("",boundary[field]['xsize'],boundary[field]['ysize'],1,gdal.GDT_Float32)
            target_geos = list(geos[:])
            target_geos[0] = minx
            target_geos[3] = maxy
            target_ds.SetGeoTransform(target_geos)
            target_ds.SetProjection(srs)
            gdal.RasterizeLayer(target_ds, [1], _sublyr, burn_values=[1])
            mask_arr = target_ds.ReadAsArray().astype(np.bool_)
            data_arr = rasds.ReadAsArray(boundary[field]['xoff'], boundary[field]['yoff'], boundary[field]['xsize'],
                                    boundary[field]['ysize'])
            if np.isnan(self.noDataValue):
                data_mask = ~np.isnan(data_arr)
            else:
                data_mask = data_arr != self.noDataValue
            _arr = data_arr[mask_arr&data_mask]
            result[header[field]] = {}
            for type,func in cbfunc.items():
                if len(_arr) == 0:
                    result[header[field]][type]= self.staNodata
                else:
                    result[header[field]][type] = func(_arr)
        del shpds,rasds,target_ds,_subds
        shutil.rmtree(_tempdir)
        return result






class Utils:

    @staticmethod
    def polygon2mask(shapefile, rasterfile, outname, field="id"):
        """ Burn a ESRI shape file with multiple features to a mask file
            based on the extend and resolution of input raster file.
            Multiple features are represented by the values of id.
        :param str shapefile: input ESRI shape file
        :param srt rasterfile: input raster file based on to burn shape file
        :param srt outname: out file name, format: csv
        :param str field: a field with which to distinguish different features,
                     the field value should be a number type, default is "id"
        :return: status, 0 for success, 1 for failure
        <version> <1> 2018-10-25 Created by: wangyj
        """
        '''Open the shape file'''
        shp_driver = ogr.GetDriverByName("ESRI shapefile")
        shp_ds = shp_driver.Open(shapefile)
        layer = shp_ds.GetLayer(0)

        '''Open the raster file'''
        raster_ds = gdal.Open(rasterfile)
        ncol = raster_ds.RasterXSize
        nrow = raster_ds.RasterYSize
        proj = raster_ds.GetProjection()
        gt = raster_ds.GetGeoTransform()

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(outname, ncol, nrow, 1, gdal.GDT_UInt32)
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(gt)
        band = out_ds.GetRasterBand(1)
        band.Fill(0)

        status = gdal.RasterizeLayer(out_ds,  # output to our new dataset
                                     [1],  # output to our new dataset's first band
                                     layer,  # rasterize this layer
                                     None, None,  # No projection transform
                                     [0],  # burn value 0
                                     ['All_TOUCHED=TRUE',
                                      'ATTRIBUTE=' + field])  # rasterize all pixels touched by polygons
        del out_ds, shp_ds, raster_ds
        return status

    @staticmethod
    def world2Pixel(getransforms, x, y):

        resx = getransforms[1]
        resy = getransforms[5]
        upx = getransforms[0]
        upy = getransforms[3]

        _x = math.ceil((x - upx) / resx)
        _y = math.ceil((upy - y) / abs(resy))

        return _x, _y

    @staticmethod
    def readBoundary(shp: str, refRaster: str, field: str) -> dict:

        def validlizeCoor(coor, MAX, MIN=0):
            """
            Deal with the situation that boudary of shp and raster are not strictly overaly, restrict the coordinate
            between MIN and MAX.
            :param coor:
            :param MAX:
            :param MIN:
            :return:
            """
            coor = MIN if coor < MIN else coor
            coor = MAX if coor > MAX else coor
            return coor

        boundary = {}
        refds = gdal.Open(refRaster)
        XMAX = refds.RasterXSize
        YMAX = refds.RasterYSize
        shpds = ogr.Open(shp)
        if shpds is None or refds is None:
            raise Exception("Open file failed.")
        geots = refds.GetGeoTransform()
        layer = shpds.GetLayer(0)
        for idx in range(layer.GetFeatureCount()):
            feat = layer.GetFeature(idx)
            geom = feat.GetGeometryRef()
            xmin, xmax,ymin, ymax = geom.GetEnvelope()
            _xmin, _ymin = Utils.world2Pixel(geots, xmin, ymax)
            _xmax, _ymax = Utils.world2Pixel(geots, xmax, ymin)

            _xmin = validlizeCoor(_xmin, XMAX)
            _xmax = validlizeCoor(_xmax, XMAX)
            _ymin = validlizeCoor(_ymin, YMAX)
            _ymax = validlizeCoor(_ymax, YMAX)
            if field.lower() == "fid":
                boundary[feat.GetFID()] = {'xoff': _xmin, 'yoff': _ymin, 'xsize': _xmax - _xmin,
                                              'ysize': _ymax - _ymin}
            else:
                boundary[feat.GetField(field)] = {'xoff': _xmin, 'yoff': _ymin, 'xsize': _xmax - _xmin,
                                              'ysize': _ymax - _ymin}
        return boundary


if __name__ == '__main__':
    ras = r"I:\project\Tongliao\modis\demo\rsei.rank.tif"
    # rds = gdal.Open(ras)
    # geot = rds.GetGeoTransform()
    # srs = rds.GetProjection()
    shp = r"I:\project\Tongliao_it\trunk\python\auxiliary_data\xiliaohe_basin\Xiliaohe_150000.shp"
    # ds = ogr.Open(shp)
    # lyr = ds.GetLayer()
    # feat = lyr.GetFeature(16)
    # geom = feat.GetGeometryRef()
    # minx,maxx,miny,maxy = geom.GetEnvelope()
    # xoff = (minx - geot[0]) / geot[1]
    # yoff = (maxy - geot[3]) / geot[5]
    # xcount = abs(int((maxx - minx) / geot[1]))
    # ycount = abs(int((maxy - miny) / geot[5]))
    # geot_ = list(geot[:])
    # geot_[0] = minx
    # geot_[3] = maxy
    # target = gdal.GetDriverByName("GTiff").Create(
    #     r"I:\temp\t.tif",
    #     xcount,
    #     ycount,
    #     1,
    #     gdal.GDT_Float32
    # )
    # target.SetProjection(srs)
    # target.SetGeoTransform(geot_)
    # # gdal.Rasterize(target, [1], lyr, burn_values=[1])
    # # gdal.Warp(r"I:\temp\t.tif",target,format="GTiff")
    # # print()
    # os.system(fr"ogr2ogr I:\temp\t.shp {shp} -where FID=16 -lco ENCODING=UTF-8")
    # ds = ogr.Open(r"I:\temp\t.shp")
    # lyr = ds.GetLayer()
    # gdal.RasterizeLayer(target, [1], lyr, burn_values=[1])

    # ZonalStatistics(shp,ras,'fid','REGION_COD','ALL',AREA_UNIT="km2",outputType="CSV",outputPath=r"I:\temp\nxt\test.csv").run('iterate')
    import datetime
    tic = datetime.datetime.now()
    ras = "I:\project\Tongliao\modis\demo\CHN_3100000000_20200727_modis_250M_soildrought.tif"
    shp = r"I:\temp\region_chn\region_3100000000.shp"
    ZonalStatistics(shp, ras, 'fid', 'ALL','REGION_COD',  AREA_UNIT="km2", outputType="CSV",
                    outputPath=r"I:\temp\nxt\test.csv").run('global')
    toc = datetime.datetime.now()
    print((toc-tic).seconds)
