#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import abc
import os
import datetime
import logging
from pathlib import Path
import shutil
from typing import Union,List
import tempfile
import tarfile

from sentinelhub import (SHConfig, CRS, SentinelHubRequest, BBox, BBoxSplitter,DataCollection)
from osgeo import ogr,gdal
from shapely import wkb,geometry,speedups
speedups.disable() # disable speedup function in case of C extention crush.

logging.basicConfig(level=logging.INFO)
CLIENT_ID = "224dc6cc-105a-4de5-bb58-47ebc62a5b4c"
CLIENT_SECRET = "P^&WP~i}Pa(Kg]h0<%t!%s5_S[~eUI@VKc>zTc@-"
SH_PROCESS_UNIT_SIZE = 512


class SHDownloader(abc.ABC):
    def __init__(
            self,
            outfile:str,
            boundary_shp:str,
            start_date:str,
            end_date:str,
            data_collection:DataCollection,
            eval_script:str,
            cloud_coverage=100
    ):
        self.outfile = Path(outfile)
        self.boundary_shp = boundary_shp
        self.start_date = datetime.datetime.strptime(start_date,"%Y%m%d")
        self.end_date = datetime.datetime.strptime(end_date,"%Y%m%d")
        self.data_collection = data_collection
        self.cloud_coverage = cloud_coverage
        self.eval_script = eval_script

        self.tempdir = Path(tempfile.mkdtemp(dir=self.outfile.parent))
        if self.tempdir.exists(): shutil.rmtree(self.tempdir)
        Path.mkdir(self.tempdir, parents=True)

        self.config = SHConfig()
        if CLIENT_ID and CLIENT_SECRET:
            self.config.sh_client_id = CLIENT_ID
            self.config.sh_client_secret = CLIENT_SECRET

    def __del__(self):
        if self.tempdir.exists():
            shutil.rmtree(self.tempdir)

    _res = {
        'MODIS':5e-3, # 500m
        'LANDSAT_OT_L2':3e-4, # 30m
        'SENTINEL2_L2A':1e-4 # 10m
    }

    def _getBBox(self,inshp):
        inds = ogr.Open(inshp)
        if inds is None:
            raise Exception(f"{inshp} Open FAILED!")
        geoms = []
        lyr = inds.GetLayer()
        for feat in lyr:
            geoms.append(wkb.loads(feat.GetGeometryRef().ExportToWkb()))
        inds = None
        return geoms

    def _getSplitShape(self,in_geoms:geometry,res:Union[float,int,List[float],List[int]])->List[int]:
        minx = miny = 1e10
        maxx = maxy = -1e10
        for geom in in_geoms:
            _minx,_miny,_maxx,_maxy = geom.bounds
            minx = minx if minx < _minx else _minx
            miny = miny if miny < _miny else _miny
            maxx = maxx if maxx > _maxx else _maxx
            maxy = maxy if maxy > _maxy else _maxy
        if isinstance(res,int) or isinstance(res,float):
            resx = resy = res
        elif isinstance(res,tuple) or isinstance(res,list) and len(res) == 2:
            resx, resy = res
        else:
            raise Exception(f" 'res' must be one of type of 'float', 'int' or 'list' contain 'float' or 'int'.Plz check the param")
        x = abs(int((maxx - minx)/resx))//SH_PROCESS_UNIT_SIZE
        y = abs(int((maxy - miny)/resy))//SH_PROCESS_UNIT_SIZE
        if x*y == 0:
            x = y = 1
        return [x,y]


    @abc.abstractmethod
    def _build_request(self,bbox:BBox)->SentinelHubRequest:
        pass

    def _mosaic2one(self,filelst,outfile):

        fd,vrt = tempfile.mkstemp(suffix=".vrt")
        vrt_all = gdal.BuildVRT(str(vrt), filelst)
        gdal.Translate(str(outfile), vrt_all, format="GTiff")
        del vrt_all
        os.close(fd)
        os.remove(vrt)

    def _post_process(self,tars):
        muls = []
        for tar in tars:
            tempdir = tempfile.mkdtemp(dir=Path(tar).parent)
            with tarfile.open(tar) as tarobj:
                tarobj.extractall(tempdir)
        if len(muls) >= 1:
            fd,tempmerged = tempfile.mkstemp(suffix=".tif")
            self._mosaic2one(muls,tempmerged)
            gdal.Warp(str(self.outfile), tempmerged, cutlineDSName=self.boundary_shp, cropToCutline=True, dstNodata=0)
            os.close(fd)
            os.remove(tempmerged)


    def exec_download(self):
        tempdir = None
        try:
            boundary = self._getBBox(self.boundary_shp)
            rows,cols = self._getSplitShape(boundary, SHDownloader._res[self.data_collection.name])
            bboxs = BBoxSplitter(boundary, CRS.WGS84, (rows,cols))
            bblen = len(bboxs.get_bbox_list())
            total_time = 0
            n = 1

            for bb in bboxs.get_bbox_list():
                tic = datetime.datetime.now()
                req = self._build_request(bb)
                req.data_folder = self.tempdir
                req.save_data()
                toc = datetime.datetime.now()
                logging.info(f"Downloaded [{n}/{bblen}], consuming {(toc - tic).seconds} seconds.")
                n += 1
                total_time += (toc - tic).seconds
            logging.info(f"Download procedure complete. Total time {total_time / 60} minutes.")
            filelst = list(map(str,self.tempdir.rglob("*.tar")))
            if len(filelst) == 0:
                raise Exception("No valid file found.")
            self._post_process(filelst)
            return True,''
        except Exception as e:
            logging.error(e)
            return False, str(e)

