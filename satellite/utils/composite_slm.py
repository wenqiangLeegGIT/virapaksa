#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datetime
from pathlib import Path
from sentinelhub import (SHConfig, CRS, SentinelHubRequest, MimeType, BBoxSplitter,DataCollection)
from algorithm.BussinessProjectDataProductionTask.utils.sh_api import SHDownloader
from algorithm.BussinessProjectDataProductionTask.utils.eval_scrpits import es_slm

class SLMDownloader(SHDownloader):

    def buildRequest(self,in_bbox,data_dir):
        request = SentinelHubRequest(
            evalscript=self.eval_script,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    identifier = "l2a",
                    time_interval=(self.start_date,self.end_date),
                    maxcc=self.cloud/100,
                    mosaicking_order="leastCC"
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.LANDSAT_OT_L2,
                    identifier="ls8",
                    time_interval=(self.start_date, self.end_date),
                    maxcc=self.cloud / 100,
                    mosaicking_order="leastCC",
                    upsampling="BILINEAR"
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.MODIS,
                    identifier="modis",
                    time_interval=(self.start_date, self.end_date),
                    maxcc=self.cloud / 100,
                    mosaicking_order="mostRecent",
                    upsampling="BILINEAR",
                )
            ],
            responses=[
                SentinelHubRequest.output_response("s2_l2a", MimeType.TIFF),
                SentinelHubRequest.output_response("landsat8_9_c2_l2", MimeType.TIFF),
                SentinelHubRequest.output_response("modis_mcd43a4_006", MimeType.TIFF)
            ],
            bbox=in_bbox,
            # resolution=(0.0001, 0.0001),
            resolution=(10, 10),
            config=self.config,
        )
        request.payload['input']['bounds']['properties']['crs'] = "http://www.opengis.net/def/crs/EPSG/0/32649"
        request.data_folder = data_dir
        request.save_data()

    def exec_download(self):
        tempdir = None
        boundary = self.getBBox(self.boundary_shp)
        # rows,cols = self.getSplitShape(boundary, 0.0001)
        rows,cols = self.getSplitShape(boundary, 10)
        bboxs = BBoxSplitter(boundary,CRS.WGS84, (rows,cols))
        bblen = len(bboxs.get_bbox_list())
        total_time = 0
        n = 1
        tempdir = Path(self.outfile).parent
        for bb in bboxs.get_bbox_list():
            tic = datetime.datetime.now()
            self.buildRequest(bb,tempdir)
            toc = datetime.datetime.now()
            print(f"Downloaded [{n}/{bblen}], consuming {(toc - tic).seconds} seconds.")
            n += 1
            total_time += (toc - tic).seconds


if __name__ == '__main__':


    # xp = r"F:\productTempRoot\temp\xipingtemp\CHN-HE-ZHU-XIP_clip.shp"
    xp = r"F:\productTempRoot\temp\xipingtemp\CHN-HE-ZHU-XIP_clip_utm.shp"
    outfile2 = r"F:\productTempRoot\temp\xipingtemp\sh\testdownload.tif"
    start = "20220420"
    end = "20220430"

    SLMDownloader(xp, start, end, 100, outfile2,eval_script=es_slm).exec_download()