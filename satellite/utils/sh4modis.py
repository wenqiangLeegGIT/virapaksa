#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from pathlib import Path
from sentinelhub import SentinelHubRequest,DataCollection,MimeType

from satellite.utils.sh_api import SHDownloader
from satellite.utils.eval_scrpits import es_zp

class SH4MODIS(SHDownloader):

    def _build_request(self,bbox):
        request = SentinelHubRequest(
            evalscript=self.eval_script,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    time_interval=(self.start_date, self.end_date),
                    maxcc=self.cloud_coverage / 100,
                    mosaicking_order="mostRecent",
                )
            ],
            responses=[
                SentinelHubRequest.output_response("modis_mcd43a4_006", MimeType.TIFF)
            ],
            bbox=bbox,
            resolution=(self._res[self.data_collection.name], self._res[self.data_collection.name]),
            config=self.config,
        )
        return request




if __name__ == '__main__':
    out = r"I:\temp\nxt\testdownload.tif"
    xp = r"F:\productTempRoot\temp\xipingtemp\CHN-HE-ZHU-XIP_clip.shp"
    zp = r"I:\temp\shtest\zouping.shp"
    outfile2 = r"F:\productTempRoot\temp\xipingtemp\sh\xptestdownload.tif"
    start = "20220420"
    end = "20220430"

    # SH4MODIS(out,zp,start,end,DataCollection.MODIS,es_zp).exec_download()
    class Distance(float):
        def __new__(cls, value, unit):
            instance = super().__new__(cls, value)
            instance.unit = unit
            return instance


    in_miles = Distance(42.0, "Miles")


    dir(in_miles)
