#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from pathlib import Path
from typedefs import *
import datetime

class FileTool:

    @staticmethod
    def get_file_time(file:PathLike,file_type='Sentinel') -> datetime.datetime:
        if file_type not in ['Sentinel','Landsat','MODIS','SMAP','GPM']:
            raise Exception("Only support 'Sentinel','Landsat','MODIS','SMAP','GPM'. Got {file_type}.")
        file = Path(file)
        if not file.exists():
            raise FileExistsError(f"{file} isn't exisit.")
        _time = None
        if file_type == "Sentinel":
            if file.name.startswith("S"):
                _time = datetime.datetime.strptime(file.name.split("_")[2],"%Y%m%dT%H%M%S")
            elif file.name.startswith("L"):
                _time = datetime.datetime.strptime(file.name.split("_")[3],"%Y%m%dT%H%M%S")
        elif file_type == "Landsat":
            pass
        elif file_type == "MODIS":
            pass
        elif file_type == "SMAP":
            pass
        elif file_type == "GPM":
            pass

        return _time


if __name__ == '__main__':
    t = FileTool.get_file_time(r"E:\360MoveData\Users\pc019\Documents\S2A_MSIL1C_20200425T031541_N0209_R118_T49SEV_20200425T061757.SAFE")
    print(t)