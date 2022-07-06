#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from pathlib import Path
from typing import TypeVar,List,Tuple,Dict,Union

import gdal

Real = TypeVar('Real',float,int)
ListInt = List[int]
ListFloat = List[float]
ListReal = List[Real]
TupleInt = Tuple[int]
TupleFloat = Tuple[float]
TupleReal = Tuple[Real]
ListStr = List[str]
TupleStr = Tuple[str]
SeqInt = Union[ListInt,TupleInt]
SeqFloat = Union[ListFloat,TupleFloat]
SeqReal = Union[ListReal,TupleReal]
SeqStr = Union[ListStr,TupleStr]
PathLike = TypeVar('PathLike',str,Path,os.PathLike)
ListPathLike = List[PathLike]
TuplePathLike = Tuple[PathLike]
SeqPathLike = Union[ListPathLike,TuplePathLike]
DictStr = Dict[str,str]
DictInt = Dict[int,int]
DictFloat = Dict[float,float]
DictStrInt = Dict[str,int]
DictIntStr = Dict[int,str]
DictStrFloat = Dict[str,float]
DictIntFloat = Dict[int,float]
DictFloatInt = Dict[float,int]
DictFloatStr = Dict[float,str]

