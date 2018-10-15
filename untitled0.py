import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Point
import os
import shapefile as shp
import matplotlib.pyplot as plt

spdShape = shp.Reader("SPD")

police_shp_gdf = gpd.read_file("EPIC.shp")
policeArrest = pd.read_csv("Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv").iloc[1:].reset_index(drop=True)