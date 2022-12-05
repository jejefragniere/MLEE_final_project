# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:40:22 2022

@author: jeje_
"""
# %% 
# import the packages 
import xarray as xr
import rioxarray as rioxr 
import numpy as np 
import matplotlib.pyplot as plt 
import cftime
# %% 
# setting the path 
P_filepath = 'D:/jeje_/GSE/Master/1er Semestre - Automne 2022/Machine leaning for Earth and Environmental Sciences/Final project/Globe_TP.nc'
N_filepath = 'D:/jeje_/GSE/Master/1er Semestre - Automne 2022/Machine leaning for Earth and Environmental Sciences/Final project/Globe_TN_sample_00.nc'
# loading the file with rioxarray 
P_data = rioxr.open_rasterio(P_filepath)
N_data = rioxr.open_rasterio(N_filepath)
# %%
# exploring the data 
print(P_data.dims)
print(P_data.attrs)
print(P_data.coords)

print(N_data.dims)
print(N_data.attrs)
print(N_data.coords)

print(P_data)

