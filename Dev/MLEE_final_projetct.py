# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:40:22 2022

@author: jeje_
"""
# %% 
# import the packages 
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
# %% 
# set the path 
P_filepath = 'D:/jeje_/GSE/Master/1er Semestre - Automne 2022/Machine leaning'+\
' for Earth and Environmental Sciences/Final project/Globe_TP.nc'
N_filepath = 'D:/jeje_/GSE/Master/1er Semestre - Automne 2022/Machine leaning'+\
' for Earth and Environmental Sciences/Final project/Globe_TN_sample_00.nc'
# load the file with xarray 
tp = xr.open_dataset(P_filepath,engine='netcdf4')
tn = xr.open_dataset(N_filepath)
# %%
# exploring the data 
print(tp)
print(tn)
tp_shape = tp['TRUTH'].shape
# %% load and process the map 
import PIL
mask = PIL.Image.open('D:/jeje_/GSE/Master/1er Semestre - Automne 2022/Machine leaning'+\
' for Earth and Environmental Sciences/Final project/map.jpg')
array_mask = np.asarray(mask)
array_mask = array_mask[:,:,2]
array_mask = array_mask[45:-45]
# %% color the map for setting the basins as a categorical variable 
m = array_mask.copy()
# north hemisphere
m[0:46,0:31] = 1
m[0:46,31:101] = 2
m[0:46,101:181] = 3
m[0:29,181:260] = 4
m[29:32,181:269] = 4
m[32:37,181:277] = 4
m[37:46,181:282] = 4
m[38:43,282] = 4
m[0:29,263:360] = 1
m[29:32,272:360] = 1
m[32:37,277:360] = 1
m[37:46,301:360] = 1

# south hemisphere 
m[46:91,0:31] = 5 
m[46:91,31:101] = 6 
m[46:91,101:290] = 7 
m[46:91,294:360] = 5 
# %% add the map array as a variable in the dataset
mask_var = xr.DataArray(data = m, # Data to be stored

                 #set the name of dimensions for the dataArray
                  dims = ['lat', 'lon'],
                  
                 #Set the dictionary pointing the name dimensions to np arrays
                  coords = {'lat':np.flip(tp.lat.values),
                            'lon':tp.lon.values},

                 name='mask')

ds = xr.merge([tp, mask_var])

# save the dataset
ds.to_netcdf('D:/jeje_/GSE/Master/1er Semestre - Automne 2022/Machine leaning'+\
' for Earth and Environmental Sciences/Final project/dataset.nc')
#%% load dataset 
ds = xr.open_dataset('D:/jeje_/GSE/Master/1er Semestre - Automne 2022/Machine leaning'+\
' for Earth and Environmental Sciences/Final project/dataset.nc')
# %% 
# reorganizing data
ds = ds.where(ds.PCCD.notnull(), drop=True)
data_dict = {}
for key in ds.keys():
    data = ds[key].values
    data = data[~np.isnan(data)] # si 1ere ligne marche pas, essayer de run 
                                 # ça en premier
    data_dict[key] = data
data_list = np.array([data for key, data in data_dict.items()])
data_list = np.moveaxis(data_list, 0, -1)
# %% Clustering 
# i will try to do the clustering with different values of k (n_clusters)
# and plot silhouette score and inertia to find the best k and compare 
# with different graphs
k_list = range(2,11,1)
kmeans_models = []
for k in k_list:
    kmeans = KMeans(n_clusters =k,
                    random_state=42)
    kmeans.fit(ds)























