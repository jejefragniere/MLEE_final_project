#!/usr/bin/env python
# coding: utf-8

# Import the packages that will be used. 

# In[46]:


import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import PIL 
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import pandas as pd


# Import the raw dataset of true positive cyclogenesis points.

# In[2]:


tp = xr.open_dataset('Globe_TP.nc').drop(['DNST','TRUTH'])


# load and process the map used for the mask. 

# In[3]:


mask = PIL.Image.open('map.jpg')
array_mask = np.asarray(mask)
array_mask = array_mask[:,:,2]
array_mask = array_mask[45:-45]


# Color the pixels of the map for the creation of the categorical oceanic bassins variable.

# In[4]:


m = array_mask.copy()
# north hemisphere
m[0:46,0:31] = 1
m[0:46,31:101] = 2
m[0:46,101:181] = 3
m[0:29,181:262] = 4
m[29:32,181:271] = 4
m[32:37,181:277] = 4
m[37:46,181:288] = 4
m[38:43,282] = 4
m[0:29,262:360] = 1
m[29:32,271:360] = 1
m[32:37,277:360] = 1
m[37:46,288:360] = 1

# south hemisphere 
m[46:91,0:31] = 5 
m[46:91,31:101] = 6 
m[46:91,101:292] = 7 
m[46:91,292:360] = 5 


# Add the bassins array as a variable in the dataset 

# In[5]:


mask_var = xr.DataArray(data = m, # Data to be stored

                 #set the name of dimensions for the dataArray
                  dims = ['lat', 'lon'],
                  
                 #Set the dictionary pointing the name dimensions to np arrays
                  coords = {'lat':np.flip(tp.lat.values),
                            'lon':tp.lon.values},

                 name='mask')

ds = xr.merge([tp, mask_var])
mask = ds.mask.values 


# Reorganizing the data into (n_samples,n_features) shape for the KMeans

# In[13]:


print([*tp.keys()])
data_array = tp.to_array().values

data_array = np.moveaxis(data_array, 0, -1)



data_array = np.pad(data_array,
                    ((0,0),
                     (0,0),
                     (0,0),
                     (0,1)), 
                    'constant',
                    constant_values=np.nan)

for i in range(len(data_array)): 
    data_array[i,:,:, -1] = mask*(~np.any(np.isnan(data_array[i,:,:,:-1]),axis=-1))
    
data_array = data_array.reshape(-1,data_array.shape[-1])
data_array = data_array[(~np.any(np.isnan(data_array),axis=-1))]
np.save('data.npy',data_array)


# Load the array to avoid doing the preprocessing everytime during the preparation of the notebook 

# In[44]:


ds = np.load('data.npy')
ds_x = np.delete(ds,11,1) # drop the bassins variable for the KMeans so it doesn't influence de group
ds_y = ds[:,11] # extract the bassins variable to use as labels 
ds_x.shape, ds_y.shape


# Cluster with KMeans model, with n_clusters 2 ≤ k ≤ 10

# In[16]:


k_list = range(2,11,1)
kmeans_models = []
for k in k_list:
    kmeans = KMeans(n_clusters =k,
                    random_state=42)
    kmeans.fit(ds_x)
    kmeans_models.append(kmeans)


# Extract the silhouette scores 

# In[17]:


silhouette_scores = [silhouette_score(ds_x, model.labels_)
                     for model in kmeans_models]


# In[19]:


np.save('sil_scores.npy',silhouette_scores) # save because it takes time to get it 


# Extract the best index for the best silhouette score, and the best model 

# In[20]:


best_index = np.array(np.argmax(silhouette_scores))
best_k = k_list[best_index]
best_score = silhouette_scores[best_index]


# Plot the silhouette scores with the best one in red 

# In[21]:


fig, ax = plt.subplots(figsize =(12,4))

ax.plot(k_list, silhouette_scores, "bo-") 

ax.set_xlabel("", fontsize=14) 
ax.set_ylabel("Silhouette score", fontsize=14)

ax.plot(best_k, best_score, "rs")
plt.show()


# Extract all the inertias and find the best one 

# In[22]:


inertias  = [model.inertia_ for model in kmeans_models]
best_inertia = inertias[best_index]


# Plot the inertias 

# In[24]:


fig, ax = plt.subplots(figsize=(12,4))

ax.plot(k_list, inertias, "bo-") 

ax.set_xlabel("", fontsize=14) 
ax.set_ylabel("Inertia", fontsize=14)

ax.plot(best_k, best_inertia, "rs")
plt.show()


# Reduce the dimensionality of the dataset with sickit's TSNE implementation

# In[27]:


tsne = TSNE(n_components=2,
       random_state=42,
       learning_rate='auto')

ds_x_plot = tsne.fit_transform(ds_x)


# In[33]:


print(ds_x_plot.shape)
np.save('ds_x_plot.npy',ds_x_plot)


# Make a list of the best three models 

# In[34]:


best_models = []
best_models.append(kmeans_models[best_index]) 
second_index,third_index = np.asarray(silhouette_scores).argsort()[-2],np.asarray(silhouette_scores).argsort()[-3]
best_models.append(kmeans_models[second_index])
best_models.append(kmeans_models[third_index])


# Make some predictions for the three best models 

# In[37]:


pred_labels = []
for model in best_models:
    pred_labels.append(model.predict(ds_x))


# Make a dataframe with the reduced input components, the truth labels and the predicted cluster labels. 

# In[61]:


plot_data = np.stack([ds_x_plot[:,0], ds_x_plot[:,1], ds_y, *pred_labels],axis=1)
df = pd.DataFrame(plot_data, columns=['X1','X2','truth','pred_1','pred_2','pred_3'])
df.to_pickle("df.pkl")


# Plot the true answers and the clusters from the different models 

# In[62]:


fig, axes = plt.subplots(2, 2, figsize=(16,16))

groups = df.groupby('truth')
for label, group in groups:
    axes[0,0].plot(group.X1, group.X2, marker='o', linestyle='', markersize=3, label=int(label))
axes[0,0].legend(fontsize=12)
axes[0,0].set_title('Truth', fontsize=16)
axes[0,0].axis('off')

groups = df.groupby('pred_1')
for label, group in groups:
    axes[0,1].plot(group.X1, group.X2, marker='o', linestyle='', markersize=3)
axes[0,1].set_title('"Best" Clustering', fontsize=16)
axes[0,1].axis('off')

groups = df.groupby('pred_2')
for label, group in groups:
    axes[1,0].plot(group.X1, group.X2, marker='o', linestyle='', markersize=3)
axes[1,0].set_title('2$^nd$ Best Clustering', fontsize=16)
axes[1,0].axis('off')

groups = df.groupby('pred_3')
for label, group in groups:
    axes[1,1].plot(group.X1, group.X2, marker='o', linestyle='', markersize=3)
axes[1,1].set_title('3$^rd$ Best Clustering', fontsize=16)
axes[1,1].axis('off')

