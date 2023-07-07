#!/usr/bin/env python
# coding: utf-8

# # Dimensions of change exploratory analysis
# 
# - In reading, Hindy et al. (2012) show a correlation between brain activation (BOLD) and the severity of the change described by each sentence. For example, a sentence describing a "cut the cake" event would likely elicit greater activitation than a "photograph the cake" event. Some events describe large physical changes (like cutting a cake) but some drastic changes could also have little corresponding physical change (like signing legal paperwork). Does the brain (and cognitive system) treat all types of change the same? What types does it care about?
# - Method: Collected ratings using Qualtrics for 120 sentences describing different events. Participants rated sentences along 22 possible dimensions of change on a scale from 1-7. 
# - Each row (120) contains ratings for a sentence.
# - Ratings of change along 22 different dimensions & BOLD values from Hindy et al. (2012).
# 

# In[5]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

plt.style.use("ggplot")

data = pd.read_csv('data.csv')
data = data.drop(columns=['Question','QuestForCorr'])
data = data.rename(columns={'1: Size':'Size','10: Temperature': 'Temperature','11: Tension':'Tension','12: Scent':'Scent',
                   '13: Flavor':'Flavor','14: Sound':'Sound','15: Stability':'Stability','16: Motoric_interaction':'Motor',
                   '17: Function':'Function','18: Value':'Value','19: Knowledge':'Knowledge','2: Shape':'Shape',
                   '20: Emotional_affect':'Affect','21: Mood':'Mood','22: Weight':'Weight','3: Position':'Position',
                   '4: Form':'Form','5: Color':'Color','6: Light':'Light','7: Surroundings':'Surroundings',
                   '8: Location':'Location','9: Texture':'Texture','Bold':'BOLD'})

bold = data['BOLD']
feats = data.drop(columns='BOLD')
feature_names = np.array(list(feats.columns.values))

sns.pairplot(data)
plt.show()


# Some of the distributions are a fairly skewed (e.g. light), but nothing too crazy. As a first step, see if any of the new dimensions already have an obvious relationship to BOLD based on the scatterplots (below). Some look potentially positively correlated (e.g. texture). I'm return back to this with linear models later.

# In[68]:


sns.pairplot(data,y_vars=['BOLD'],kind='reg')
plt.show()


# In[67]:


data_corrmat = data.corr()
originalhm = sns.heatmap(data_corrmat, 
                 cbar=True, 
                 annot=False, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=data_corrmat.columns, 
                 xticklabels=data_corrmat.columns, 
                 cmap="Spectral_r")


# In[10]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(feats)
feats_scaled = scaler.transform(feats)


# # PCA
# - Including only these new dimensions (no BOLD), I ran a PCA to see if any of the new dimensions could be reduced down in an unsupervised way (e.g. physical types of pattern similarly). Analysis of this PCA shows that it takes quite a few components to explain most of the variation (11 components to explain about 90% of the original variance). We are explaining quite a bit of variation in the first few however, so we do seem to be getting at some underlying similarity. 

# In[11]:


from sklearn.decomposition import PCA

pca_all = PCA()
pca_all.fit(feats_scaled)
feats_pca_all = pca_all.transform(feats_scaled)

np.cumsum(pca_all.explained_variance_ratio_ * 100)

plt.plot(np.cumsum(pca_all.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')


# # PCA Stats
# - Let's see if any of the variation we're picking up with any of these components naturally maps on to the BOLD signal. From the scatterplot of components 0 and 1, nothing seemed obvious to me at first glance. In the correlation matrix below we can see a small positive correlation between component 0 and BOLD (r=0.13,p=0.16), however it's not significant.
# - Considering the PCA was trained without any reference to the BOLD signal, it is interesting to note this correlation with component 0, even though it's not significant. Beyond that suggestive finding, nothing obvious is coming out from this analysis.

# In[72]:


pca_2 = PCA(n_components=2)
pca_2.fit(feats_scaled)
feats_pca_2 = pca_2.transform(feats_scaled)

sns.scatterplot(x=feats_pca_2[:,0],y=feats_pca_2[:,1], hue = bold)
plt.xlabel('Component 0')
plt.ylabel('Component 1')


# In[71]:


pca_5 = PCA(n_components=5)
pca_5.fit(feats_scaled)
feats_pca_5 = pca_5.transform(feats_scaled)

corr_df = pd.DataFrame(feats_pca_5)
corr_df['bold'] = bold

sns.pairplot(corr_df, y_vars=['bold'])
plt.show()


# In[15]:


corr_df_corrmat = corr_df.corr()
corr_dfhm = sns.heatmap(corr_df_corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=corr_df_corrmat.columns, 
                 xticklabels=corr_df_corrmat.columns, 
                 cmap="Spectral_r")


# In[20]:


from scipy import stats
corr_df = corr_df.set_axis(['C0', 'C1', 'C2', 'C3', 'C4', 'bold'], axis=1)
stats.pearsonr(corr_df['C0'], corr_df['bold'])


# # K-means clustering
# - Here I'm going to try a k-means clustering algorithm to see if any obvious groupings emerge in the new 22 dimensions. Here the algorithm was made to generate 2 clusters with the 120 items. I then took the labels generated by the clustering algorithm and compared BOLD ratings across these groups. We do show significantly higher BOLD activation for items in the label 2 group (t=-2.42, p=0.02) compared to label 1. This result is suggestive that it would be worthwhile to explore those clusters. 

# In[22]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, n_init="auto").fit(feats_scaled)
cluster_labels=kmeans.labels_

clust_df = pd.DataFrame(list(zip(cluster_labels,bold)),columns=['label','bold'])
clust_df.head()


# In[33]:


lab1 = clust_df[clust_df["label"] == 0]
lab2 = clust_df[clust_df["label"] == 1]

from scipy import stats
import seaborn as sns

sns.barplot(data=clust_df, x="label", y="bold", errorbar = "ci")


# In[30]:


stats.ttest_ind(lab1['bold'], lab2['bold'], equal_var=False)


# Interestingly the boundary between group 1 and 2 generated by the k-means algorithm can actually be seen clearly on a graph, bisected along component 0 generated by the PCA. This leads me to think that the variation we were getting at with the PCA was likely approximating a general "change" rating average regardless of dimension. This clustering algorithm seems to be getting at a similar thing, but probably more reliably based on the results of the t-test. Component 0 does seem worth investigating, both because of the small correaltion and also because it bisects the clusters. To do this I extracted the weights from component 0, where dimensions with higher weightings are be more influential in determining the component.

# In[59]:


sns.scatterplot(x=feats_pca_2[:,0],y=feats_pca_2[:,1], hue = cluster_labels)
plt.xlabel('Component 0')
plt.ylabel('Component 1')


# In[56]:


weights = pca_2.components_
weights = pd.DataFrame(weights, columns=feats.columns)
weights['Component'] = [0,1]
weights.head()


# In[66]:


melted = weights.melt(id_vars=['Component'], var_name='Dimension', value_name='Weights')
melted.sort_values(by=['Component', 'Weights'], ascending= [True, False], inplace = True)

melted.head()
#melted.to_csv("PCAWeights.csv")


# # Tentative Takeaway
# The k-means clustering seems useful for describing BOLD based on sentence feeatures, however I'm curious the extent to which it is just approximating a split along a general change rating. Component 0 does correlate decently with bold and also bisects the k-means clusters. The dimensions of Form, Weight, and Texture have the highest weights, and intuitively do all seem to be related to physical characteristics. This suggests that BOLD is likely particularly sensitive to changes along these physical dimensions. Value, however, has the 4th highest weight for component 0. This is a much more abstract feature than the previous physical ones, but it likely does have a strong attentional component (e.g. losing a very valuable lottery ticket has very little physical change but would likely draw a large amount of focus). This reaffirms the importance of physical, visual change, and also leaves me more curious about the role of attention and it's relationship to these change ratings. For next steps I'm going to do some feature importance tests of these dimensions, but in another document.
