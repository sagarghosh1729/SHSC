#Importing the Modules and Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
from scipy import linalg
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from itertools import combinations
import random


def standardize(x):
  max=np.max(x,axis=0)
  min=np.min(x,axis=0)
  range=max-min
  s=(x-min)/max
  scaled=2*s-1
  return scaled


#Defining the Hyperbolic Functions and Embedding the dataset into a Poincare Disc
#Defining Mobius Addition and Mobius Multiplication
def mob_multiplication(scalar,point,curvature):
  c=-curvature
  d=math.tanh(scalar*(math.atanh(math.sqrt(c)*np.linalg.norm(point))))
  s=(d/np.linalg.norm(point))*point
  return s

def mob_addition(point1,point2,curvature):
  c=-curvature
  n1=np.linalg.norm(point1)
  n2=np.linalg.norm(point2)
  numerator=(1+2*c*np.dot(point1,point2)+c*n2**2)*point1 + (1-c*n1**2)*point2
  denominator=1+2*c*np.dot(point1,point2)+(c*n1*n2)**2
  return numerator*(1/denominator)



#Defining the Exponential Embedding
import math
def exp_embedding(point, center,curvature):
  c=-curvature
  l=np.linalg.norm(point)
  if(l==0):
    s=0
  else:
    s=math.tanh(math.sqrt(c)*l)*(1/(math.sqrt(c)*l))
  return mob_addition(center,s*point,curvature)

def poincare_dist(point1,point2,center,curvature):
  c=-curvature
  p=exp_embedding(point1,center,curvature)
  q=exp_embedding(point2,center,curvature)
  d=(2/c)*(np.linalg.norm(p-q)**2)/(((1/c)-np.linalg.norm(p)**2)*((1/c)-np.linalg.norm(q)**2))
  s=2*math.asinh(math.sqrt(d))
  return s

#Defining the Gromov Hyperbolicity and Finding the Best Frechet Centroid

#random.seed(42)
def calculate_root_lowest_k(points,sample_size,iteration,k,curvature):
  for i in range(iteration):
    recurring_centroid=np.zeros((points.shape[1]))
    random_choice=np.random.choice(points.shape[0],sample_size,replace=False)
    distances=np.zeros((sample_size,sample_size))
    for l in range(sample_size):
      for j in range(sample_size):
        distances[l,j]=poincare_dist(points[random_choice[l],:],points[random_choice[j],:],0,curvature)
    hyperbolicity_indices=[]
    for l, j, k in combinations(range(sample_size), 3):
        d_xy = distances[l, j]
        d_yz = distances[j, k]
        d_xz = distances[l, k]

        # Compute Gromov hyperbolicity for the triplet
        delta = 0.5 * (d_xy + d_yz - d_xz)
        hyperbolicity_indices.append((delta, (i, j, k)))
    hyperbolicity_indices.sort(reverse=False, key=lambda x: x[0])
    lowest_k_triplets = [triplet for _, triplet in hyperbolicity_indices[:k]]
    for l in range(k):
      update=np.zeros((points.shape[1]))
      best_1=points[lowest_k_triplets[l][0],:]
      best_2=points[lowest_k_triplets[l][1],:]
      best_3=points[lowest_k_triplets[l][2],:]
      update=mob_addition(update,mob_addition(mob_addition(best_1,best_2,curvature),best_3,curvature),curvature)
    recurring_centroid=mob_addition(recurring_centroid,update,curvature)
  final_centroid=mob_multiplication(1/(sample_size*iteration),recurring_centroid,curvature)
  return final_centroid


#Shifting all the points with respect to the centroid
def shifted_embedding(data, center,curvature):
  for i in range(data.shape[0]):
    data[i,:]=mob_addition(data[i,:],-center,curvature)
  return data

#Nearest Neighbour Label Assignment
def nearest_neighbour_label(labelled_data,unlabelled_data, labels, center, curvature):
  assigned_labels=[]
  for point in unlabelled_data:
    distances=[poincare_dist(point,labelled_data[i],center,curvature) for i in range(len(labelled_data))]
    min_index=np.argmin(distances)
    assigned_labels.append(labels[min_index])
  return assigned_labels


######################################
#read the file and store the input data in x and output labels in y


curvature=-1
#k being the number of unique labels of y
prime_p=50*k
epsilon=100000
hyp_par=0.1 #setting up the sigma
n_clusters=k

#Standardize the dataset to avoid redundancy in floating point computation
zs=standardize(x)

#n=no of samples, m= dimension of each input data

#Embedding the dataset into the Poincare Disc

z=exp_embedding(zs,0,curvature)


#calculating the Frechet Centroid
centroid=calculate_root_lowest_k(z,50,100,20,curvature)

#Shifting the entire dataset with respect to the centroid
z_shifted=shifted_embedding(z,centroid,curvature)


#Performing the Scalable SPectral Clustering
#Selecting some random rows from the data
random.seed(42)
random_choice=np.random.choice(n,prime_p,replace=False)
sample_data=z_shifted[random_choice,:]

data=sample_data
center=centroid

m=data.shape[0]

R=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        R[i][j]=poincare_dist(data[i,:],data[j,:],center,curvature)


W=np.zeros((m,m))
for i in range(m):
     for j in range(m):
          if(R[i][j]<epsilon):
               W[i][j]=np.exp(-hyp_par*R[i][j]*R[i][j])
          else:
               W[i][j]=0



#Applying the Spectral CLustering on the affinity matrices as per HSCA
spectral_cluster=SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
cluster_labels_h=spectral_cluster.fit_predict(W)

#Assigning Neighbours Based On Nearest Neighbour Approach

labels=np.zeros((n))

lb=nearest_neighbour_label(sample_data,z,cluster_labels_h,0.0,-1)


#calculating ARI, NMI

ari=adjusted_rand_score(y,lb)
nmi=normalized_mutual_info_score(y,lb)
print(ari)
print(nmi)

#Vizualization of the clusters
original_data=df.iloc[:,:-1].values

tsne=TSNE(n_components=2,random_state=42)
tsne_data=tsne.fit_transform(original_data)
plt.scatter(tsne_data[:,0], tsne_data[:,1],c=y, cmap='viridis', edgecolors='k')
plt.title('t-SNE Visualization of the dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.subplots()
plt.scatter(tsne_data[:,0], tsne_data[:,1],c=lb, cmap='viridis', edgecolors='k')
plt.title('t-SNE Visualization of the SHSC Clusters')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()






