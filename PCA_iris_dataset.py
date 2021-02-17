# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:08:56 2021

@author: User
"""

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Get the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url,names=['sepal length','sepal width','petal length','petal width','target'])
target=data.iloc[:,4]
#Retreiving and standardising data
X=data.iloc[:,0:4]
X_zero_meaned = X- np.mean(X,axis=0)

#Calculate Covariance Matrix
cov_matrix = np.cov(X_zero_meaned, rowvar= False)

#Calculating eigenvalues and eigenvectors
eigenvalues,eigenvectors = np.linalg.eig(cov_matrix)

#Sorting eigenvalues and eigenvectors in descending order
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:,sorted_index]

#Choose no. of principal components(2) 
eigenvectors_reduced = sorted_eigenvectors[:,0:2]

#Calculating principal Components
final_data = (eigenvectors_reduced.transpose() @ X_zero_meaned.transpose()).transpose()
final_data.columns = ['PC1','PC2']

#Plotting Results

sns.scatterplot(data=final_data,x='PC1',y='PC2', hue=target)
plt.title('Scatter plot using 2 principal components')
plt.show()

plt.plot(np.cumsum(sorted_eigenvalues)/np.sum(sorted_eigenvalues),'-o',color='k')
plt.show()