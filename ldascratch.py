
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

mydata = load_iris()
x = mydata.data 
y = mydata.target  

def LDA ( data , target , n_features):
    n = n_features
    linear_disc = []
    n_f = data.shape[1]
    totalmean = np.mean(data , axis=0)
    classlabels = np.unique(target)
    
    s_w= np.zeros((n_f , n_f))
    s_b = np.zeros((n_f , n_f))

    for c in classlabels:
        
        x_c = data[target == c]
        class_mean = np.mean(x_c , axis=0)

        # to get number of samples 
        n_s = x_c.shape[ 0] 
        s_w += (x_c - class_mean).T.dot((x_c - class_mean) ) #within class scatter 
        mean_diff = (class_mean-totalmean).reshape(n_f , 1)
        s_b += n_s * (mean_diff).dot(mean_diff.T)   #between class scatter 

    matrix= np.linalg.inv(s_w).dot(s_b)

    # Get eigenvalues and eigenvectors of SW^-1 * SB

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # -> eigenvector v = [:,i] column vector, transpose for easier calculations
    # sort eigenvalues high to low

    eigenvectors = eigenvectors.T
    idxs = np.argsort(abs(eigenvalues))[::-1] #sort descending to get the maximum 
    # get maximum eigen vectors depending on eigen values 
    eigenvectors = eigenvectors[idxs]

    # store first n eigenvectors
    linear_disc = eigenvectors[0 : n]

    # project data
    # return the new set of data
    return np.dot(data, linear_disc.T)
   

new_data = LDA(x , y , 2 )
print (x.shape)
print (new_data.shape)

plt.scatter(new_data[: , 0] , new_data[: , 1])
plt.show()


