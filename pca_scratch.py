

from sklearn.datasets import load_iris
import numpy as np 
from numpy.linalg import eig 
import matplotlib.pyplot as plt

def pca ( x , n_components ):
    
    final_vector = []

    mean = np.mean(x , axis=0)
    data = x - mean

    CM= np.cov(data.T)

    eigenvalues , eigenvector= eig(CM)
    eigenvector = eigenvector.T
    indx = np.argsort(eigenvalues)[::-1]
    eigenvector = eigenvector[indx]

    final_vector= eigenvector[0:n_components]

    new_data = np.dot (data , final_vector.T)

    return new_data 

#implemintation 

data = load_iris()
x = data.data 

print (x[:5])
new_data =pca (x , 2)
print (new_data[:5] )

print ( x.shape)
print ( new_data.shape)

plt.scatter(new_data[: , 0] , new_data[: , 1])
plt.show()