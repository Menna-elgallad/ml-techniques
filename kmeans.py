

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
import pandas as pd 
from collections import Counter



data , clusters = make_blobs(n_samples=200 , n_features=2 , centers=4 ,random_state=5 )
df = pd.DataFrame(data , columns=['f1' , 'f2'] )
print ( df.head())

plt.scatter (df['f1'] , df['f2'])
plt.show()

model = KMeans(n_clusters=4)
model.fit(df)

# make dataframe with clusters
predict =model.fit_predict(df[['f1','f2']])
df['clusters'] = predict
print ( df)


c = model.cluster_centers_  
print (c)  #center point for clusters 

labels = model.labels_
countt = Counter(labels ) #size of each cluster 
print ( countt)


plt.scatter(df['f1'] , df['f2'] ,50, labels)
plt.legend()
plt.show()

