from statistics import mode
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd 
import matplotlib.pyplot as plt 



data , clusters = make_blobs (n_samples=200 , n_features=2 , centers=4 , random_state=5)
df = pd.DataFrame(data , columns=['f1' , 'f2' ])
print(df.head())


plt.scatter(df['f1'] , df['f2'])
plt.show()

model = KMeans(n_clusters=4)
model.fit(df)

prediction = model.fit_predict(df[['f1' , 'f2']])
# print ( prediction)
df['class'] = prediction
labels = model.labels_

plt.scatter(df['f1'] , df['f2'] , 50 , labels)
plt.show()