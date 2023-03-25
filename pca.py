


from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import pandas as pd
data = load_iris()

x = data.data
y = data.target 

x_train , x_test , y_train , y_test = train_test_split( x  , y ,test_size=0.2 ,random_state= 42)

#standard scalar to normalize values between 0 and 1 
s = StandardScaler()
x_train = s.fit_transform ( x_train) 
print ( x_train)
x_test = s.transform(x_test)
print (x_test)

# pca 
model = PCA(n_components=2)
x_train = model.fit_transform(x_train)
x_test = model.transform(x_test)
print ( x_train)
print (x_test)


# kmeans 

kmenas = KMeans(n_clusters=3)
kmenas.fit(x_train)
labels = kmenas.labels_

print ( labels)

# make data frame 
df = pd.DataFrame(x_train , columns=['f1' , 'f2']  )
print (df.head())

#plotting

plt.scatter(df['f1'] , df['f2'] , 50 , labels)
plt.show()