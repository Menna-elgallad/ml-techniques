from statistics import mode
from cv2 import kmeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC


data = load_iris()
x = data.data
y = data.target

x_train,x_test , y_train,y_test = train_test_split ( x,y,test_size=0.4 , random_state=100)
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.transform(x_test)

model = PCA(n_components=2)
new_train= model.fit_transform(x_train , y_train)
new_test= model.fit_transform(x_test)
print ( x_train.shape)
print (new_train.shape)

model = SVC(kernel='linear')
model.fit(new_train , y_train)
model.predict(new_test)


