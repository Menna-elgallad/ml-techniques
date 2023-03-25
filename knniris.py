
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

Mydata =load_iris()

x = Mydata.data
y = Mydata.target 

x_train ,   x_test ,y_tarin, y_test = train_test_split(x,y,test_size=0.2 ,random_state=42)

model = KNeighborsClassifier(7)
model.fit ( x_train,y_tarin)
predict = model.predict(x_test)
print (y_test)
print (predict)

score = metrics.accuracy_score(y_test,predict)
print (score)
