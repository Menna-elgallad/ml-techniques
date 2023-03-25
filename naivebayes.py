
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import metrics 
from sklearn.model_selection import  train_test_split 

iris = load_iris()
x = iris.data
y = iris.target 

x_train,x_test , y_train,y_test = train_test_split ( x,y,test_size=0.4 , random_state=49)

model = GaussianNB()
model.fit(x_train,y_train)

Predict = model.predict(x_test)
print ( "predicted data" , Predict)
print ( "actual " , y_test)
score =metrics.accuracy_score(y_test,Predict)
print(score*100,"%")