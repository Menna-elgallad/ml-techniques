
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd 
from sklearn.metrics import accuracy_score , confusion_matrix

dataset = pd.read_csv("iris.csv")
print(dataset.head())

x = dataset.drop(["variety"],axis=1)
y = dataset['variety']

# print (x)
# print (y)

x_train,x_test,y_train ,y_test = train_test_split ( x,y,test_size=0.3 , random_state=90 )
model = SVC (kernel="linear")
model.fit ( x_train,y_train)

Predict = model.predict(x_test )

print ("pred", Predict)
print ( "actual",y_test)
print ( accuracy_score(y_test,Predict))
print ( confusion_matrix(y_test,Predict))