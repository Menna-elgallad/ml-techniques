
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
wine = load_wine ()

x = wine.data 
y = wine.target
print ( wine.feature_names)
print ( wine.target_names)

x_train,x_test , y_train , y_test =train_test_split( x,y , test_size=0.3,random_state=42)

knn = KNeighborsClassifier ( 7)
knn.fit ( x_train , y_train )

predict = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test,predict)
print ( "predicted",predict)
print ( " ture" , y_test)

conM = confusion_matrix ( y_test , predict , labels=[0,1,2] )

print ( conM)
print ( accuracy)