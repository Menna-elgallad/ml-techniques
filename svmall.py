import imp
from importlib.machinery import FrozenImporter
from sklearn.svm import SVC 
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()

x = data.data 
y = data.target 

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

"linear"
model = SVC (kernel='linear')
model.fit(x_train,y_train)

predict = model.predict(x_test)
print ( "linear")
print ( " prediction vs actual classes \n"  , predict ,"\n"  , y_test)
print ( " accuarcy "   , accuracy_score(y_test , predict))
print ( "confusion matrix"  , confusion_matrix(y_test , predict))


"poly kernel "
model = SVC (kernel='poly' , degree=8)
model.fit(x_train,y_train)

predict = model.predict(x_test)
print ( "poly kernel")
print ( " prediction vs actual classes \n"  , predict ,"\n"  , y_test)
print ( " accuarcy "   , accuracy_score(y_test , predict))
print ( "confusion matrix"  , confusion_matrix(y_test , predict))

"rbf kernel "
model = SVC (kernel='rbf' )
model.fit(x_train,y_train)
print ( "rbf kernel")
predict = model.predict(x_test)
print ( " prediction vs actual classes \n"  , predict ,"\n"  , y_test)
print ( " accuarcy "   , accuracy_score(y_test , predict))
print ( "confusion matrix"  , confusion_matrix(y_test , predict))

"sigmoic kernel "
model = SVC (kernel='sigmoid' )
model.fit(x_train,y_train)

predict = model.predict(x_test)
print ( "sigmoid kernel")
print ( " prediction vs actual classes  \n"  , predict ,"\n"  , y_test)
print ( " accuarcy "   , accuracy_score(y_test , predict))
print ( "confusion matrix"  , confusion_matrix(y_test , predict))

