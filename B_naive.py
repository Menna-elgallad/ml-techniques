
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

X = np.random.randint(2, size=(500,10))
Y = np.random.randint(2, size=(500, 1))

print ("data", X)
print ( "target",Y)


X_test= X[:50, :10]
y_test= Y[:50, :1]

print ("xtest",X_test)
print ( "ytest",y_test)

clf= BernoulliNB()
model= clf.fit(X, Y)

y_pred=clf.predict(X_test)
acc_score= accuracy_score(y_test, y_pred)
print(acc_score)