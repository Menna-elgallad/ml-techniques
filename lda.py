


from pyexpat import model
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.svm import SVC
data = load_iris()

x = data.data
y = data.target 

x_train , x_test , y_train , y_test = train_test_split( x  , y ,test_size=0.2 ,random_state= 42)

#standard scalar to normalize values between 0 and 1 
s = StandardScaler ()
x_train = s.fit_transform(x_train)
x_test =s.transform ( x_test)

# lda 
lda = LDA(n_components=1)
x_train = lda.fit_transform(x_train,y_train)
x_test = lda.transform(x_test)
print ( x_train)


sv =SVC(kernel='linear')
sv.fit(x_train , y_train)
x_train = sv.fit(x_train , y_train)
predictt =sv.predict(x_test)
print ( x_train)
# print (predictt)
# plt.plot(sv)
# plt.show()