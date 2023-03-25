


# from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston



boston = load_boston()
x = boston.data
y = boston.target 

# print ( x)
# print ( y)
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=49)
model = KNeighborsRegressor(n_neighbors=5 , weights="uniform" , algorithm='auto')
model.fit(x_train,y_train)

predecit = model.predict(x_test) 
score = model.score (x_test , y_test)

print("the accuarcy of testing process ",score)
print ("predicted values ",predecit )

print ("true values ", y_test)