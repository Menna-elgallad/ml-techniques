
from sklearn.datasets import load_iris
from collections import Counter
from sklearn.model_selection import train_test_split
from math import dist

data = load_iris()

x = data.data
y = data.target 
x_train , x_test , y_train , y_test = train_test_split (x,y , test_size=0.2 , random_state=49)

def predict ( k, x_train , x_test , y_train ):
    
    finallclass = []
    
    for i in range ( len(x_test)):
        
        distances= []
        classes = []

        for k in range ( len(x_train)):
            My_dist = dist(x_train[k] , x_test[i])
            distances.append([My_dist , k])
        
        distances.sort()
        distances = distances[0:k]
        print(distances)
        for d , c in distances  :      
            classes.append(y_train[c])
        
        print ( " nearest classes  \n" , classes)
        
        ans = Counter(classes).most_common(1)[0][0]
        finallclass.append ( ans)
        print ( "final belonged class " , ans)
        
    return finallclass

def accuarcy ( y_test  ) : 
    prediction = predict(3 , x_train , x_test , y_train)
    score = (prediction==y_test ).sum() / len(y_test)
    return score*100


print (predict ( 3 , x_train , x_test , y_train))
print ( "the accuarcy of the algorithm is  " , accuarcy(y_test)   )

