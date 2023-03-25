

from linalg import pivotize
from matplotlib.pyplot import prism
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier



weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']




coder = LabelEncoder()
weather_encoder = coder.fit_transform(weather)
temp_encoder = coder.fit_transform(temp)

target_encoder = coder.fit_transform(play)
print(weather_encoder)


featuers = list (zip(weather_encoder,temp_encoder))
model = KNeighborsClassifier(3)
model.fit ( featuers,target_encoder)
predict =model.predict([[2,2]])
print ( predict)
