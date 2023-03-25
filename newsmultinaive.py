from re import sub
from statistics import mode
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer



data = fetch_20newsgroups()
print ( data.target_names)

topics = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
train = fetch_20newsgroups(subset='train' , categories=topics)
test = fetch_20newsgroups (subset='test' , categories=topics)

x1 = train.data
y1 = train.target
x2 = test.data
y2 = test.target 

model = make_pipeline(TfidfVectorizer() , MultinomialNB())
model.fit ( x1,y1)

# print ( x2)
predicts = model.predict(x2)
print (y2)
print ( predicts)

def predect_category(s , train = train , predicts=predicts) :
   pre= model.predict([s])
   print (pre)
   print ( train.target_names)
   return train.target_names[pre[0]]

print(predect_category("my hardware "))   

