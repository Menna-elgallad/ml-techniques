# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# actual values
actual =    [1,0,0,1,0,1,1,0,0,0]
# predicted values
predicted = [1,0,0,0,0,0,0,1,1,0]

# confusion matrix
matrix = confusion_matrix(actual,predicted, labels=[0,1])
print('Confusion matrix : \n',matrix)

# outcome values order in sklearn
tp, fp, fn, tn = confusion_matrix(actual,predicted,labels=[0,1]).ravel()
print('Outcome values : \n', tp, fp, fn, tn)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(actual,predicted,labels=[0,1])
print('Classification report : \n',matrix)