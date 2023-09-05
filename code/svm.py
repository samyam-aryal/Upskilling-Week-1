from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('E:\Apprenticeship\data\preprocessed.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

features = df.iloc[:, :-1]
label = df.iloc[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(features, label)

svc = SVC(class_weight='balanced')
# {3: 103, 4: 24, 5: 2, 6: 2, 7: 6, 8: 19}

model = svc.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
rec = recall_score(y_test, y_pred, average='weighted')
prec = precision_score(y_test, y_pred, average='weighted')
#roc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

print(f"Accuracy: {acc_score}\nPrecision: {prec}, Recall: {rec}\n\n and CM Score: \n{conf}")
sns.heatmap(conf)
plt.show()

# Apparently, SVM doesn't work too well for 6 classes in this dataset. 
# We could group the classes into two: good and bad,
# based on the quality rating in the dataframe. 
