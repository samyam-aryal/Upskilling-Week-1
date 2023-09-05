from sklearn.ensemble import RandomForestClassifier
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

model = RandomForestClassifier(n_estimators=1000, class_weight='balanced')

model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

pred = model.predict(X_test)
conf = confusion_matrix(y_test, pred)
