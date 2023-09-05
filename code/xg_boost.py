import xgboost
from xgboost import XGBClassifier
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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(label)
label = label.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(features, label)
class_weights = {3: 103, 4: 24, 5: 2, 6: 2, 7: 6, 8: 19}
model = XGBClassifier(random_state=42, class_weight=class_weights)
model.fit(X_train, y_train)

print("\n\n\n\n")

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("CM:", confusion_matrix(y_test, y_pred))