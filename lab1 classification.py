import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("voice.csv")
# data.describe()
# data.info()
print(data.shape)
data['label'].replace({'male': 0, 'female': 1}, inplace=True)

data = data.astype('float64')
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

y = data.label.values
x = data.drop(["label"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# nn = 20
# y_pred = []
# for i in range(2, nn):
#     knn = KNeighborsClassifier(p=2, n_jobs=-1, n_neighbors=i, weights="distance")
#     knn.fit(x_train, y_train)
#     y_pred = knn.predict(x_test)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     print(f"({i}): precision={precision:.10f}, recall={recall:.10f}, f1={f1:.10f}")

knn = KNeighborsClassifier(p=1, n_neighbors=4, weights='distance')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
precision = precision_score(y_test, y_pred)  # not p n
recall = recall_score(y_test, y_pred)  # all pos
f1 = f1_score(y_test, y_pred)
print(f"KNN: precision={precision:.10f}, recall={recall:.10f}, f1={f1:.10f}")

conf_mat = confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
