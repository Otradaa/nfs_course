import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("voice.csv")
print(data.shape)
data['label'].replace({'male': 0, 'female': 1}, inplace=True)

data = data.astype('float64')
scaler = MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

Y = data.label.values
X = data.drop(["label"], axis=1)

classifier = KNeighborsClassifier(p=1, n_neighbors=4, weights='distance')
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    print(f"silhouette={metrics.silhouette_score(X, kmeans.labels_)}")

    x_train, x_test, y_train, y_test = train_test_split(X, kmeans.labels_, test_size=0.3,
                                                        random_state=1, stratify=Y)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    precision = precision_score(y_test, y_pred, average='micro')  # not p n
    recall = recall_score(y_test, y_pred, average='micro')  # all pos
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f"{k}: precision={precision:.10f}, recall={recall:.10f}, f1={f1:.10f}")

