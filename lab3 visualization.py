import pandas as pd
from sklearn import metrics, preprocessing, decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("voice.csv")
print(data.shape)
data['label'].replace({'male': 0, 'female': 1}, inplace=True)

data = data.astype('float64')
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

Y = data.label.values
X = data.drop(["label"], axis=1)

components = 2
pca = decomposition.PCA(n_components=components)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=1)
X = kmeans.fit_predict(X_pca)

for i in range(components):
    for j in range(i + 1, components):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X_pca[Y == 0, i], X_pca[Y == 0, j], cmap='viridis', s=10, alpha=0.5)
        scatter = ax.scatter(X_pca[Y == 1, i], X_pca[Y == 1, j], c='r', s=10, alpha=0.5)
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5)
        ax.set_title('K-Means Clustering')
        ax.set_xlabel(i)
        ax.set_ylabel(j)
        plt.show()



# for i in range(len(X)):
#     for j in range(i+1,len(X)):
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         scatter = ax.scatter(X.values[:,i], X.values[:,j], c=y_kmeans, s=10, cmap='viridis')
#
#         centers = kmeans.cluster_centers_
#         ax.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5)
#         ax.set_title('K-Means Clustering')
#         ax.set_xlabel(X.keys()[i])
#         ax.set_ylabel(X.keys()[j])
#         plt.show()