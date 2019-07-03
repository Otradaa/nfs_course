import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

params = [{'solver': 'adam', 'activation': 'relu', 'alpha': 0.0001,
           'hidden_layer_sizes': (100, 100, 100, 100)},
          {'solver': 'adam', 'activation': 'tanh', 'alpha': 0.0001,
           'hidden_layer_sizes': (100, 100, 100, 100)}]
# {'solver': 'adam', 'activation': 'relu', 'alpha': 0.0001,'learning_rate_init':0.0001,
#            'hidden_layer_sizes': (100,100,100,100,100)}]
# {'solver': 'adam', 'activation': 'logistic', 'alpha': 0.0001,
#            'hidden_layer_sizes': (100, 100, 100)}]
# {'solver': 'adam', 'activation': 'relu', 'alpha': 0.0001,
#            'hidden_layer_sizes': (100, 100, 100)}]
# {'solver': 'sgd', 'activation': 'relu', 'alpha': 0.0001,
#  'hidden_layer_sizes': (100, 100, 100)},
# {'solver': 'lbfgs', 'activation': 'relu', 'alpha': 0.0001,
#  'hidden_layer_sizes': (100, 100, 100)}]

for param in params:
    mlp = MLPClassifier(verbose=0, random_state=1, max_iter=400, **param)
    mlp.fit(x_train, y_train)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mlp.loss_curve_, c='green')
    plt.show()

    y_pred = mlp.predict(x_test)
    precision = precision_score(y_test, y_pred)  # not p n
    recall = recall_score(y_test, y_pred)  # all pos
    f1 = f1_score(y_test, y_pred)

    print("Training set loss: %f" % mlp.loss_)
    print("Training set score: %f" % mlp.score(x_train, y_train))
    print("Training set score: %f" % mlp.score(x_test, y_test))
    print(f"NN: precision={precision:.10f}, recall={recall:.10f}, f1={f1:.10f}")

