from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def support_vector_machines_main():
    data = load_breast_cancer()

    X = data.data
    Y = data.target

    # , random_state=23 => ensures same data and result every time
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    clf = SVC(kernel='linear', C=3)
    clf.fit(x_train, y_train)

    clf2 = KNeighborsClassifier(n_neighbors=3)
    clf2.fit(x_train, y_train)

    print(f'SVC: {clf.score(x_test, y_test)}')
    print(f'KNN: {clf2.score(x_test, y_test)}')
