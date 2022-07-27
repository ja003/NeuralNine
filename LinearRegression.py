import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_csv('data.csv')


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return [m, b]


def linear_regression_main():
    m = 0
    b = 0
    L = 0.0001
    epochs = 1000

    for i in range(epochs):
        m, b = gradient_descent(m, b, points, L)

    print(m, b)