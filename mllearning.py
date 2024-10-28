import math
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 415532
    n = len(x)
    learning_rate = 0.0002
    cost = 0
    for i in range (iterations):
        y_predicted = m_curr * x + b_curr
        new_cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum((y-y_predicted))
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        prev_cost = cost
        cost = new_cost
        if math.isclose(prev_cost, new_cost, rel_tol=1e-20):
            break
        print("m {} , b {} , iteration {}, cost {}".format(m_curr,b_curr,i, cost))

    return m_curr, b_curr


dataframe1 = pd.read_csv('test_scores.csv')
print(dataframe1)

math_col = dataframe1['math'].to_numpy()
cs_col = dataframe1['cs'].to_numpy()


x = math_col
y = cs_col

m, b = gradient_descent(x, y)
print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

m_sklearn, b_sklearn = predict_using_sklean()
print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))