
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import pandas as pd

X, y = load_iris(return_X_y=True)

gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)
print("Number of mislabeled points out of a total %d points : %d"
      % (X.shape[0], (y!= y_pred).sum()))