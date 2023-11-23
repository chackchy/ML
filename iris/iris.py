from sklearn.datasets import load_iris
from icecream import ic

iris = load_iris()
X, y = iris.data, iris.target
X = X.T
y = y.reshape(-1, 1).T

ic(X.shape, y.shape)