import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
dataset = fetch_mldata("MNIST original")

X = dataset.data
y = dataset.target

sum_digit = X[19999]
sum_digit_image = sum_digit.reshape(28, 28)

plt.imshow(sum_digit_image)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test ,y_train, y_test = train_test_split(X,y)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 8)
dtf.fit(X_train , y_train)

dtf.score(X_train, y_train)
dtf.score(X_test , y_test)

dtf.predict(X[[555, 1500, 10011, 19999], 0:784])


from sklearn.tree import export_graphviz

export_graphviz(dtf, out_file = "tree.dot")


import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
    graphviz.Source(dot_graph)
