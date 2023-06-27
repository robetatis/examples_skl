import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

# make synthetic data set
X, y = make_classification(
    n_samples=1000, 
    n_features=2, 
    n_redundant=0, 
    n_clusters_per_class=2, 
    weights=[0.99, 0.01], 
    flip_y=0.01, 
    random_state=0)

# check class distribution
counter = Counter(y)
print(counter)

# scatterplot
for k, _ in counter.items():
    rows = np.where(y == k)
    X1, X2 = X[rows, 0], X[rows, 1]
    plt.scatter(x=X1, y=X2, label=str(k))
plt.legend()
plt.ylabel('X2')
plt.xlabel('X1')
plt.savefig('scatter.png')

# don't use accuracy! always look at precision (TP/(TP + FP)) and recall (TP/(TP + FN)) for minority class. 
# # also F1-score (2*precision*recall/(precision + recall)) (different weights can be given to precision and recall)
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, stratify=y)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Vanilla approach -------------------------------------------------------')
print(f'accuracy = {accuracy_score(y_test, y_pred):.2f} ********* misleading!!')
print(f'precision = {precision_score(y_test, y_pred):.2f} ********* misleading!!')
print(f'recall = {recall_score(y_test, y_pred):.2f}')
print(f'f1_score = {f1_score(y_test, y_pred):.2f}')


# random undersampling majority class
unders = RandomUnderSampler(sampling_strategy=0.5)
X_unders, y_unders = unders.fit_resample(X, y)
counter_unders = Counter(y_unders)
print(counter_unders)

X_train, X_test, y_train, y_test =  train_test_split(X_unders, y_unders, test_size=0.3, stratify=y_unders)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Undersampling majority class -------------------------------------------------------')
print(f'accuracy = {accuracy_score(y_test, y_pred):.2f} ********* misleading!!')
print(f'precision = {precision_score(y_test, y_pred):.2f} ********* misleading!!')
print(f'recall = {recall_score(y_test, y_pred):.2f}')
print(f'f1_score = {f1_score(y_test, y_pred):.2f}')


# oversampling minority class -> SMOTE (synthetic minority oversampling technique)





