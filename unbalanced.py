import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


class Data:

    # make synthetic data set
    def make_synthetic(self, n_samples, n_features, n_redundant, n_clusters_per_class, weights, flip_y, random_state):
        self.n_samples = n_samples 
        self.n_features = n_features 
        self.n_redundant = n_redundant 
        self.n_clusters_per_class = n_clusters_per_class 
        self.weights = weights 
        self.flip_y = flip_y 
        self.random_state = random_state

        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_redundant=self.n_redundant,
            n_clusters_per_class=self.n_clusters_per_class,
            weights=self.weights,
            flip_y=self.flip_y,
            random_state=self.random_state)        

    # scatterplot
    def save_bivariate_scatterplot(self, col_no_X1, col_no_X2, name):
        for k, _ in self.counter.items():
            rows = np.where(self.y == k)
            X1, X2 = self.X[rows, col_no_X1], self.X[rows, col_no_X2]
            plt.scatter(x=X1, y=X2, label=str(k))
        plt.legend()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.savefig(name)

    # compute class frequencies
    def count_y_frequencies(self, y):
        self.counter = Counter(y)
    
    # random undersampling of majority class
    def random_undersample_majority_class(self, sampling_strategy=0.5):
        self.sampling_strategy = sampling_strategy
        unders = RandomUnderSampler(sampling_strategy=sampling_strategy)
        self.X_unders, self.y_unders = unders.fit_resample(self.X, self.y)
        self.count_y_frequencies(self.y_unders)

    # oversampling minority class -> SMOTE (synthetic minority oversampling technique)
    def smote(self, sampling_strategy=0.5):
        overs = SMOTE(sampling_strategy=sampling_strategy)
        self.X_overs, self.y_overs = overs.fit_resample(self.X, self.y)
        self.count_y_frequencies(self.y_overs)

    # combined undersampling majority and oversampling minority class
    def smoteen(self, sampling_strategy=0.5):
        self.sampling_strategy = sampling_strategy
        over_under = SMOTEENN(sampling_strategy=sampling_strategy)
        self.X_overs_unders, self.y_overs_unders = over_under.fit_resample(self.X, self.y)
        self.count_y_frequencies(self.y_overs_unders)

    # make train and test sets
    def train_test_split(self, X, y, test_size=0.3):
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, stratify=y)


class Model:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test, data):
        y_pred = (self.model.predict(X_test)).astype(np.int32)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred)
        self.recall = recall_score(y_test, y_pred)
        self.f1_score = f1_score(y_test, y_pred)
        self.print_test_report(data)

    def print_test_report(self, data):
        
        # don't use accuracy! always look at precision (TP/(TP + FP)) and recall (TP/(TP + FN)) for minority class. 
        # # also F1-score (2*precision*recall/(precision + recall)) (different weights can be given to precision and recall)
        print(f'\n{self.name} -------------------------------------------------------')
        print('Class frequencies:', data.counter)
        print(f'accuracy = {self.accuracy:.2f}')
        print(f'precision = {self.precision:.2f}')
        print(f'recall = {self.recall:.2f}')
        print(f'f1_score = {self.f1_score:.2f}')


class Pipeline:
    def run(self):
        data = Data()
        data.make_synthetic(1000, 2, 0, 1, [0.98, 0.02], 0.01, 0)
        data.count_y_frequencies(data.y)
        data.save_bivariate_scatterplot(0, 1, 'scatter.png')
        data.train_test_split(data.X, data.y)

        model = Model(LogisticRegression(), 'vanilla')
        model.train(data.X_train, data.y_train)
        model.test(data.X_test, data.y_test, data)

        model = Model(LogisticRegression(), 'random undersampling majority class')
        data.random_undersample_majority_class()
        data.train_test_split(data.X_unders, data.y_unders)
        model.train(data.X_train, data.y_train)
        model.test(data.X_test, data.y_test, data)

        model = Model(LogisticRegression(), 'SMOTE')
        data.smote()
        data.train_test_split(data.X_overs, data.y_overs)
        model.train(data.X_train, data.y_train)
        model.test(data.X_test, data.y_test, data)

        model = Model(LogisticRegression(), 'SMOTEEN')
        data.smoteen()
        data.train_test_split(data.X_overs_unders, data.y_overs_unders)
        model.train(data.X_train, data.y_train)
        model.test(data.X_test, data.y_test, data)

        # cost-sensitive algorithms -> class_weight parameter in sklearn
        model = Model(LogisticRegression(), 'cost-sensitive -> class_weight = f(class frequency)')
        data.train_test_split(data.X, data.y)
        model.train(data.X_train, data.y_train)
        model.test(data.X_test, data.y_test, data)


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
