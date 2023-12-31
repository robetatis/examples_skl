import pandas as pd
import numpy as np
from constants import path_to_data, features_to_encode, numeric_features, X_names, y_name, primary_key
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.combine import SMOTEENN

class Data:
    
    def __init__(self, path, primary_key, numeric_features, features_to_encode, y_name, X_names):
        self.path = path
        self.primary_key = primary_key
        self.numeric_features = numeric_features
        self.features_to_encode = features_to_encode
        self.y_name = y_name
        self.X_names = X_names
        
    def read(self):
        self.df_raw = pd.read_csv(self.path, na_values=' ', index_col=self.primary_key)

    def drop_missing(self):
        self.df = self.df_raw.dropna(axis=0)

    def set_type_numeric_features(self):
        for col in self.numeric_features:
            self.df.loc[:, col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def encode_categorical_features(self):
        onehot_encoder = OneHotEncoder()
        encoded_array = onehot_encoder.fit_transform(self.df[self.features_to_encode]).toarray()
        self.df_encoded = pd.DataFrame(encoded_array, columns=onehot_encoder.get_feature_names_out())
        self.df_encoded.index = self.df.index

    def make_y(self):
        self.y = pd.to_numeric(
            self.df[self.y_name].map({
            'No': 0.0,
            'Yes': 1.0
        }))
        
    def make_X(self):
        self.X = self.df_encoded.merge(self.df[self.numeric_features], left_index=True, right_index=True, how='inner') 
        self.X.columns = self.X_names
        
    def check_balance_y(self):
        print(self.df[self.y_name].value_counts())

    # combined undersampling majority and oversampling minority class
    def smoteen(self, sampling_strategy=0.5):
        self.sampling_strategy = sampling_strategy
        over_under = SMOTEENN(sampling_strategy=sampling_strategy)
        self.X_overs_unders, self.y_overs_unders = over_under.fit_resample(self.X, self.y)

    def train_test_split(self, X, y, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size, 
            stratify=y,
            random_state=0)


class Model:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def test(self, X_test, y_test):
        self.y_pred = (self.model.predict(X_test) > 0.5).astype(np.int32)
        report = metrics.classification_report(y_test, self.y_pred)
        print(self.name, '\n------------------------------')
        print(report)


class Pipeline:
        
    def run(self):
        
        data = Data(path_to_data, primary_key, numeric_features, features_to_encode, y_name, X_names)
        data.read()
        data.drop_missing()
        data.set_type_numeric_features()
        data.encode_categorical_features()
        data.check_balance_y()
        data.make_X()
        data.make_y()
        data.train_test_split(data.X, data.y, test_size=0.3)
        
        model_logsitic = Model(LogisticRegression(solver='liblinear'), name='logistic')
        model_logsitic.train(data.X_train, data.y_train)
        model_logsitic.test(data.X_test, data.y_test)

        model_rf = Model(RandomForestClassifier(), name='random forest')
        model_rf.train(data.X_train, data.y_train)
        model_rf.test(data.X_test, data.y_test)

        data.smoteen()
        data.train_test_split(data.X_overs_unders, data.y_overs_unders)
        model_logistic_smoteen = Model(LogisticRegression(solver='liblinear'), name='logistic smoteen')        
        model_logistic_smoteen.train(data.X_train, data.y_train)
        model_logistic_smoteen.test(data.X_test, data.y_test)
        data.check_balance_y()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()



