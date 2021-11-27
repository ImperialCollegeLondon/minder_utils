from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from minder_utils.formatting.format_util import y_to_categorical


class Classifiers:
    def __init__(self, model_type='nn'):
        self.model_type = model_type
        self.model = getattr(self, model_type)()
        self.callbacks = [EarlyStopping(monitor='loss', patience=3)]
        self.epochs = 50
        self.batch_size = 10

    def reset(self):
        self.model = getattr(self, self.model_type)()

    @property
    def methods(self):
        return {
            'nn': 'neural network',
            # 'lstm': 'LSTM',
            'lr': 'logistic regression',
            'bayes': 'naive bayesian',
            'dt': 'decision tree',
            'knn': 'KNN'
        }

    def get_info(self, verbose=False):
        if verbose:
            print('Available methods:')
            for idx, key in enumerate(self.methods):
                print(str(idx).ljust(10, ' '), key.ljust(10, ' '), self.methods[key].ljust(10, ' '))
        return self.methods

    @staticmethod
    def nn():
        model = Sequential()
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model

    @staticmethod
    def lstm():
        model = Sequential()
        model.add(LSTM(256, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model

    def fit(self, data, targets, cat=False):
        data = data.reshape(data.shape[0], -1)
        if self.model_type in ['pnn', 'nn', 'lstm']:
            if self.model_type in ['lstm', 'pnn']:
                data = data.reshape(data.shape[0], 24, -1)
            self.model.fit(data, targets, epochs=self.epochs, batch_size=self.batch_size, callbacks=self.callbacks,
                           verbose=0)
        else:
            targets = y_to_categorical(targets) if cat else targets
            targets = np.argmax(targets, axis=1) if targets.ndim > 1 else targets
            self.model.fit(data, targets)

    def predict(self, data):
        data = data.reshape(data.shape[0], -1)
        if self.model_type in ['pnn', 'nn', 'lstm']:
            if self.model_type in ['lstm', 'pnn']:
                data = data.reshape(data.shape[0], 24, -1)
            return np.argmax(self.model.predict(data), axis=1)
        else:
            return self.model.predict(data)

    def predict_probs(self, data):
        data = data.reshape(data.shape[0], -1)
        if self.model_type in ['pnn', 'nn', 'lstm']:
            if self.model_type in ['lstm', 'pnn']:
                data = data.reshape(data.shape[0], 24, -1)
            return self.model.predict(data)
        else:
            return self.model.predict_proba(data)

    @staticmethod
    def lr():
        return LogisticRegression(max_iter=1000)

    @staticmethod
    def bayes():
        return GaussianNB()

    @staticmethod
    def dt():
        return tree.DecisionTreeClassifier()

    @staticmethod
    def knn():
        return KNeighborsClassifier()

    def __name__(self):
        return self.model_type
