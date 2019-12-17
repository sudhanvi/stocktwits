from os.path import join
from os import environ

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score, recall_score

from tffm import TFFMRegressor

import numpy as np
import pandas as pd
import tensorflow as tf

import scipy
import sys


class FactorisationMachines:
    def __init__(self, onlyResults=False, metric='rmse'):
        self.dpath = '../data/csv/'
        self._libfm_path = '../data/csv/'
        self.X_train, self.y_train = load_svmlight_file(join(self.dpath, 'train.libfm'))
        self.X_test, self.y_test = load_svmlight_file(join(self.dpath, 'test.libfm'))

        self.onlyResults = onlyResults
        self.metric = metric

    def tffm(self):
        show_progress = True if not self.onlyResults else False
        X_train, y_train, X_test, y_test = self.X_train.todense(), np.transpose(
            self.y_train).flatten(), self.X_test.todense(), np.transpose(self.y_test).flatten()
        if self.onlyResults: environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        model = TFFMRegressor(
            order=2,
            rank=4,
            optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
            n_epochs=100,
            batch_size=-1,
            init_std=0.001,
            input_type='dense'
        )

        model.fit(X_train, y_train, show_progress=show_progress)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        if not self.onlyResults:
            print('RMSE: {:.6f}'.format(rmse))
        model.destroy()
        if self.onlyResults:
            print("Completed tffm evaluation.")
        return rmse

    def libfm(self):
        predictions = pd.read_csv(join(self.dpath, 'out.libfm'), header=None).values.flatten()
        testY = self.y_test

        prec = precision_score(testY, predictions.round(), average='weighted')
        rec = recall_score(testY, predictions.round(), average='weighted')
        fmeasure = 2 * ((prec * rec) / (prec + rec))
        auc = roc_auc_score(testY, predictions, average='weighted')
        rmse = np.sqrt(mean_squared_error(testY, predictions))
        print("LibFM RMSE: {}".format(rmse))
        return auc, rmse


    def run(self):

        print(self.tffm())
        print(self.libfm())

if __name__ == "__main__":
    fm = FactorisationMachines(onlyResults=False)
    fm.run()
