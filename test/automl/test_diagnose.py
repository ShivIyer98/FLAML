import unittest
import numpy as np
import scipy.sparse
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
from flaml import AutoML
from flaml.model import LGBMEstimator
from flaml import tune


class TestDiagnose(unittest.TestCase):

    def test_visualization(self):
        
        automl_settings = {
            "time_budget": 30,  # total running time in seconds
            "metric": 'accuracy',  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
                                # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
            "task": 'classification',  # task type
            "log_file_name": 'airlines_experiment.log',  # flaml log file
            "seed": 7654321,    # random seed
            }       
        automl_experiment = AutoML(**automl_settings)

        from flaml.data import load_openml_dataset
        X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir='./')
        automl_experiment.fit(X_train=X_train, y_train=y_train, **automl_settings)
        automl_experiment.diagnose(level=0)
        print(automl_experiment.predict(X_train))
        print(automl_experiment.model)
        print(automl_experiment.config_history)
        print(automl_experiment.best_model_for_estimator("lrl2"))
        print(automl_experiment.best_iteration)
        print(automl_experiment.best_estimator)


if __name__ == "__main__":
    unittest.main()
