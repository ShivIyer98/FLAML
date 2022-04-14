import unittest

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from flaml.automl import AutoML
from flaml.model import XGBoostSklearnEstimator
from flaml import tune


dataset = "credit-g"




def test_simple(method=None):
    automl = AutoML()

    automl_settings = {
        "estimator_list": ["lgbm"],
        "task": "regression",
        "n_jobs": 1,
        "hpo_method": method,
        "log_type": "all",
        "retrain_full": "budget",
        "keep_search_state": True,
        "time_budget": 1,
    }
    from sklearn.externals._arff import ArffException
    
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    print(automl.estimator_list)
    print(automl.search_space)
    print(automl.points_to_evaluate)
    #automl.diagnose(problem_type="regression", level=1)
    automl.diagnose(problem_type="regression", explainer = "SHAP", plot_type = "force", level=1)
    config = automl.best_config.copy()
    config["learner"] = automl.best_estimator
    automl.trainable(config)
    from flaml import tune
    from flaml.automl import size
    from functools import partial





if __name__ == "__main__":
    #unittest.main()
    test_simple()


