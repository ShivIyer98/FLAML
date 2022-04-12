import unittest

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from flaml.automl import AutoML
from flaml.model import XGBoostSklearnEstimator
from flaml import tune


dataset = "credit-g"


class XGBoost2D(XGBoostSklearnEstimator):
    @classmethod
    def search_space(cls, data_size, task):
        upper = min(32768, int(data_size[0]))
        return {
            "n_estimators": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "low_cost_init_value": 4,
            },
            "max_leaves": {
                "domain": tune.lograndint(lower=4, upper=upper),
                "low_cost_init_value": 4,
            },
        }


def test_simple(method=None):
    automl = AutoML()
    automl.add_learner(learner_name="XGBoost2D", learner_class=XGBoost2D)

    automl_settings = {
        "estimator_list": ["XGBoost2D"],
        "task": "classification",
        "log_file_name": f"test/xgboost2d_{dataset}_{method}.log",
        "n_jobs": 1,
        "hpo_method": method,
        "log_type": "all",
        "retrain_full": "budget",
        "keep_search_state": True,
        "time_budget": 1,
    }
    from sklearn.externals._arff import ArffException
    '''
    try:
        X, y = fetch_openml(name=dataset, return_X_y=True)
    except (ArffException, ValueError):
        from sklearn.datasets import load_wine
    '''
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    print(automl.estimator_list)
    print(automl.search_space)
    print(automl.points_to_evaluate)
    automl.diagnose(class_names=['class_0', 'class_1', 'class_2'], labels = y_train, problem_type="classification", level=1)
    config = automl.best_config.copy()
    config["learner"] = automl.best_estimator
    automl.trainable(config)
    from flaml import tune
    from flaml.automl import size
    from functools import partial





if __name__ == "__main__":
    #unittest.main()
    test_simple()