import importlib

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB


def _optional_import(module_name, attr_names):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return tuple(None for _ in attr_names)
    return tuple(getattr(module, attr_name) for attr_name in attr_names)


XGBClassifier, XGBRegressor = _optional_import('xgboost', ('XGBClassifier', 'XGBRegressor'))
LGBMClassifier, LGBMRegressor = _optional_import('lightgbm', ('LGBMClassifier', 'LGBMRegressor'))
CatBoostClassifier, CatBoostRegressor = _optional_import('catboost', ('CatBoostClassifier', 'CatBoostRegressor'))


class ModelFactory:
    @staticmethod
    def get_models(task_type):
        if task_type == 'classification':
            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'KNN': KNeighborsClassifier(n_jobs=-1),
                'NaiveBayes': GaussianNB()
            }
            if XGBClassifier is not None:
                models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
            if LGBMClassifier is not None:
                models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
            if CatBoostClassifier is not None:
                models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0)
            return models
        else:
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'DecisionTree': DecisionTreeRegressor(random_state=42),
                'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'KNN': KNeighborsRegressor(n_jobs=-1),
                'SVR': SVR()
            }
            if XGBRegressor is not None:
                models['XGBoost'] = XGBRegressor(random_state=42, n_jobs=-1)
            if LGBMRegressor is not None:
                models['LightGBM'] = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
            if CatBoostRegressor is not None:
                models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=0)
            return models

    @staticmethod
    def get_param_space(model_name):
        if 'RandomForest' in model_name:
            return {
                'n_estimators': (100, 500),
                'max_depth': (3, 15),
                'min_samples_split': (2, 20)
            }
        elif 'XGBoost' in model_name or 'XGB' in model_name:
            return {
                'n_estimators': (100, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0)
            }
        elif 'LightGBM' in model_name or 'LGBM' in model_name:
            return {
                'n_estimators': (100, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'num_leaves': (20, 100)
            }
        elif 'CatBoost' in model_name:
            return {
                'iterations': (100, 500),
                'depth': (3, 10),
                'learning_rate': (0.01, 0.3)
            }
        elif 'GradientBoosting' in model_name:
            return {
                'n_estimators': (100, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3)
            }
        return {}
