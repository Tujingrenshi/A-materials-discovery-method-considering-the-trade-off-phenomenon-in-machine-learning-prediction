import numpy as np
from sklearn.utils import check_array, check_X_y
from joblib import Parallel, delayed
from ..criterion import MSE_by_model
from .regression_tree import RegressionTree
import copy
import random


class Extrapolation_Random_Forest():

    def __init__(
        self,
        n_estimators_1=100,   
        n_estimators_2=0,   
        n_estimators_3=0,   
        criterion=None,
        # regressor=None,
        threshold_selector=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features_poly = 3,  
        max_features='auto',
        f_select=True,
        ensemble_pred='mean',
        scaling=False,
        bootstrap=True,
        random_state=None,
        split_continue=False,
        supplement_model_train_manner="origin",           #  
        verbose=0
    ):
        self.n_estimators_1 = n_estimators_1  ###
        self.n_estimators_2 = n_estimators_2  ###
        self.n_estimators_3 = n_estimators_3  ###
        self.criterion = criterion
        # self.regressor = regressor
        self.threshold_selector = threshold_selector
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_poly = max_features_poly
        self.max_features = max_features
        self.f_select = f_select
        self.ensemble_pred = ensemble_pred
        self.scaling = scaling
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.split_continue = split_continue
        self.supplement_model_train_manner = supplement_model_train_manner           #  
        self.verbose = verbose

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y)."""
        X, y = check_X_y(X, y, ['csr', 'csc'])

        n_estimators = self.n_estimators_1 + self.n_estimators_2 + self.n_estimators_3  ###

        random_state = self.check_random_state(self.random_state)
        seeds = random_state.randint(
            np.iinfo(np.int32).max, size=n_estimators)


        
        zeros = [0] * self.n_estimators_1  ###
        ones = [1] * self.n_estimators_2  ###
        twos = [1] * self.n_estimators_3  ###
        
        degree_list = zeros + ones + twos  ###
        random.seed(self.random_state)  ###
        random.shuffle(degree_list)  ###

        
        from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.feature_selection import SelectKBest, f_regression
        # 定义一个函数，用于随机选择特征
        def random_feature_selection(X, n_features_to_select):
            n_features = X.shape[1]
            selected_features = np.random.choice(n_features, n_features_to_select, replace=False)
            return X[:, selected_features]



        regressor_list_1 = [LinearRegression()] * self.n_estimators_1  ###
        regressor_list_2 = []
        for random_state in range(self.n_estimators_2):  ###
            random.seed(random_state)  ###
            import math
            num = math.comb(len(X[0]), 2)
            feature_num = random.randint(1, num)  ###
            # 使用 FunctionTransformer 将函数包装为转换器
            random_feature_transformer = FunctionTransformer(
                random_feature_selection, 
                kw_args={'n_features_to_select': feature_num}
            )
            pipeline_poly_2 = make_pipeline(
                PolynomialFeatures(degree=2),
                SelectKBest(f_regression, k=feature_num),  # 只保留feature_num个最佳特征
                # random_feature_transformer,
                LinearRegression()
            )
            regressor_list_2.append(pipeline_poly_2)
        regressor_list_3 = []
        for random_state in range(self.n_estimators_3):  ###
            random.seed(random_state)  ###
            import math
            num = math.comb(len(X[0]), 3)
            feature_num = random.randint(1, num)  ###
            # 使用 FunctionTransformer 将函数包装为转换器
            random_feature_transformer = FunctionTransformer(
                random_feature_selection, 
                kw_args={'n_features_to_select': feature_num}
            )
            pipeline_poly_3 = make_pipeline(
                PolynomialFeatures(degree=3),
                SelectKBest(f_regression, k=feature_num),  # 只保留随机个个最佳特征
                # random_feature_transformer,
                LinearRegression()
            )
            regressor_list_3.append(pipeline_poly_3)
        regressor_list = regressor_list_1 + regressor_list_2 + regressor_list_3  ###
        # regressor_list = [regressor_list[i] for i in degree_list]           ###

        # regressor_list = [self.regressor[i] for i in degree_list]           ###

        self.forest = Parallel(n_jobs=-1, verbose=self.verbose)(
            delayed(self._build_trees)(X, y, seeds[i], regressor_list[i])  ###
            for i in range(n_estimators))

    def predict(self, X, return_std=False, shuchushuju=False, shuchuallpred = False):  
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean or median predicted regression targets of the trees in the forest.
        """
        def find_closest_index(lst, target):
            closest_index = 0
            min_diff = abs(lst[0] - target)
            
            for i, value in enumerate(lst):
                current_diff = abs(value - target)
                if current_diff < min_diff:
                    min_diff = current_diff
                    closest_index = i
            
            return closest_index

        if shuchushuju:   
            X = check_array(X, accept_sparse='csr')
            pred, yejiedian_shuju = [], []  
            for tree in self.forest:  
                pred_results, yejiedian_shuju_results = tree.predict(X, shuchushuju=shuchushuju)  
                pred.append(pred_results.tolist())  
                yejiedian_shuju.append(yejiedian_shuju_results)  
            if return_std:
                if self.ensemble_pred == 'mean':
                    mean_index = find_closest_index(pred, np.mean(pred, axis=0))
                    if shuchuallpred:  
                        return np.mean(pred, axis=0), np.std(pred, axis=0), yejiedian_shuju[mean_index], pred  
                    else:  
                        return np.mean(pred, axis=0), np.std(pred, axis=0), yejiedian_shuju[mean_index]  
                elif self.ensemble_pred == 'median':
                    median_indices = np.argsort(pred, axis=0)[int(len(pred) / 2)]  
                    yejiedianshuju_shuchu = yejiedian_shuju[median_indices[0]]  
                    if shuchuallpred:  
                        return np.median(pred, axis=0), np.std(pred, axis=0), yejiedianshuju_shuchu, pred  
                    else:  
                        return np.median(pred, axis=0), np.std(pred, axis=0), yejiedianshuju_shuchu  
            else:
                if self.ensemble_pred == 'mean':
                    mean_index = find_closest_index(pred, np.mean(pred, axis=0))
                    if shuchuallpred:
                        return np.mean(pred, axis=0), yejiedian_shuju[mean_index], pred
                    else:
                        return np.mean(pred, axis=0), yejiedian_shuju[mean_index]  
                elif self.ensemble_pred == 'median':
                    median_indices = np.argsort(pred, axis=0)[int(len(pred) / 2)]  
                    yejiedianshuju_shuchu = yejiedian_shuju[median_indices[0]]  
                    if shuchuallpred:
                        return np.median(pred, axis=0), yejiedianshuju_shuchu, pred
                    else:
                        return np.median(pred, axis=0), yejiedianshuju_shuchu  
                else:
                    return pred
        if 1 - shuchushuju:
            X = check_array(X, accept_sparse='csr')
            pred = np.array([tree.predict(X).tolist() for tree in self.forest])
            if return_std:
                if self.ensemble_pred == 'mean':
                    if shuchuallpred:
                        return np.mean(pred, axis=0), np.std(pred, axis=0), pred
                    else:
                        return np.mean(pred, axis=0), np.std(pred, axis=0)
                elif self.ensemble_pred == 'median':
                    if shuchuallpred:
                        return np.median(pred, axis=0), np.std(pred, axis=0), pred
                    else:
                        return np.median(pred, axis=0), np.std(pred, axis=0)
            else:
                if self.ensemble_pred == 'mean':
                    if shuchuallpred:  
                        return np.mean(pred, axis=0), pred
                    else:
                        return np.mean(pred, axis=0)
                elif self.ensemble_pred == 'median':
                    if shuchuallpred:  
                        return np.median(pred, axis=0), pred
                    else:
                        return np.median(pred, axis=0)
                else:
                    return pred

    def _build_trees(self, X, y, seed, regressor_):  ###
        if self.criterion == "MSE_by_model":           #  
            criterion_ = MSE_by_model(regressor_)           #  
        else:           #  
            criterion_ = self.criterion           #  
        tree = RegressionTree(
            criterion=criterion_,           #  
            regressor=regressor_,           #  
            threshold_selector=self.threshold_selector,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            f_select=self.f_select,
            scaling=self.scaling,
            random_state=seed,
            split_continue=self.split_continue,
            supplement_model_train_manner = self.supplement_model_train_manner  
        )
        if self.bootstrap:
            X_bootstrap, y_bootstrap = self._bootstrap(seed, X, y)
            tree.fit(X_bootstrap, y_bootstrap)
        else:
            tree.fit(X, y)
        return tree

    def count_selected_feature(self):
        """Count the number of features used to divide the tree."""
        return np.array(
            [tree.count_feature() for tree in self.forest])

    def _bootstrap(self, seed, X, y):
        n_samples, n_features = X.shape
        random_state = np.random.RandomState(seed)
        boot_index = random_state.randint(0, n_samples, n_samples)

        if isinstance(self.max_features, int):
            if not 1 <= self.max_features:
                print('The number of features must be one or more.')
            boot_features = self.max_features

        elif isinstance(self.max_features, float):
            if not 1. >= self.max_features:
                print('The fraction of features is must be less than 1.0.')
            elif not 0 < self.max_features:
                print('The fraction of features is must be more than 0.')
            boot_features = int(n_features * self.max_features)

        else:
            if self.max_features == 'auto' or 'None':
                boot_features = n_features
            elif self.max_features == 'sqrt':
                boot_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                boot_features = int(np.log2(n_features))

        boot_feature_index = random_state.permutation(
            n_features)[0:boot_features]
        remove_feature_index = list(set(range(
            n_features)) - set(boot_feature_index.tolist()))
        boot_X = X[boot_index, :].copy()
        boot_X[:, remove_feature_index] = 0.0
        return boot_X, y[boot_index]

    def check_random_state(self, seed):
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, int):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'regressor': self.regressor,
            'threshold_selector': self.threshold_selector,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'f_select': self.f_select,
            'ensemble_pred': self.ensemble_pred,
            'scaling': self.scaling,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'split_continue': self.split_continue,
            'supplement_model_train_manner':self.supplement_model_train_manner
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
