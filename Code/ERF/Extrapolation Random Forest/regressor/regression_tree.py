import numpy as np
from sklearn.utils import check_array, check_X_y
from .node import Node


class RegressionTree():
    """Class for regression tree."""
    def __init__(
        self,
        criterion=None,
        regressor=None,
        threshold_selector=None,
        max_depth=(2**31)-1,
        min_samples_split=2,
        min_samples_leaf=1,
        f_select=True,
        scaling=False,
        random_state=None,
        split_continue=False,
        supplement_model_train_manner = 'origin'
    ):
        self.tree = None
        self.criterion = criterion
        self.regressor = regressor
        self.threshold_selector = threshold_selector
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.f_select = f_select
        self.scaling = scaling
        self.random_state = random_state
        self.split_continue = split_continue
        self.supplement_model_train_manner = supplement_model_train_manner

    def fit(self, X, y):
        X, y = check_X_y(X, y, ['csr', 'csc'])

        self.tree = Node(
            criterion=self.criterion,
            regressor=self.regressor,
            threshold_selector=self.threshold_selector,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            f_select=self.f_select,
            scaling=self.scaling,
            random_state=self.random_state,
            split_continue=self.split_continue,
            supplement_model_train_manner = self.supplement_model_train_manner
        )
        self.tree.fit(X, y, 0)

    def predict(self, X, shuchushuju=False):

        X = check_array(X, accept_sparse='csr')
        if shuchushuju:
            results, training_data = self.tree.predict(X, shuchushuju=shuchushuju)
            return results, training_data
        if 1 - shuchushuju:
            return self.tree.predict(X, shuchushuju=shuchushuju)

    def count_feature(self):
        return self.tree._count_feature()
