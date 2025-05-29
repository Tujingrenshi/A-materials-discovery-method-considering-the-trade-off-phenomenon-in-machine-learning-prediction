import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import copy


class MSE_by_model():
    """Class for partitioning based on MSE between the predicted values of
    the regression model applied to the sample group and measured values.
    """
    def __init__(self, model):
        self.model = copy.copy(model)

    def __call__(self, X, X_l, X_r, y, y_l, y_r, supplement_model_train_manner='origin'):   # gai
        a = self._calc(X, y, supplement_model_train_manner)
        b = self._calc(X_l, y_l, supplement_model_train_manner)
        c = self._calc(X_r, y_r, supplement_model_train_manner)
        return self._calc(X, y, supplement_model_train_manner) - self._calc(X_l, y_l, supplement_model_train_manner) - self._calc(X_r, y_r, supplement_model_train_manner)   # gai

    def _calc(self, X, y, supplement_model_train_manner):    #  gai
        if supplement_model_train_manner == "origin":   # gai
            self.model.fit(X, y)
            error = mean_squared_error(y, self.model.predict(X)) * len(X)    #  gai
        return error



class MSE():
    """Class for partitioning based on MSE between the mean and measured
    values of the sample group.
    """
    def __init__(self):
        pass

    def __call__(self, X, X_l, X_r, y, y_l, y_r, supplement_model_train_manner):
        return self._calc(X, y) - self._calc(X_l, y_l) - self._calc(X_r, y_r)

    def _calc(self, X, y):
        return np.sum(np.power(y - np.mean(y), 2))



# 自己根据AI完成的

class SquaredError():
    """Class for partitioning based on Squared Error (MSE) between the mean and measured
    values of the sample group.
    """
    def __init__(self):
        pass

    def __call__(self, X, X_l, X_r, y, y_l, y_r, supplement_model_train_manner):
        return self._calc(X, y) - self._calc(X_l, y_l) - self._calc(X_r, y_r)

    def _calc(self, X, y):
        return np.sum(np.power(y - np.mean(y), 2))


class AbsoluteError():
    """Class for partitioning based on Absolute Error between the mean and measured
    values of the sample group.
    """
    def __init__(self):
        pass

    def __call__(self, X, X_l, X_r, y, y_l, y_r, supplement_model_train_manner):
        return self._calc(X, y) - self._calc(X_l, y_l) - self._calc(X_r, y_r)

    def _calc(self, X, y):
        return np.sum(np.abs(y - np.mean(y)))


class FriedmanMSE():
    """Class for partitioning based on Friedman's Mean Squared Error.
    This is a variation of MSE that is used in Gradient Boosting with decision trees.
    """
    def __init__(self):
        pass

    def __call__(self, X, X_l, X_r, y, y_l, y_r, supplement_model_train_manner):
        return self._calc(y, y_l, y_r)

    def _calc(self, y, y_l, y_r):
        n_total = len(y)
        n_l = len(y_l)
        n_r = len(y_r)
        y_mean = np.mean(y)
        y_l_mean = np.mean(y_l)
        y_r_mean = np.mean(y_r)

        score = (
            (n_l * np.power(y_l - y_l_mean, 2).sum() +
             n_r * np.power(y_r - y_r_mean, 2).sum()) / n_total
        )

        return score


class Poisson():
    """Class for partitioning based on Poisson Loss.
    This is used when the target variable represents counts or rates.
    """
    def __init__(self):
        pass

    def __call__(self, X, X_l, X_r, y, y_l, y_r, supplement_model_train_manner):
        return self._calc(X, y) - self._calc(X_l, y_l) - self._calc(X_r, y_r)

    def _calc(self, X, y):
        y_mean = np.mean(y)
        return np.sum(y_mean - y * np.log(y_mean + 1e-10))

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import copy
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 1 + np.random.randn(100, 1) * 2  # 创建一个带噪声的线性关系

    # 分割数据
    X_l = X[:50]
    X_r = X[50:]
    y_l = y[:50]
    y_r = y[50:]

    # 实例化模型
    model = LinearRegression()

    # 使用原始训练方式
    mse_by_model_origin = MSE_by_model()
    mse_origin = mse_by_model_origin(X, X_l, X_r, y, y_l, y_r, supplement_model_train_manner='origin')
    print(f"MSE with original training: {mse_origin}")

    # # 使用微调训练方式
    # mse_by_model_finetune = MSE_by_model(model, supplement_model_train_manner='finetune')
    # mse_finetune = mse_by_model_finetune(X, X_l, X_r, y, y_l, y_r)
    # print(f"MSE with finetuned training: {mse_finetune}")