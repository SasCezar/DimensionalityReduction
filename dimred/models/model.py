from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def fit_trasform(self, X):
        X_tilde = self.fit(X)
        X_trans = self.transform(X_tilde)

        return X_trans