from abc import ABC, abstractmethod

import numpy as np

from models.model import AbstractModel


class PCA(AbstractModel, ABC):
    def __init__(self, n_components):
        super().__init__()
        self._n_components = n_components
        self._components = None
        self._explained_variance = None
        self._mean = None

    @property
    def mean(self):
        return self._mean

    @property
    def explained_variance(self):
        return self._explained_variance

    @property
    def components(self):
        return self._components

    def transform(self, X):
        if self._mean is not None:
            X -= self._mean
        X_transformed = np.dot(X, self._components.T)
        return X_transformed


class PCAEign(PCA):
    """
    PCA's implementation using Eigendecomposition

    Steps:
        1 - Compute the covariance C = (X^T X)/(n-1)
        2 - Diagonalize using the spectral decomposition C = QAQ^T
        where:
            - Q is the matrix of the eigenvectors (each column is an eigenvector) - Principal axes
            - A is the diagonal matrix with the eigenvalues lambda_i (sorted in decreasing order)

        The principal components are found by XV, where the j-th one is the j-th column of the resulting matrix.
        The coordinates of the i-th data point in the new PC space are given by the i-th row of XV.

    Used links:
        - https://stats.stackexchange.com/a/134283
    """

    def fit(self, X, y=None, *args, **kwargs):
        """
        Finds the principal axes, and their explained variance
        :param X: Numpy array
        :return:
        """
        n, _ = X.shape

        self._mean = np.mean(X, axis=0)
        X -= self._mean

        covariance = 1 / (n - 1) * np.matmul(X.T, X)
        Q, A = np.linalg.eig(covariance)
        components = A[:, :self._n_components]
        explained_variance = Q[:self._n_components]
        self._components = components.transpose()
        self._explained_variance = explained_variance.T

        return self


class PCASVD(PCA):
    """
    PCA's implementation using Singular Values Decomposition

    Steps:
        1 - Decompose X using SVD X = UEV^T
        where:
            - U is a unitary matrix
            - E is the diagonal matrix of the singular values \sigma_i

        2 - The covariance matrix C = (VSU^T USV^T) / (n - 1) = V (S^2)/(n-1) V^T

        Which means that the singular values are principal directions and the singular values, are
        related to the covariance matrix via lambda_i = s^2_i/(n-1)

        The principal components are found by XU = VEU^TU = VE

    Used links:
        - https://stats.stackexchange.com/a/134283
    """

    def fit(self, X, y=None, *args, **kwargs):
        """
        Finds the principal axes and their explained variance
        :param X: Numpy array
        :return:
        """
        n, _ = X.shape
        self._mean = np.mean(X, axis=0)
        X -= self._mean
        u, s, v = np.linalg.svd(X)
        self._components = v[:self._n_components]
        explained_variance = np.square(s[:self._n_components]) / (n - 1)
        self._explained_variance = explained_variance

        return self
