import unittest
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

from models.PCA import PCAEign, PCASVD


class TestPCAEigen(unittest.TestCase):
    def test_fit_equal_sklearn(self):
        iris = datasets.load_iris()
        p = iris.data.shape[1]
        k = 2
        X_reduced = PCA(n_components=k).fit(iris.data)
        X_reduced_eig = PCAEign(n_components=k).fit(iris.data)
        comp_diff = np.round(np.absolute(X_reduced.components_) - np.absolute(X_reduced_eig.components), 3)
        self.assertTrue(np.array_equal(comp_diff, np.zeros([k,p])))
        var_diff = np.round(X_reduced.explained_variance_ - X_reduced_eig.explained_variance, 3)
        self.assertTrue(np.array_equal(var_diff, np.zeros(k)))
        return

    def test_transform_equal_sklearn(self):
        """
        This test might fail as our PCA finds eigenvectors with sign opposite to the one of sklearn. Sklearn, solves
        this undeterministic behaviour by using sklearn.utils.extmath.svd_flip
        """
        iris = datasets.load_iris()

        k = 2
        PCA_sk = PCA(n_components=k, whiten=False).fit(iris.data)
        PCA_eign = PCAEign(n_components=k).fit(iris.data)
        X_reduced = PCA_sk.transform(iris.data)

        X_reduced_eig = PCA_eign.transform(iris.data)
        comp_diff = np.round(np.absolute(X_reduced) - np.absolute(X_reduced_eig), 3)
        self.assertTrue(np.array_equal(comp_diff, np.zeros_like(comp_diff)))
        return



class TestPCASVD(unittest.TestCase):
    def test_fit_equal_sklearn(self):
        iris = datasets.load_iris()
        p = iris.data.shape[1]
        k = 2
        X_reduced = PCA(n_components=k).fit(iris.data)
        X_reduced_svd = PCASVD(n_components=k).fit(iris.data)
        comp_diff = np.round(np.absolute(X_reduced.components_) - np.absolute(X_reduced_svd.components), 3)
        self.assertTrue(np.array_equal(comp_diff, np.zeros([k,p])))
        var_diff = np.round(X_reduced.explained_variance_ - X_reduced_svd.explained_variance, 3)
        self.assertTrue(np.array_equal(var_diff, np.zeros(k)))
        return

    def test_transform_equal_sklearn(self):
        iris = datasets.load_iris()

        k = 2
        PCA_sk = PCA(n_components=k, whiten=False).fit(iris.data)
        PCA_eign = PCASVD(n_components=k).fit(iris.data)
        X_reduced = PCA_sk.transform(iris.data)

        X_reduced_eig = PCA_eign.transform(iris.data)
        comp_diff = np.round(np.absolute(X_reduced) - np.absolute(X_reduced_eig), 3)
        self.assertTrue(np.array_equal(comp_diff, np.zeros_like(comp_diff)))
        return