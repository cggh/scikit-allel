# -*- coding: utf-8 -*-
import numpy as np


from allel.stats.preprocessing import get_scaler


def pca(gn, n_components=10, copy=True, scaler="patterson", ploidy=2):
    """Perform principal components analysis of genotype data, via singular
    value decomposition.

    Parameters
    ----------

    gn : array_like, float, shape (n_variants, n_samples)
        Genotypes at biallelic variants, coded as the number of alternate
        alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).
    n_components : int, optional
        Number of components to keep.
    copy : bool, optional
        If False, data passed to fit are overwritten.
    scaler : {'patterson', 'standard', None}
        Scaling method; 'patterson' applies the method of Patterson et al
        2006; 'standard' scales to unit variance; None centers the data only.
    ploidy : int, optional
        Sample ploidy, only relevant if 'patterson' scaler is used.

    Returns
    -------

    coords : ndarray, float, shape (n_samples, n_components)
        Transformed coordinates for the samples.
    model : GenotypePCA
        Model instance containing the variance ratio explained and the stored
        components (a.k.a., loadings). Can be used to project further data
        into the same principal components space via the transform() method.

    Notes
    -----

    Genotype data should be filtered prior to using this function to remove
    variants in linkage disequilibrium.

    See Also
    --------

    randomized_pca, allel.stats.ld.locate_unlinked

    """

    # set up the model
    model = GenotypePCA(n_components, copy=copy, scaler=scaler, ploidy=ploidy)

    # fit the model and project the input data onto the new dimensions
    coords = model.fit_transform(gn)

    return coords, model


class GenotypePCA(object):
    def __init__(self, n_components=10, copy=True, scaler="patterson", ploidy=2):
        self.n_components = n_components
        self.copy = copy
        self.scaler = scaler
        self.scaler_ = get_scaler(scaler, copy, ploidy)

    def fit(self, gn):
        self._fit(gn)
        return self

    def fit_transform(self, gn):
        self.u, s, v = self._fit(gn)
        self.u = self.u[:, : self.n_components]
        self.u *= s[: self.n_components]
        return self.u

    def _fit(self, gn):
        import scipy.linalg

        # apply scaling
        gn = self.scaler_.fit(gn).transform(gn)

        # transpose for svd
        x = gn.T
        n_samples, n_features = x.shape

        # singular value decomposition
        u, s, v = scipy.linalg.svd(x, full_matrices=False)

        # calculate explained variance
        explained_variance_ = (s**2) / n_samples
        explained_variance_ratio_ = explained_variance_ / np.sum(explained_variance_)

        # store variables
        n_components = self.n_components
        self.components_ = v[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]

        return u, s, v

    def transform(self, gn, copy=None):
        if not hasattr(self, "components_"):
            raise ValueError("model has not been not fitted")

        # scaling
        gn = self.scaler_.transform(gn, copy=copy)

        # transpose for transformation
        x = gn.T

        # apply transformation
        x_transformed = np.dot(x, self.components_.T)

        return x_transformed

    def project(self, gnp, missing=-1):
        """
        Project samples with missing variants using the precomputed eigenvectors.

        Parameters
        ----------
        gnp : array_like, float, shape (n_variants, n_samples)
            Genotypes at biallelic variants for samples to project,
            coded as the number of alternate alleles per call.
            (i.e., 0 = hom ref, 1 = het, 2 = hom alt, -1 = missing).
        missing : int, optional
            Value used to represent missing genotypes in the input data.

        Returns
        -------
        array-like, shape (n_samples, n_components):
            PCA projection of both base and projected missing samples.

        Notes
        -----

        This is the implementation of the least squares projection method
        first implemented in smartPCA by Patterson in 2013, described here:
        https://github.com/chrchang/eigensoft/blob/master/POPGEN/lsqproject.pdf

        It is used to project samples with missing variants in the PC space defined
        by samples with no missing variants.
        """

        if not hasattr(self, "components_"):
            raise ValueError("model has not been fitted")
        if not hasattr(self, "u"):
            raise ValueError("genotype data has not been transformed")

        gnp = self.scaler_.transform(gnp, copy=None).astype(
            np.float32, copy=False
        )  # cast to float32 for np.linalg

        projected_missing_samples = np.zeros((gnp.shape[1], self.n_components))

        for i, sample in enumerate(gnp.T):
            # Identify non-missing entries in the current sample
            non_missing_variants = sample != missing

            # Subset the eigenvectors to only the non-missing variants
            eigenvectors_no_missing = self.components_[:, non_missing_variants].T
            X_non_missing = sample[non_missing_variants]

            # Ensure dimensional compatibility and project using least squares
            if eigenvectors_no_missing.shape[1] > 0 and eigenvectors_no_missing.shape[
                0
            ] == len(X_non_missing):
                projected_missing_samples[i] = np.linalg.lstsq(
                    eigenvectors_no_missing, X_non_missing, rcond=None
                )[0]
            else:
                projected_missing_samples[i] = np.zeros(
                    self.n_components
                )  # Handle missing projection

        # Stack the base samples with the projected missing samples
        return np.vstack([self.u, projected_missing_samples])


def randomized_pca(
    gn,
    n_components=10,
    copy=True,
    iterated_power=3,
    random_state=None,
    scaler="patterson",
    ploidy=2,
):
    """Perform principal components analysis of genotype data, via an
    approximate truncated singular value decomposition using randomization
    to speed up the computation.

    Parameters
    ----------

    gn : array_like, float, shape (n_variants, n_samples)
        Genotypes at biallelic variants, coded as the number of alternate
        alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).
    n_components : int, optional
        Number of components to keep.
    copy : bool, optional
        If False, data passed to fit are overwritten.
    iterated_power : int, optional
        Number of iterations for the power method.
    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.
    scaler : {'patterson', 'standard', None}
        Scaling method; 'patterson' applies the method of Patterson et al
        2006; 'standard' scales to unit variance; None centers the data only.
    ploidy : int, optional
        Sample ploidy, only relevant if 'patterson' scaler is used.

    Returns
    -------

    coords : ndarray, float, shape (n_samples, n_components)
        Transformed coordinates for the samples.
    model : GenotypeRandomizedPCA
        Model instance containing the variance ratio explained and the stored
        components (a.k.a., loadings). Can be used to project further data
        into the same principal components space via the transform() method.

    Notes
    -----

    Genotype data should be filtered prior to using this function to remove
    variants in linkage disequilibrium.

    Based on the :class:`sklearn.decomposition.RandomizedPCA` implementation.

    See Also
    --------

    pca, allel.stats.ld.locate_unlinked

    """

    # set up the model
    model = GenotypeRandomizedPCA(
        n_components,
        copy=copy,
        iterated_power=iterated_power,
        random_state=random_state,
        scaler=scaler,
        ploidy=ploidy,
    )

    # fit the model and project the input data onto the new dimensions
    coords = model.fit_transform(gn)

    return coords, model


class GenotypeRandomizedPCA(object):
    def __init__(
        self,
        n_components=10,
        copy=True,
        iterated_power=3,
        random_state=None,
        scaler="patterson",
        ploidy=2,
    ):
        self.n_components = n_components
        self.copy = copy
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.scaler = scaler
        self.scaler_ = get_scaler(scaler, copy, ploidy)

    def fit(self, gn):
        self._fit(gn)
        return self

    def fit_transform(self, gn):
        self.u, s, v = self._fit(gn)
        self.u *= s
        return self.u

    def _fit(self, gn):
        from sklearn.utils.validation import check_random_state
        from sklearn.utils.extmath import randomized_svd

        # apply scaling
        gn = self.scaler_.fit(gn).transform(gn)

        # transpose for svd
        # TODO eliminate need for transposition
        x = gn.T

        # intermediates
        random_state = check_random_state(self.random_state)
        n_components = self.n_components
        n_samples, n_features = x.shape

        # singular value decomposition
        u, s, v = randomized_svd(
            x, n_components, n_iter=self.iterated_power, random_state=random_state
        )

        # calculate explained variance
        self.explained_variance_ = exp_var = (s**2) / n_samples
        full_var = np.var(x, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var

        # store components
        self.components_ = v

        return u, s, v

    def transform(self, gn, copy=None):
        if not hasattr(self, "components_"):
            raise ValueError("model has not been not fitted")

        # scaling
        gn = self.scaler_.transform(gn, copy=copy)

        # transpose for transformation
        # TODO eliminate need for transposition
        x = gn.T

        # apply transformation
        x_transformed = np.dot(x, self.components_.T)

        return x_transformed

    def project(self, gnp, missing=-1):
        """
        Project samples with missing variants using the precomputed eigenvectors.

        Parameters
        ----------
        gnp : array_like, float, shape (n_variants, n_samples)
            Genotypes at biallelic variants for samples to project,
            coded as the number of alternate alleles per call.
            (i.e., 0 = hom ref, 1 = het, 2 = hom alt, -1 = missing).
        missing : int, optional
            Value used to represent missing genotypes in the input data.

        Returns
        -------
        array-like, shape (n_samples, n_components):
            PCA projection of both base and projected missing samples.

        Notes
        -----

        This is the implementation of the least squares projection method
        first implemented in smartPCA by Patterson in 2013, described here:
        https://github.com/chrchang/eigensoft/blob/master/POPGEN/lsqproject.pdf

        It is used to project samples with missing variants in the PC space defined
        by samples with no missing variants.
        """

        if not hasattr(self, "components_"):
            raise ValueError("model has not been fitted")
        if not hasattr(self, "u"):
            raise ValueError("genotype data has not been transformed")

        gnp = self.scaler_.transform(gnp, copy=None).astype(
            np.float32, copy=False
        )  # cast to float32 for np.linalg

        projected_missing_samples = np.zeros((gnp.shape[1], self.n_components))

        for i, sample in enumerate(gnp.T):
            # Identify non-missing entries in the current sample
            non_missing_variants = sample != missing

            # Subset the eigenvectors to only the non-missing variants
            eigenvectors_no_missing = self.components_[:, non_missing_variants].T
            X_non_missing = sample[non_missing_variants]

            # Ensure dimensional compatibility and project using least squares
            if eigenvectors_no_missing.shape[1] > 0 and eigenvectors_no_missing.shape[
                0
            ] == len(X_non_missing):
                projected_missing_samples[i] = np.linalg.lstsq(
                    eigenvectors_no_missing, X_non_missing, rcond=None
                )[0]
            else:
                projected_missing_samples[i] = np.zeros(
                    self.n_components
                )  # Handle missing projection

        # Stack the base samples with the projected missing samples
        return np.vstack([self.u, projected_missing_samples])
