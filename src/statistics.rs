use crate::{
    linalg::{Matrix, Vector},
    Distribution,
    Probability,
};
use spaces::Space;

pub trait UnivariateMoments: Distribution {
    /// Computes the expected value of the distribution.
    fn mean(&self) -> f64;

    /// Computes the variance of the distribution.
    fn variance(&self) -> f64 {
        let std = self.standard_deviation();

        std * std
    }

    /// Computes the standard deviation of the distribution.
    fn standard_deviation(&self) -> f64 { self.variance().sqrt() }

    /// Computes the skewness of the distribution.
    fn skewness(&self) -> f64;

    /// Computes the kurtosis of the distribution.
    fn kurtosis(&self) -> f64 { self.excess_kurtosis() + 3.0 }

    /// Computes the excess kurtosis of the distribution.
    fn excess_kurtosis(&self) -> f64 { self.kurtosis() - 3.0 }
}

pub trait MultivariateMoments: Distribution {
    /// Computes the vector of expected values of the distribution.
    fn mean(&self) -> Vector<f64>;

    /// Computes the vector of variances of the distribution.
    ///
    /// A default implementation is provided by calling `self.covariance()` and
    /// taking the diagonal. It is recommended, however, that you provide
    /// specialised version to remove the overhead of computing the
    /// off-diagonal terms.
    fn variance(&self) -> Vector<f64> { self.covariance().into_diag() }

    /// Computes the covariance matrix of the distribution.
    fn covariance(&self) -> Matrix<f64>;

    /// Computes the correlation matrix of the distribution.
    ///
    /// A default implementation is provided by calling `self.covariance()` and
    /// using the terms to compute `corr(Xi, Xj) = cov(Xi, Xj) /
    /// sqrt(var(Xi) var(Xj)).`
    fn correlation(&self) -> Matrix<f64> {
        let cov = self.covariance();

        Matrix::from_shape_fn(cov.dim(), |(i, j)| {
            cov[(i, j)] * (cov[(i, i)] * cov[(j, j)]).sqrt()
        })
    }
}

pub trait Quantiles: Distribution {
    /// Evaluates the inverse cumulative distribution function (CDF) at `p`.
    ///
    /// The quantile function specifies the value `x` of a random variable `X`
    /// at which the probability is less than or equal to `p`: `Q(p) = inf{
    /// x in R : p <= F(x)}`. For continuous and strictly monotonically
    /// increasing CDFs, it takes the exact form: `Q = 1 / F`.
    fn quantile(&self, p: Probability) -> f64 { self.cquantile(!p) }

    /// Evaluates the complementary quantile function at `x`: `Q(1 - p)`.
    fn cquantile(&self, p: Probability) -> f64 { self.quantile(!p) }

    /// Computes the lower quartile of the distribution, `Q(0.25)`.
    fn lower_quartile(&self) -> f64 { self.quantile(Probability::new_unchecked(0.25)) }

    /// Computes the median value of the distribution, `Q(0.5)`.
    ///
    /// A default implementation is provided by calling `self.quantile(0.5)`. It
    /// is recommended, however, that you provide specialised version as it
    /// is generally more efficient than evaluating `Q`.
    fn median(&self) -> f64 { self.quantile(Probability::half()) }

    /// Computes the upper quartile of the distribution, `Q(0.75)`.
    fn upper_quartile(&self) -> f64 { self.quantile(Probability::new_unchecked(0.75)) }

    /// Computes the interquartile range (IQR) of the distribution, `Q(0.75) -
    /// Q(0.25)`.
    fn iqr(&self) -> f64 { self.upper_quartile() - self.lower_quartile() }

    /// Computes the lower fence of the distribution, `Q(0.25) - 1.5 · IQR`.
    ///
    /// The lower fence is typically used to define the lower limit of the
    /// distribution, beyond which values are deemed outliers.
    fn lower_fence(&self) -> f64 { self.lower_quartile() - 1.5 * self.iqr() }

    /// Computes the lower fence of the distribution, `Q(0.75) + 1.5 · IQR`.
    ///
    /// The upper fence is typically used to define the upper limit of the
    /// distribution, beyond which values are deemed outliers.
    fn upper_fence(&self) -> f64 { self.upper_quartile() + 1.5 * self.iqr() }
}

pub trait Modes: Distribution {
    /// Computes the mode(s) of the distribution.
    fn modes(&self) -> Vec<<Self::Support as Space>::Value>;
}

pub trait ShannonEntropy: Distribution {
    /// Computes the Shannon entropy of the distribution, \\(H(X)\\).
    fn shannon_entropy(&self) -> f64;
}

pub trait FisherInformation: Distribution {
    /// Computes the Fisher information matrix of the distribution, \\(I(X)\\).
    fn fisher_information(&self) -> Matrix<f64>;
}
