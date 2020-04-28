//! Probability distributions and statistics in Rust with integrated fitting
//! routines, convolution support and mixtures.
// #![warn(missing_docs)]
extern crate failure;

extern crate rand;
extern crate rand_distr;

extern crate num;
extern crate special_fun;

extern crate spaces;

extern crate ndarray;
#[cfg(feature = "ndarray-linalg")]
extern crate ndarray_linalg;

#[cfg_attr(feature = "serde", macro_use)]
#[cfg(feature = "serde")]
extern crate serde_crate;

macro_rules! undefined {
    () => (panic!("quantity undefined"));
    ($($arg:tt)+) => (panic!("quantity undefined: {}", std::format_args!($($arg)+)));
}

mod consts;
mod utils;

pub mod linalg;

mod probability;
pub use self::probability::{
    InvalidProbabilityError, Probability,
    InvalidSimplexError, SimplexVector, UnitSimplex,
};

#[macro_use]
pub mod params;

/// Iterator for drawing random samples from a
/// [distribution](trait.Distribution.html).
pub struct Sampler<'a, D: ?Sized, R: ?Sized> {
    pub(crate) distribution: &'a D,
    pub(crate) rng: &'a mut R,
}

impl<'a, D, R> Iterator for Sampler<'a, D, R>
where
    D: Distribution + ?Sized,
    R: rand::Rng + ?Sized,
{
    type Item = <D::Support as spaces::Space>::Value;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> { Some(self.distribution.sample(&mut self.rng)) }

    fn size_hint(&self) -> (usize, Option<usize>) { (usize::max_value(), None) }
}

macro_rules! ln_variant {
    ($(#[$attr:meta])* => $name:ident, $name_ln:ident, $x:ty) => {
        $(#[$attr])*
        fn $name_ln(&self, x: $x) -> f64 {
            self.$name(x).ln()
        }
    }
}

/// Type alias for the sample type of a [distribution](trait.Distribution.html).
pub type Sample<D> = <<D as Distribution>::Support as spaces::Space>::Value;

/// Trait for probability distributions with a well-defined CDF.
pub trait Distribution: From<<Self as Distribution>::Params> {
    /// Support of sample elements.
    type Support: spaces::Space;

    /// Parameter set uniquely defining the instance.
    type Params;

    /// Returns an instance of the support `Space`, `Self::Support`.
    ///
    /// # Examples
    /// ```
    /// # use spaces::{Space, BoundedSpace, Dim};
    /// # use rstat::{Distribution, univariate, params::Param};
    /// let dist = univariate::beta::Beta::default();
    /// let support = dist.support();
    ///
    /// assert_eq!(support.dim(), Dim::Finite(1));
    /// assert_eq!(support.inf().unwrap(), 0.0);
    /// assert_eq!(support.sup().unwrap(), 1.0);
    /// ```
    fn support(&self) -> Self::Support;

    /// Converts `self` into an instance of `Self::Support`.
    fn into_support(self) -> Self::Support { self.support() }

    /// Returns an instance of the distribution parameters, `Self::Params`.
    ///
    /// # Examples
    /// ```
    /// # use rstat::{Distribution, univariate, params::Param};
    /// let dist = univariate::normal::Normal::standard();
    /// let params = dist.params();
    ///
    /// assert_eq!(params.mu.value(), &0.0);
    /// assert_eq!(params.Sigma.value(), &1.0);
    /// ```
    fn params(&self) -> Self::Params;

    /// Converts `self` into an instance of `Self::Params`.
    fn into_params(self) -> Self::Params { self.params() }

    /// Evaluates the cumulative distribution function (CDF) at \\(x\\).
    ///
    /// The CDF is defined as the probability that a random variable \\(X\\)
    /// takes on a value less than or equal to \\(x\\), i.e. \\(F(x) = P(X
    /// \leq x)\\).
    ///
    /// # Examples
    /// ```
    /// # use rstat::{Distribution, Probability, univariate};
    /// # use std::f64;
    /// let dist = univariate::normal::Normal::standard();
    ///
    /// assert_eq!(dist.cdf(&f64::NEG_INFINITY), Probability::zero());
    /// assert_eq!(dist.cdf(&0.0), Probability::half());
    /// assert_eq!(dist.cdf(&f64::INFINITY), Probability::one());
    /// ```
    fn cdf(&self, x: &Sample<Self>) -> Probability {
        Probability::new_unchecked(self.log_cdf(x).exp())
    }

    /// Evaluates the complementary CDF at \\(x\\).
    ///
    /// The complementary CDF (also known as the survival function) is defined
    /// as the probability that a random variable \\(X\\) takes on a value
    /// strictly greater than \\(x\\), i.e. \\(\bar{F}(x) = P(X > x) = 1 -
    /// F(x)\\).
    fn ccdf(&self, x: &Sample<Self>) -> Probability { !self.cdf(x) }

    ln_variant!(
        /// Evaluates the log CDF at \\(x\\), i.e. \\(\ln{F(x)}\\).
        => cdf, log_cdf, &Sample<Self>
    );

    ln_variant!(
        /// Evaluates the log complementary CDF at \\(x\\), i.e. \\(\ln{(1 - F(x))}\\).
        => ccdf, log_ccdf, &Sample<Self>
    );

    /// Draw a random value from the distribution support.
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Sample<Self>;

    /// Draw n random value from the distribution support.
    fn sample_n<R: rand::Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<Sample<Self>> {
        (0..n).into_iter().map(move |_| self.sample(rng)).collect()
    }

    /// Draw an indefinite number of random values from the distribution
    /// support.
    fn sample_iter<'a, R: rand::Rng + ?Sized>(&'a self, rng: &'a mut R) -> Sampler<'a, Self, R> {
        Sampler {
            distribution: self,
            rng,
        }
    }
}

macro_rules! new_dist {
    ($(#[$attr:meta])* $name:ident<$pt:ty>) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        pub struct $name(pub(crate) $pt);

        impl From<$pt> for $name {
            fn from(from: $pt) -> $name { $name(from) }
        }
    };
}

#[inline]
pub fn params_of<D: Distribution>(dist: &D) -> D::Params { dist.params() }

#[inline]
pub fn support_of<D: Distribution>(dist: &D) -> D::Support { dist.support() }

/// Trait for [distributions](trait.Distribution.html) over a countable
/// `Support`.
///
/// The PMF is defined as the probability that a random variable \\(X\\) takes a
/// value exactly equal to \\(x\\), i.e. \\(f(x) = P(X = x) = P(s \in S : X(s) =
/// x)\\). We require that all sum of probabilities over all possible outcomes
/// sums to 1.
pub trait DiscreteDistribution: Distribution {
    /// Evaluates the probability mass function (PMF) at \\(x\\).
    ///
    /// # Examples
    /// ```
    /// # use rstat::{DiscreteDistribution, Probability, univariate::binomial::Binomial};
    /// let dist = Binomial::new_unchecked(5, Probability::new_unchecked(0.75));
    ///
    /// assert_eq!(dist.pmf(&0), Probability::new_unchecked(0.0009765625));
    /// assert_eq!(dist.pmf(&1), Probability::new_unchecked(0.0146484375));
    /// assert_eq!(dist.pmf(&2), Probability::new_unchecked(0.087890625));
    /// assert_eq!(dist.pmf(&3), Probability::new_unchecked(0.263671875));
    /// assert_eq!(dist.pmf(&4), Probability::new_unchecked(0.3955078125));
    /// assert_eq!(dist.pmf(&5), Probability::new_unchecked(0.2373046875));
    /// ```
    fn pmf(&self, x: &Sample<Self>) -> Probability {
        Probability::new_unchecked(self.log_pmf(x).exp())
    }

    ln_variant!(
        /// Evaluates the log PMF at \\(x\\).
        => pmf, log_pmf, &Sample<Self>
    );
}

/// Trait for [distributions](trait.Distribution.html) with an absolutely
/// continuous CDF.
///
/// The PDF can be interpreted as the relative likelihood that a random variable
/// \\(X\\) takes on a value equal to \\(x\\). For absolutely continuous
/// univariate distributions it is defined by the derivative of the CDF, i.e
/// \\(f(x) = F'(x)\\). Intuitively, one may think of \\(f(x)\text{d}x\\) that
/// as representing the probability that the random variable \\(X\\) lies in the
/// infinitesimal interval \\([x, x + \text{d}x]\\). Alternatively, one can
/// interpret the PDF, for infinitesimally small \\(\text{d}t\\), as:
/// \\(f(t)\text{d}t = P(t < X < t + \text{d}t)\\). For a finite interval \\([a,
/// b],\\) we have that: \\[P(a < X < b) = \int_a^b f(t)\text{d}t.\\]
pub trait ContinuousDistribution: Distribution {
    /// Evaluates the probability density function (PDF) at \\(x\\).
    ///
    /// # Examples
    /// ```
    /// # use rstat::{ContinuousDistribution, Probability, univariate::triangular::Triangular};
    /// let dist = Triangular::new_unchecked(0.0, 0.5, 0.5);
    ///
    /// assert_eq!(dist.pdf(&0.0), 0.0);
    /// assert_eq!(dist.pdf(&0.25), 1.0);
    /// assert_eq!(dist.pdf(&0.5), 2.0);
    /// assert_eq!(dist.pdf(&0.75), 1.0);
    /// assert_eq!(dist.pdf(&1.0), 0.0);
    /// ```
    fn pdf(&self, x: &Sample<Self>) -> f64 { self.log_pdf(x).exp() }

    ln_variant!(
        /// Evaluates the log PDF at \\(x\\).
        => pdf, log_pdf, &Sample<Self>
    );
}

/// Trait for [distributions](trait.Distribution.html) that support the convolve
/// operation.
///
/// The convolution of probability [distributions](trait.Distribution.html)
/// amounts to taking linear combinations of independent random variables. For
/// example, consider a set of \\(N\\) random variables \\(X_i \sim
/// \text{Bernoulli}(p)\\), where \\(p \in (0, 1)\\) and \\(1 \leq i \leq N\\).
/// We then have that the random variables \\(Y = \sum_{i=1}^N X_i\\) and \\(Z
/// \sim \text{Binomial}(N, p)\\) are exactly equivalent, i.e. \\(Y
/// \stackrel{\text{d}}{=} Z\\).
pub trait Convolution<T: Distribution = Self> {
    /// The resulting [Distribution](trait.Distribution.html) type.
    type Output: Distribution;

    /// Return the unweighted linear sum of `self` with another
    /// [Distribution](trait.Distribution.html) of type `T`.
    ///
    /// # Examples
    /// ```
    /// # use rstat::{Distribution, Convolution, params::Param, univariate::normal::Normal};
    /// let dist_a = Normal::new_unchecked(0.0, 1.0f64.powi(2));
    /// let dist_b = Normal::new_unchecked(1.0, 2.0f64.powi(2));
    ///
    /// let dist_c = dist_a.convolve(dist_b).unwrap();
    /// let params = dist_c.params();
    ///
    /// assert_eq!(params.mu.value(), &1.0);
    /// assert_eq!(params.Sigma.value(), &5.0f64);
    /// ```
    fn convolve(self, rv: T) -> Result<Self::Output, failure::Error>;

    /// Return the unweighted linear sum of `self` with a set of
    /// [Distributions](trait.Distribution.html) of type `T`.
    fn convolve_many(self, mut rvs: Vec<T>) -> Result<Self::Output, failure::Error>
    where
        Self::Output: Convolution<T, Output = Self::Output>,
        Self: Sized,
    {
        let n = rvs.len();
        let _ = assert_constraint!(n > 1).map_err(|e| e.with_target("n"))?;

        let new_dist = self.convolve(rvs.pop().unwrap());

        rvs.into_iter()
            .fold(new_dist, |acc, rv| acc.and_then(|d| d.convolve(rv)))
    }
}

pub mod metrics;
pub mod statistics;
pub mod fitting;

pub mod normal;
pub mod univariate;
pub mod bivariate;
pub mod multivariate;

mod mixture;
pub use self::mixture::Mixture;

pub mod builder;
