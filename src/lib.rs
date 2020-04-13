//! Probability distributions and statistics in Rust with integrated fitting routines, convolution
//! support and mixtures.
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
extern crate serde;

#[macro_use]
mod macros;

mod consts;
mod linalg;

mod prelude;

mod probability;
pub use self::probability::{Probability, ProbabilityError};

mod simplex;
pub use self::simplex::{UnitSimplex, SimplexVector, SimplexError};

#[macro_use]
pub mod params;

/// Iterator for drawing random samples from a [distribution](trait.Distribution.html).
pub struct Sampler<'a, D: ?Sized, R: ?Sized> {
    pub(crate) distribution: &'a D,
    pub(crate) rng: &'a mut R,
}

impl<'a, D, R> Iterator for Sampler<'a, D, R>
    where D: Distribution + ?Sized,
          R: rand::Rng + ?Sized,
{
    type Item = <D::Support as spaces::Space>::Value;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.distribution.sample(&mut self.rng))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::max_value(), None)
    }
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
    fn support(&self) -> Self::Support;

    fn into_support(self) -> Self::Support { self.support() }

    /// Returns an instance of `Self::Params` matching the parameters of `self`.
    fn params(&self) -> Self::Params;

    fn into_params(self) -> Self::Params { self.params() }

    /// Evaluates the cumulative distribution function (CDF) at \\(x\\).
    ///
    /// The CDF is defined as the probability that a random variable \\(X\\) takes on a value less
    /// than or equal to \\(x\\), i.e. \\(F(x) = P(X \leq x)\\).
    fn cdf(&self, x: &Sample<Self>) -> Probability;

    /// Evaluates the complementary CDF at \\(x\\).
    ///
    /// The complementary CDF (also known as the survival function) is defined as the probability
    /// that a random variable \\(X\\) takes on a value strictly greater than \\(x\\), i.e.
    /// \\(\bar{F}(x) = P(X > x) = 1 - F(x)\\).
    fn ccdf(&self, x: &Sample<Self>) -> Probability {
        !self.cdf(x)
    }

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

    /// Draw an indefinite number of random values from the distribution support.
    fn sample_iter<'a, R: rand::Rng + ?Sized>(&'a self, rng: &'a mut R) -> Sampler<'a, Self, R> {
        Sampler {
            distribution: self,
            rng,
        }
    }
}

macro_rules! new_dist {
    ($name:ident<$pt:ty>) => {
        #[derive(Debug, Clone, Copy)]
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

/// Trait for [distributions](trait.Distribution.html) over a countable `Support`.
///
/// The PMF is defined as the probability that a random variable \\(X\\) takes a value exactly
/// equal to \\(x\\), i.e. \\(f(x) = P(X = x) = P(s \in S : X(s) = x)\\). We require that
/// all sum of probabilities over all possible outcomes sums to 1.
pub trait DiscreteDistribution: Distribution {
    /// Evaluates the probability mass function (PMF) at \\(x\\).
    fn pmf(&self, x: &Sample<Self>) -> Probability;

    ln_variant!(
        /// Evaluates the log PMF at \\(x\\).
        => pmf, log_pmf, &Sample<Self>
    );
}

/// Trait for [distributions](trait.Distribution.html) with an absolutely continuous CDF.
///
/// The PDF can be interpreted as the relative likelihood that a random variable \\(X\\) takes on a
/// value equal to \\(x\\). For absolutely continuous univariate distributions it is defined by the
/// derivative of the CDF, i.e \\(f(x) = F'(x)\\). Intuitively, one may think of
/// \\(f(x)\text{d}x\\) that as representing the probability that the random variable \\(X\\) lies
/// in the infinitesimal interval \\([x, x + \text{d}x]\\). Alternatively, one can interpret the
/// PDF, for infinitesimally small \\(\text{d}t\\), as: \\(f(t)\text{d}t = P(t < X < t +
/// \text{d}t)\\). For a finite interval \\([a, b],\\) we have that: \\[P(a < X < b) = \int_a^b
/// f(t)\text{d}t.\\]
pub trait ContinuousDistribution: Distribution {
    /// Evaluates the probability density function (PDF) at \\(x\\).
    fn pdf(&self, x: &Sample<Self>) -> f64;

    ln_variant!(
        /// Evaluates the log PDF at \\(x\\).
        => pdf, log_pdf, &Sample<Self>
    );
}

/// Trait for [distributions](trait.Distribution.html) that support the convolve operation.
///
/// The convolution of probability [distributions](trait.Distribution.html) amounts to taking
/// linear combinations of independent random variables. For example, consider a set of \\(N\\)
/// random variables \\(X_i \sim \text{Bernoulli}(p)\\), where \\(p \in (0, 1)\\) and \\(1 \leq i
/// \leq N\\). We then have that the random variables \\(Y = \sum_{i=1}^N X_i\\) and \\(Z \sim
/// \text{Binomial}(N, p)\\) are exactly equivalent, i.e. \\(Y \stackrel{\text{d}}{=} Z\\).
pub trait Convolution<T: Distribution = Self> {
    /// The resulting [Distribution](trait.Distribution.html) type.
    type Output: Distribution;

    /// Return the unweighted linear sum of `self` with another
    /// [Distribution](trait.Distribution.html) of type `T`.
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

        rvs.into_iter().fold(new_dist, |acc, rv| {
            acc.and_then(|d| d.convolve(rv))
        })
    }
}

pub mod statistics;
pub mod fitting;

pub mod univariate;
pub mod multivariate;

mod mixture;
pub use self::mixture::Mixture;

// #[derive(Clone, Debug)]
// pub struct RandomVariable<D: Distribution> {
    // pub symbol: String,
    // pub distribution: D,
// }

// impl<D: Distribution> RandomVariable<D> {
    // pub fn new<S: ToString>(symbol: S, distribution: D) -> RandomVariable<D> {
        // RandomVariable {
            // symbol: symbol.to_string(),
            // distribution,
        // }
    // }
// }

// impl<D: Distribution> std::fmt::Display for RandomVariable<D> {
    // fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // write!(f, "{}", self.symbol)
    // }
// }

// #[cfg(test)]
// mod tests {
    // use super::{Probability, RandomVariable};
    // use crate::univariate::bernoulli::Bernoulli;

    // #[test]
    // fn test_rv() {
        // let dist = Bernoulli::new(Probability::half());
        // let coin = RandomVariable::new("X", dist);

        // println!("{} ~ {}", coin, coin.distribution);

        // assert!(false);
    // }
// }
