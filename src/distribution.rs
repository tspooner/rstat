use crate::Probability;
use ndarray::{Array, Array1, Dimension, ShapeBuilder};
use rand::Rng;
use spaces::Space;

pub struct Sampler<D, R> {
    pub(super) distribution: D,
    pub(super) rng: R,
}

impl<D, R> Iterator for Sampler<D, R>
    where D: Distribution,
          R: Rng,
{
    type Item = <D::Support as Space>::Value;

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
            f64::from(self.$name(x)).ln()
        }
    }
}

macro_rules! batch_variant {
    ($(#[$attr:meta])* => $name:ident, $name_batch:ident, $x:ty, $res:ty) => {
        $(#[$attr])*
        fn $name_batch(&self, xs: Array1<$x>) -> Array1<$res> {
            xs.mapv(|x| self.$name(x))
        }
    }
}

pub type Sample<D> = <<D as Distribution>::Support as Space>::Value;

pub trait Distribution {
    type Support: Space;

    /// Returns an instance of the support `Space`, `Self::Support`.
    fn support(&self) -> Self::Support;

    /// Evaluates the cumulative distribution function (CDF) at `x`.
    ///
    /// The CDF is defined as the probability that a random variable X takes on a value less than
    /// or equal to `x`: `F(x) = P(X <= x)`.
    fn cdf(&self, x: Sample<Self>) -> Probability;

    /// Evaluates the complementary cumulative distribution function at `x`.
    ///
    /// The complementary CDF is defined as the probability that a random variable X takes on a
    /// value strictly greater than `x`: `P(X > x) = 1 - F(x)`, where `F(.)` is the CDF.
    fn ccdf(&self, x: Sample<Self>) -> Probability {
        !self.cdf(x)
    }

    ln_variant!(
        /// Evaluates the log CDF at `x`: `ln F(x)`.
        => cdf, logcdf, Sample<Self>
    );

    ln_variant!(
        /// Evaluates the log complementary CDF at `x`: `ln (1 - F(x))`.
        => ccdf, logccdf, Sample<Self>
    );

    batch_variant!(
        /// Evaluates the CDF element-wise for a batch `xs`.
        => cdf, cdf_batch, Sample<Self>, Probability
    );

    batch_variant!(
        /// Evaluates the complementary CDF element-wise for a batch `xs`.
        => ccdf, ccdf_batch, Sample<Self>, Probability
    );

    batch_variant!(
        /// Evaluates the log CDF element-wise for a batch `xs`.
        => logcdf, logcdf_batch, Sample<Self>, f64
    );

    batch_variant!(
        /// Evaluates the log complementary CDF element-wise for a batch `xs`.
        => logccdf, logccdf_batch, Sample<Self>, f64
    );

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Sample<Self>;

    fn sample_n<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<Sample<Self>> {
        (0..n).into_iter().map(move |_| self.sample(rng)).collect()
    }

    fn sample_shape<D, Sh, R>(&self, rng: &mut R, shape: Sh) -> Array<Sample<Self>, D>
        where D: Dimension,
              Sh: ShapeBuilder<Dim=D>,
              R: Rng + ?Sized,
    {
        Array::from_shape_fn(shape, move |_| self.sample(rng))
    }

    fn sample_iter<R>(self, rng: R) -> Sampler<Self, R>
        where Self: Sized,
              R: Rng,
    {
        Sampler {
            distribution: self,
            rng: rng,
        }
    }
}

pub trait DiscreteDistribution: Distribution {
    /// Evaluates the probability mass function (PMF) at `x`.
    ///
    /// The PMF is defined as the probability that a random variable `X` takes a value exactly
    /// equal to `x`: `f(x) = P(X = x) = P({s in S : X(s) = x})`. We require that all sum of
    /// probabilities over all possible outcomes sums to 1.
    fn pmf(&self, x: Sample<Self>) -> Probability {
        Probability::new_unchecked(self.logpmf(x).exp())
    }

    ln_variant!(
        /// Evaluates the log PMF at `x`.
        => pmf, logpmf, Sample<Self>
    );

    batch_variant!(
        /// Evaluates the PMF element-wise for a batch `xs`.
        => pmf, pmf_batch, Sample<Self>, Probability
    );

    batch_variant!(
        /// Evaluates the log PMF element-wise for a batch `xs`.
        => logpmf, logpmf_batch, Sample<Self>, f64
    );
}

pub trait ContinuousDistribution: Distribution {
    /// Evaluates the probability density function (PDF) at `x`.
    ///
    /// The PDF can be interpreted as the relative likelihood that a random variable X takes on a
    /// value equal to `x`. For absolutely continuous univariate distributions it is defined by the
    /// derivative of the CDF: `f(x) = F'(x)`. Intuitively, one may think of `f(x)dx` that as
    /// representing the probability that the random variable `X` lies in the infinitesimal
    /// interval `[x, x + dx]`.
    ///
    /// Alternatively, one may interpret the PDF, for infinitely small `dt`, as the following:
    /// `f(t)dt = P(t < X < t + dt)`.
    fn pdf(&self, x: Sample<Self>) -> f64 {
        self.logpdf(x).exp().into()
    }

    ln_variant!(
        /// Evaluates the log PDF at `x`.
        => pdf, logpdf, Sample<Self>
    );

    batch_variant!(
        /// Evaluates the PDF element-wise for a batch `xs`.
        => pdf, pdf_batch, Sample<Self>, f64
    );

    batch_variant!(
        /// Evaluates the log PDF element-wise for a batch `xs`.
        => logpdf, logpdf_batch, Sample<Self>, f64
    );
}
