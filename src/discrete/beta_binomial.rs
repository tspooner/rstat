use crate::core::*;
use rand::Rng;
use spaces::discrete::Ordinal;
use special_fun::FloatSpecial;
use std::fmt;
use super::choose;

#[derive(Debug, Clone, Copy)]
pub struct BetaBinomial {
    pub n: usize,

    pub alpha: f64,
    pub beta: f64,

    pi: f64,
    rho: f64,
}

impl BetaBinomial {
    pub fn new(n: usize, alpha: f64, beta: f64) -> BetaBinomial {
        assert_positive_real!(alpha);
        assert_positive_real!(beta);

        BetaBinomial {
            n,

            alpha,
            beta,

            pi: alpha / (alpha + beta),
            rho: 1.0 / (alpha + beta + 1.0),
        }
    }

    fn pmf_raw(&self, k: usize) -> f64 {
        let c = choose(self.n as u64, k as u64) as f64;
        let z = self.alpha.beta(self.beta);
        let k = k as f64;

        c * (k + self.alpha).beta(self.n as f64 - k + self.beta) / z
    }
}

impl Distribution for BetaBinomial {
    type Support = Ordinal;

    fn support(&self) -> Ordinal { Ordinal::new(self.n) }

    fn cdf(&self, k: usize) -> Probability {
        if k >= self.n {
            1.0
        } else {
            unimplemented!("Need an implmentation of the 3F2 generalised hypergeometric function.")
        }.into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> usize {
        unimplemented!()
    }
}

impl DiscreteDistribution for BetaBinomial {
    fn pmf(&self, k: usize) -> Probability {
        self.pmf_raw(k).into()
    }
}

impl UnivariateMoments for BetaBinomial {
    fn mean(&self) -> f64 {
        self.n as f64 * self.pi
    }

    fn variance(&self) -> f64 {
        let n = self.n as f64;

        n * self.pi * (1.0 - self.pi) * (1.0 + (n - 1.0) * self.rho)
    }

    fn skewness(&self) -> f64 {
        let n = self.n as f64;
        let ap1 = self.alpha + 1.0;
        let ap2 = ap1 + 1.0;

        // Compute bracket term:
        let b =
            n * n * ap1 * ap2 +
            3.0 * n * ap1 * self.beta +
            self.beta * (self.beta - self.alpha);

        self.pi * self.rho * n * b / (ap2 + self.beta)
    }

    fn kurtosis(&self) -> f64 {
        let n = self.n as f64;
        let apb = self.alpha + self.beta;

        let b1 = self.alpha / (
            n * self.pi * self.pi * self.rho * self.beta *
            (apb + 2.0) * (apb + 3.0) * (apb + n)
        );
        let b2 = (apb - 1.0 + 6.0 * n) / self.alpha;
        let b3 = 3.0 * self.alpha * self.beta * (n - 2.0);
        let b4 = 6.0 * n * n;
        let b5 = -3.0 * self.pi * self.beta * n * (6.0 - n);
        let b6 = -18.0 * self.pi * self.beta * n * n / apb;

        b1 * (b2 + b3 + b4 + b5 + b6)
    }
}

impl fmt::Display for BetaBinomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BB({}, {}, {})", self.n, self.alpha, self.beta)
    }
}
