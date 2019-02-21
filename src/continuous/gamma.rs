use crate::{
    continuous::Exponential,
    core::*,
};
use rand::Rng;
use spaces::continuous::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    pub alpha: f64,
    pub beta: f64,
}

impl Gamma {
    pub fn new(alpha: f64, beta: f64) -> Gamma {
        assert_positive_real!(alpha);
        assert_positive_real!(beta);

        Gamma { alpha, beta }
    }

    pub fn with_scale(k: f64, theta: f64) -> Gamma {
        Gamma::new(k, 1.0 / theta)
    }
}

impl Default for Gamma {
    fn default() -> Gamma {
        Gamma {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl Distribution for Gamma {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        (self.alpha.gammainc(self.beta * x) / self.alpha.gamma()).into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Gamma {
    fn pdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        (self.beta.powf(self.alpha) * x.powf(self.alpha - 1.0) * (-self.beta * x).exp()
            / self.alpha.gamma())
        .into()
    }
}

impl UnivariateMoments for Gamma {
    fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    fn variance(&self) -> f64 {
        self.alpha / self.beta / self.beta
    }

    fn skewness(&self) -> f64 {
        2.0 / self.alpha.sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        6.0 / self.alpha
    }
}

impl Modes for Gamma {
    fn modes(&self) -> Vec<f64> {
        if self.alpha < 1.0 {
            unimplemented!("Mode is undefined for alpha < 1.")
        }

        vec![(self.alpha - 1.0) / self.beta]
    }
}

impl Entropy for Gamma {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        self.alpha - self.beta.ln()
            + self.alpha.gamma().ln()
            + (1.0 - self.alpha) * self.alpha.digamma()
    }
}

impl Convolution<Gamma> for Gamma {
    fn convolve(self, rv: Gamma) -> ConvolutionResult<Gamma> {
        Self::convolve_pair(self, rv)
    }

    fn convolve_pair(a: Gamma, b: Gamma) -> ConvolutionResult<Gamma> {
        if a.beta == b.beta {
            Ok(Gamma::new(a.alpha + b.alpha, a.beta))
        } else {
            Err(ConvolutionError::MixedParameters)
        }
    }
}

impl Convolution<Exponential> for Gamma {
    fn convolve(self, rv: Exponential) -> ConvolutionResult<Gamma> {
        if rv.lambda == self.beta {
            Ok(Gamma::new(self.alpha + 1.0, self.beta))
        } else {
            Err(ConvolutionError::MixedParameters)
        }
    }

    fn convolve_pair(a: Exponential, b: Exponential) -> ConvolutionResult<Gamma> {
        if a.lambda == b.lambda {
            Ok(Gamma::new(2.0, a.lambda))
        } else {
            Err(ConvolutionError::MixedParameters)
        }
    }
}

impl fmt::Display for Gamma {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Gamma({}, {})", self.alpha, self.beta)
    }
}
