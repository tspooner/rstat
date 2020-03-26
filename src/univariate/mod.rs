pub use crate::statistics::UnivariateMoments as Moments;

#[inline]
pub(self) fn factorial(n: u64) -> u64 { (2..n).product() }

#[inline]
pub(self) fn choose(n: u64, k: u64) -> u64 { (1..k).fold(n, |acc, i| acc * (n - i)) / factorial(k) }

// Miscellaneous:
mod uniform;
pub use self::uniform::*;

mod degenerate;
pub use self::degenerate::*;

// Continuous:
// mod arcsine;
// pub use self::arcsine::*;

// mod beta;
// pub use self::beta::*;

mod beta_prime;
pub use self::beta_prime::*;

mod cauchy;
pub use self::cauchy::*;

mod chi;
pub use self::chi::*;

mod chi_sq;
pub use self::chi_sq::*;

mod cosine;
pub use self::cosine::*;

mod erlang;
pub use self::erlang::*;

mod exponential;
pub use self::exponential::*;

mod f_dist;
pub use self::f_dist::*;

mod folded_normal;
pub use self::folded_normal::*;

mod frechet;
pub use self::frechet::*;

mod gamma;
pub use self::gamma::*;

mod gev;
pub use self::gev::*;

mod gpd;
pub use self::gpd::*;

mod gumbel;
pub use self::gumbel::*;

mod inverse_gamma;
pub use self::inverse_gamma::*;

mod inverse_normal;
pub use self::inverse_normal::*;

mod kumaraswamy;
pub use self::kumaraswamy::*;

mod laplace;
pub use self::laplace::*;

mod levy;
pub use self::levy::*;

mod logistic;
pub use self::logistic::*;

mod lognormal;
pub use self::lognormal::*;

mod normal;
pub use self::normal::*;

mod pareto;
pub use self::pareto::*;

mod rayleigh;
pub use self::rayleigh::*;

mod student_t;
pub use self::student_t::*;

mod triangular;
pub use self::triangular::*;

mod weibull;
pub use self::weibull::*;

// Discrete:
mod bernoulli;
pub use self::bernoulli::*;

mod beta_binomial;
pub use self::beta_binomial::*;

mod binomial;
pub use self::binomial::*;

mod categorical;
pub use self::categorical::*;

mod geometric;
pub use self::geometric::*;

mod poisson;
pub use self::poisson::*;
