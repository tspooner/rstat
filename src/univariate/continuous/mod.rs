// TODO: Noncentral chi-squared distribution
// TODO: Normal-inverse Gaussian distribution
// TODO: VonMises distribution

pub use crate::ContinuousDistribution;

mod arcsine;
pub use self::arcsine::*;

mod beta;
pub use self::beta::*;

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

pub type Uniform = super::Uniform<f64>;
