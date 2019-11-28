// TODO: Hypergeometric distribution
// TODO: NegativeBinomial distribution
// TODO: PoissonBinomial distribution
// TODO: Skellam distribution

#[inline]
pub(self) fn factorial(n: u64) -> u64 { (2..n).product() }

#[inline]
pub(self) fn choose(n: u64, k: u64) -> u64 { (1..k).fold(n, |acc, i| acc * (n - i)) / factorial(k) }

pub use crate::DiscreteDistribution;

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

pub type Uniform = super::Uniform<i64>;
