// TODO: Hypergeometric distribution
// TODO: NegativeBinomial distribution
// TODO: PoissonBinomial distribution
// TODO: Skellam distribution

#[inline]
pub(self) fn factorial(n: u64) -> u64 { (2..n).product() }

#[inline]
pub(self) fn choose(n: u64, k: u64) -> u64 { (1..k).fold(n, |acc, i| acc * (n - i)) / factorial(k) }

pub use core::DiscreteDistribution;

import_all!(bernoulli);
import_all!(beta_binomial);
import_all!(binomial);
import_all!(categorical);
import_all!(geometric);
import_all!(multinomial);
import_all!(poisson);
import_all!(uniform);
