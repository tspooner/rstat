pub use crate::statistics::MultivariateMoments as Moments;

// Continuous:
mod normal;
pub use self::normal::*;

mod lognormal;
pub use self::lognormal::*;

mod dirichlet;
pub use self::dirichlet::*;

// Discrete:
mod multinomial;
pub use self::multinomial::*;
