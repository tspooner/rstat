pub use crate::statistics::MultivariateMoments as Moments;

// Continuous:
mod normal;
pub use self::normal::*;

mod normal_biv;
pub use self::normal_biv::*;

mod normal_diag;
pub use self::normal_diag::*;

mod lognormal;
pub use self::lognormal::*;

mod dirichlet;
pub use self::dirichlet::*;

// Discrete:
mod multinomial;
pub use self::multinomial::*;
