use crate::params::constraints::{All, NonNegative};

param!(Count s.t. All<NonNegative> { All(NonNegative) });
