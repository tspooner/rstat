use crate::params::{
    constraints::{self, All, Constraint, Interval},
    Param,
};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Corr(pub f64);

impl Corr {
    pub fn new(value: f64) -> Result<Self, constraints::UnsatisfiedConstraintError<f64>> {
        Ok(Corr(All(Interval { lb: -1.0, ub: 1.0 }).check(value)?))
    }
}

impl Param for Corr {
    type Value = f64;

    fn value(&self) -> &f64 { &self.0 }

    fn into_value(self) -> f64 { self.0 }

    fn constraints() -> constraints::Constraints<Self::Value> {
        vec![Box::new(constraints::All(constraints::Interval {
            lb: -1.0,
            ub: 1.0,
        }))]
    }
}
