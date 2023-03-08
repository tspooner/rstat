use crate::params::{
    constraints::{self, Constraint},
    Param,
};

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Concentrations<const N: usize>(
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::utils::serde_arrays")
    )]
    pub [f64; N]
);

impl<const N: usize> Concentrations<N> {
    pub fn new(
        value: [f64; N]
    ) -> Result<Self, constraints::UnsatisfiedConstraintError<[f64; N]>>
    {
        constraints::All(constraints::Positive)
            .check(value)
            .and_then(|value| constraints::Not(constraints::Empty).check(value))
            .map(Concentrations)
    }
}

impl<const N: usize> Param for Concentrations<N> {
    type Value = [f64; N];

    fn value(&self) -> &[f64; N] { &self.0 }

    fn into_value(self) -> [f64; N] { self.0 }

    fn constraints() -> constraints::Constraints<Self::Value> {
        let c1 = constraints::Not(constraints::Empty);
        let c2 = constraints::All(constraints::Positive);

        vec![Box::new(c1), Box::new(c2)]
    }
}
