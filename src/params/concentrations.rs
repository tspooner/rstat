use crate::{
    linalg::Vector,
    params::{
        constraints::{self, Constraint},
        Param,
    },
};

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Concentrations(pub Vector<f64>);

impl Concentrations {
    pub fn new(
        value: Vector<f64>,
    ) -> Result<Self, constraints::UnsatisfiedConstraintError<Vector<f64>>> {
        constraints::All(constraints::Positive)
            .check(value)
            .and_then(|value| constraints::Not(constraints::Empty).check(value))
            .map(Concentrations)
    }
}

impl Param for Concentrations {
    type Value = Vector<f64>;

    fn value(&self) -> &Vector<f64> { &self.0 }

    fn into_value(self) -> Vector<f64> { self.0 }

    fn constraints() -> constraints::Constraints<Self::Value> {
        let c1 = constraints::Not(constraints::Empty);
        let c2 = constraints::All(constraints::Positive);

        vec![Box::new(c1), Box::new(c2)]
    }
}
