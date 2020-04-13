use num::{Zero, One};
use std::fmt::{Debug, Display};
use super::{Param, constraints::{Positive, Constraints, UnsatisfiedConstraintError}};

#[cfg_attr(any(feature = "test", feature = "serde"), derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct Rate<T: Zero + One + PartialOrd>(pub T);

impl<T: Zero + One + PartialOrd + Display> Rate<T> {
    pub fn new(rate: T) -> Result<Self, UnsatisfiedConstraintError<T>> {
        assert_constraint!(rate+)
            .map_err(|e| e.with_target("rate"))
            .map(|s| Rate(s))
    }
}

impl<T: Zero + One + PartialOrd> Param for Rate<T> {
    type Value = T;

    fn value(&self) -> &T { &self.0 }

    fn name(&self) -> &str { "rate" }

    fn constraints(&self) -> Option<Constraints<T>> {
        Some(vec![Box::new(Positive)])
    }
}
