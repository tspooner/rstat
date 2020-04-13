use num::{Zero, One};
use std::fmt::{Debug, Display};
use super::{Param, constraints::{Positive, Constraints, UnsatisfiedConstraintError}};

#[cfg_attr(any(feature = "test", feature = "serde"), derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct Scale<T: Zero + One + PartialOrd>(pub T);

impl<T: Zero + One + PartialOrd + Display> Scale<T> {
    pub fn new(scale: T) -> Result<Self, UnsatisfiedConstraintError<T>> {
        assert_constraint!(scale+)
            .map_err(|e| e.with_target("scale"))
            .map(|s| Scale(s))
    }
}

impl<T: Zero + One + PartialOrd> Param for Scale<T> {
    type Value = T;

    fn value(&self) -> &T { &self.0 }

    fn name(&self) -> &str { "scale" }

    fn constraints(&self) -> Option<Constraints<T>> {
        Some(vec![Box::new(Positive)])
    }
}
