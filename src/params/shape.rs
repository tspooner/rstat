use num::{Zero, One};
use std::fmt::{Debug, Display};
use super::{Param, constraints::{Positive, Constraints, UnsatisfiedConstraintError}};

#[cfg_attr(any(feature = "test", feature = "serde"), derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct Shape<T: Zero + One + PartialOrd>(pub T);

impl<T: Zero + One + PartialOrd + Display> Shape<T> {
    pub fn new(shape: T) -> Result<Self, UnsatisfiedConstraintError<T>> {
        assert_constraint!(shape+)
            .map_err(|e| e.with_target("shape"))
            .map(|s| Shape(s))
    }
}

impl<T: Zero + One + PartialOrd> Param for Shape<T> {
    type Value = T;

    fn value(&self) -> &T { &self.0 }

    fn name(&self) -> &str { "shape" }

    fn constraints(&self) -> Option<Constraints<T>> {
        Some(vec![Box::new(Positive)])
    }
}
