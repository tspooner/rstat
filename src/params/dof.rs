use num::{Unsigned, Integer};
use std::fmt::{Debug, Display};
use super::{Param, constraints::{Positive, Constraints, UnsatisfiedConstraintError}};

#[cfg_attr(any(feature = "test", feature = "serde"), derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct DOF<T: Integer + Unsigned>(pub T);

impl<T: Integer + Unsigned + Display> DOF<T> {
    pub fn new(dof: T) -> Result<Self, UnsatisfiedConstraintError<T>> {
        assert_constraint!(dof+)
            .map_err(|e| e.with_target("dof"))
            .map(|s| DOF(s))
    }
}

impl<T: Integer + Unsigned> Param for DOF<T> {
    type Value = T;

    fn value(&self) -> &T { &self.0 }

    fn name(&self) -> &str { "dof" }

    fn constraints(&self) -> Option<Constraints<T>> {
        Some(vec![Box::new(Positive)])
    }
}
