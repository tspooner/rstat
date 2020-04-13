use num::{Unsigned, Integer};
use std::fmt::{Debug, Display};
use super::{Param, constraints::{NonNegative, Constraints, UnsatisfiedConstraintError}};

#[cfg_attr(any(feature = "test", feature = "serde"), derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct Count<T: Integer + Unsigned>(pub T);

impl<T: Integer + Unsigned + Display> Count<T> {
    pub fn new(count: T) -> Result<Self, UnsatisfiedConstraintError<T>> {
        assert_constraint!(count+)
            .map_err(|e| e.with_target("count"))
            .map(|s| Count(s))
    }
}

impl<T: Integer + Unsigned> Param for Count<T> {
    type Value = T;

    fn value(&self) -> &T { &self.0 }

    fn name(&self) -> &str { "count" }

    fn constraints(&self) -> Option<Constraints<T>> {
        Some(vec![Box::new(NonNegative)])
    }
}
