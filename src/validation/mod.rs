use ndarray::Array2;
use std::{iter::Sum, result::Result as StdResult};

mod constraints;
pub use constraints::*;

pub type Result<T> = StdResult<T, UnsatisfiedConstraint>;

#[derive(Clone, Copy, Debug)]
pub struct Validator;

impl Validator {
    pub fn build<T>(self, f: impl Fn() -> T) -> T { f() }

    pub fn require(self, f: impl Fn() -> bool) -> Result<Self> {
        if f() { Ok(self) } else { Err(UnsatisfiedConstraint::Generic) }
    }

    // TODO: This should be generic over all types implementing PartialOrd.
    pub fn require_positive(self, x: f64) -> Result<Self> {
        if x < 0.0 {
            Err(UnsatisfiedConstraint::Numeric(NumericConstraint::Positive))
        } else {
            Ok(self)
        }
    }

    pub fn require_positive_real(self, x: f64) -> Result<Self> {
        if x <= 0.0 {
            Err(UnsatisfiedConstraint::Numeric(NumericConstraint::PositiveReal))
        } else {
            Ok(self)
        }
    }

    // TODO: This should be generic over all types implementing PartialOrd.
    pub fn require_negative(self, x: f64) -> Result<Self> {
        if x > 0.0 {
            Err(UnsatisfiedConstraint::Numeric(NumericConstraint::Negative))
        } else {
            Ok(self)
        }
    }

    pub fn require_negative_real(self, x: f64) -> Result<Self> {
        if x >= 0.0 {
            Err(UnsatisfiedConstraint::Numeric(NumericConstraint::NegativeReal))
        } else {
            Ok(self)
        }
    }

    // TODO: This should be generic over all integer types.
    pub fn require_natural(self, x: usize) -> Result<Self> {
        // Only need to test that it's not zero as usize cannot be negative.
        if x == 0 {
            Err(UnsatisfiedConstraint::Numeric(NumericConstraint::Natural))
        } else {
            Ok(self)
        }
    }

    pub fn require_lte<T: PartialOrd>(self, a: T, b: T) -> Result<Self> {
        if a <= b {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Numeric(NumericConstraint::LTE))
        }
    }

    pub fn require_gte<T: PartialOrd>(self, a: T, b: T) -> Result<Self> {
        if a >= b {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Numeric(NumericConstraint::GTE))
        }
    }
}

impl Validator {
    pub fn require_len<T>(self, tensor: &[T], len: usize) -> Result<Self> {
        if tensor.len() == len {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Tensor(TensorConstraint::Length(len)))
        }
    }

    pub fn require_min_len<T>(self, tensor: &[T], len: usize) -> Result<Self> {
        if tensor.len() >= len {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Tensor(TensorConstraint::MinLength(len)))
        }
    }

    pub fn require_ndim<T, D>(
        self,
        tensor: &ndarray::Array<T, D>,
        ndim: usize
    ) -> Result<Self>
    where
        D: ndarray::Dimension
    {
        if tensor.ndim() == ndim {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Tensor(TensorConstraint::NDimensional(ndim)))
        }
    }

    pub fn require_square<T>(self, tensor: &Array2<T>) -> Result<Self> {
        if tensor.is_square() {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Tensor(TensorConstraint::Square))
        }
    }

    pub fn require_sum<'a, T>(self, tensor: impl Iterator<Item = &'a T>, total: T)
        -> Result<Self>
    where
        T: 'a + Sum<&'a T> + PartialEq,
    {
        let sum: T = tensor.sum();

        if sum == total {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Tensor(TensorConstraint::SumsTo))
        }
    }

    pub fn require_normalised<'a>(self, tensor: impl Iterator<Item = &'a f64>) -> Result<Self> {
        let sum: f64 = tensor.sum();

        if (sum - 1.0f64).abs() < 1e-7 {
            Ok(self)
        } else {
            Err(UnsatisfiedConstraint::Tensor(TensorConstraint::Normalised))
        }
    }
}
