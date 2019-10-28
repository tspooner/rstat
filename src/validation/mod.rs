use ndarray::{Array1, Array2};
use std::{error::Error, fmt, iter::Sum, result::Result as StdResult};

#[derive(Debug, Clone, Copy)]
pub enum Numeric {
    Negative,
    NegativeReal,

    Positive,
    PositiveReal,

    Natural,

    LTE,
    GTE,
}

#[derive(Debug, Clone, Copy)]
pub enum Tensor {
    OfLength(usize),
    MinLength(usize),

    NDimensional(usize),

    Square,

    SumsTo,
    Normalised,
}

#[derive(Clone, Copy, Debug)]
pub enum ValidationError {
    Numeric(Numeric),
    Tensor(Tensor),
    Probability(crate::probability::ProbabilityError),
}

pub type Result<T> = StdResult<T, ValidationError>;

impl ValidationError {
    pub fn assert_positive(x: f64) -> Result<f64> {
        if x < 0.0 {
            Err(ValidationError::Numeric(Numeric::Positive))
        } else {
            Ok(x)
        }
    }

    pub fn assert_positive_real(x: f64) -> Result<f64> {
        if x <= 0.0 {
            Err(ValidationError::Numeric(Numeric::PositiveReal))
        } else {
            Ok(x)
        }
    }

    pub fn assert_negative(x: f64) -> Result<f64> {
        if x > 0.0 {
            Err(ValidationError::Numeric(Numeric::Negative))
        } else {
            Ok(x)
        }
    }

    pub fn assert_negative_real(x: f64) -> Result<f64> {
        if x > 0.0 {
            Err(ValidationError::Numeric(Numeric::NegativeReal))
        } else {
            Ok(x)
        }
    }

    pub fn assert_natural(x: usize) -> Result<usize> {
        if x == 0 {
            Err(ValidationError::Numeric(Numeric::Natural))
        } else {
            Ok(x)
        }
    }

    pub fn assert_lte<T: PartialOrd>(a: T, b: T) -> Result<(T, T)> {
        if a <= b {
            Ok((a, b))
        } else {
            Err(ValidationError::Numeric(Numeric::LTE))
        }
    }

    pub fn assert_gte<T: PartialOrd>(a: T, b: T) -> Result<(T, T)> {
        if a >= b {
            Ok((a, b))
        } else {
            Err(ValidationError::Numeric(Numeric::GTE))
        }
    }
}

impl ValidationError {
    pub fn assert_len<T>(tensor: &[T], len: usize) -> Result<&[T]> {
        if tensor.len() != len {
            Err(ValidationError::Tensor(Tensor::OfLength(len)))
        } else {
            Ok(tensor)
        }
    }

    pub fn assert_min_len<T>(tensor: &[T], len: usize) -> Result<&[T]> {
        if tensor.len() < len {
            Err(ValidationError::Tensor(Tensor::MinLength(len)))
        } else {
            Ok(tensor)
        }
    }

    pub fn assert_ndim<T, D>(
        tensor: &ndarray::Array<T, D>,
        ndim: usize
    ) -> Result<&ndarray::Array<T, D>>
    where
        D: ndarray::Dimension
    {
        if tensor.ndim() != ndim {
            Err(ValidationError::Tensor(Tensor::NDimensional(ndim)))
        } else {
            Ok(tensor)
        }
    }

    pub fn assert_square<T>(tensor: &Array2<T>) -> Result<&Array2<T>> {
        if tensor.is_square() {
            Ok(tensor)
        } else {
            Err(ValidationError::Tensor(Tensor::Square))
        }
    }

    pub fn assert_sum<'a, T>(tensor: impl Iterator<Item = &'a T>, total: T) -> Result<()>
    where
        T: 'a + Sum<&'a T> + PartialEq,
    {
        let sum: T = tensor.sum();

        if sum == total {
            Ok(())
        } else {
            Err(ValidationError::Tensor(Tensor::SumsTo))
        }
    }

    pub fn assert_normalised<'a>(tensor: impl Iterator<Item = &'a f64>) -> Result<()> {
        let sum: f64 = tensor.sum();

        if (sum - 1.0f64).abs() < 1e-7 {
            Ok(())
        } else {
            Err(ValidationError::Tensor(Tensor::Normalised))
        }
    }
}

impl From<crate::probability::ProbabilityError> for ValidationError {
    fn from(err: crate::probability::ProbabilityError) -> ValidationError {
        ValidationError::Probability(err)
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ValidationError::Numeric(_) => f.write_str("Numeric"),
            ValidationError::Tensor(_) => f.write_str("Tensor"),
            ValidationError::Probability(err) => err.fmt(f),
        }
    }
}

impl Error for ValidationError {
    fn description(&self) -> &str {
        match self {
            ValidationError::Numeric(_) => unimplemented!(),
            ValidationError::Tensor(_) => unimplemented!(),
            ValidationError::Probability(err) => err.description(),
        }
    }
}
