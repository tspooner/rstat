use core::distribution::Distribution;
use std::{error::Error, fmt};


#[derive(Debug, Clone)]
pub enum ConvolutionError {
    TooFewVariables,
    MixedParameters,
}

impl ConvolutionError {
    #[inline(always)]
    pub fn check_count<T>(rvs: &Vec<T>) -> ConvolutionResult<usize> {
        let n = rvs.len();

        if n < 2 {
            Err(ConvolutionError::TooFewVariables)
        } else {
            Ok(n)
        }
    }
}

impl fmt::Display for ConvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ConvolutionError::TooFewVariables => f.write_str("TooFewVariables"),
            ConvolutionError::MixedParameters => f.write_str("MixedParameters "),
        }
    }
}


impl Error for ConvolutionError {
    fn description(&self) -> &str {
        match *self {
            ConvolutionError::TooFewVariables =>
                "Convolution requires two or more independent random variables",
            ConvolutionError::MixedParameters  =>
                "Variables must all have the same parameters",
        }
    }
}

pub type ConvolutionResult<D> = Result<D, ConvolutionError>;

pub trait Convolution<T: Distribution = Self>
    where Self: Sized
{
    fn convolve(self, rv: T) -> ConvolutionResult<Self>;
    fn convolve_pair(x: T, y: T) -> ConvolutionResult<Self>;

    fn convolve_many(mut rvs: Vec<T>) -> ConvolutionResult<Self> {
        let _ = ConvolutionError::check_count(&rvs)?;
        let rv1 = rvs.pop().unwrap();
        let rv2 = rvs.pop().unwrap();

        let new_dist = Self::convolve_pair(rv1, rv2);

        rvs.into_iter().fold(new_dist, |acc, rv| {
            acc.and_then(|d| d.convolve(rv))
        })
    }
}
