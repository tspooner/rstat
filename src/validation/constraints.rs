#[derive(Debug, Clone, Copy)]
pub enum NumericConstraint {
    Negative,
    NegativeReal,

    Positive,
    PositiveReal,

    Natural,

    LTE,
    GTE,
}

#[derive(Debug, Clone, Copy)]
pub enum TensorConstraint {
    Length(usize),
    MinLength(usize),

    NDimensional(usize),

    Square,

    SumsTo,
    Normalised,
}

#[derive(Debug, Clone, Copy)]
pub enum UnsatisfiedConstraint {
    Generic,
    Numeric(NumericConstraint),
    Tensor(TensorConstraint),
    Probability(crate::probability::ProbabilityError),
}

impl From<crate::probability::ProbabilityError> for UnsatisfiedConstraint {
    fn from(err: crate::probability::ProbabilityError) -> UnsatisfiedConstraint {
        UnsatisfiedConstraint::Probability(err)
    }
}
