///
pub struct Parameter<T> {
    symbol: String,
    constraints: Vec<Box<dyn Constraint<T>>>,

    _phantom: PhantomData<T>,
}

impl<T> Parameter<T> {
    pub fn new(symbol: String) -> Self {
        Self::with_constraints(symbol, Vec::default())
    }

    pub fn with_constraints(
        symbol: String,
        constraints: Vec<Box<dyn Constraint<T>>>,
    ) -> Self {
        Parameter {
            symbol,
            constraints,

            _phantom: PhantomData,
        }
    }

    pub fn verify<'a>(&self, value: &'a T) -> Result<&'a T, String>
    where
        T: fmt::Display,
        dyn Constraint<T>: fmt::Display,
    {
        for c in self.constraints.iter() {
            if !value.satisfies(c) {
                return Err(format!(
                    "Constraint {} on {} unsatisfied for value {}.",
                    c, self.symbol, value
                ))
            }
        }

        Ok(value)
    }
}

impl<T: fmt::Display> fmt::Display for Parameter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.symbol)
    }
}
