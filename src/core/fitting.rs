use crate::core::Distribution;
use spaces::{Space, Vector};


pub trait MLE: Distribution {
    fn fit_mle(samples: Vector<<Self::Support as Space>::Value>) -> Self;
}
