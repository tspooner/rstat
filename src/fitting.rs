use crate::Distribution;
use spaces::Space;

pub trait MLE: Distribution {
    fn fit_mle(samples: Vec<<Self::Support as Space>::Value>) -> Self;
}
