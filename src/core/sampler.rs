use core::Distribution;
use rand::Rng;
use spaces::Space;


pub struct Sampler<D, R> {
    pub(super) distribution: D,
    pub(super) rng: R,
}

impl<D, R> Iterator for Sampler<D, R>
    where D: Distribution,
          R: Rng,
{
    type Item = <D::Support as Space>::Value;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.distribution.sample(&mut self.rng))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::max_value(), None)
    }
}
