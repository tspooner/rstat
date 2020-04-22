use crate::{Distribution, Sample};

pub trait Mahalanobis: Distribution {
    /// Compute the Mahalanobis distance of sample \\(\\boldsymbol{x}\\).
    ///
    /// The Mahalanobis distance represents the distance between the sample
    /// \\(\\boldsymbol{x}\\) and the mean of the distribution,
    /// \\(\\boldsymbol{\\mu}\\): \\[\\sqrt{(\\boldsymbol{x} -
    /// \\boldsymbol{\\mu}){ \\Sigma^{-1} (\\boldsymbol{x} -
    /// \\boldsymbol{\\mu})}}.\\]
    fn d_mahalanobis(&self, x: &Sample<Self>) -> f64 { self.d_mahalanobis_squared(x).sqrt() }

    /// Compute the squared Mahalanobis distance of sample
    /// \\(\\boldsymbol{x}\\).
    fn d_mahalanobis_squared(&self, x: &Sample<Self>) -> f64 { self.d_mahalanobis(x).powi(2) }
}
