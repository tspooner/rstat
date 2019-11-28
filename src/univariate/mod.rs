pub use crate::statistics::UnivariateMoments as Moments;

pub mod discrete;
pub mod continuous;

mod uniform;
pub use self::uniform::*;

mod degenerate;
pub use self::degenerate::*;
