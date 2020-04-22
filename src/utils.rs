use crate::consts;

pub fn factorial_exact(n: u64) -> u64 { (1..=n).product() }

pub fn factorial(n: u64) -> u64 {
    if n > 11 {
        factorial_exact(n)
    } else {
        consts::FACTORIALS_16[n as usize]
    }
}

pub fn log_factorial_stirling(n: u64) -> f64 {
    if n > 254 {
        let x = (n + 1) as f64;

        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * consts::PI).ln() + 1.0 / (12.0 * x)
    } else {
        consts::LOG_FACTORIALS_255[n as usize]
    }
}

pub fn choose(n: u64, k: u64) -> u64 {
    let k = if k > n - k { n - k } else { k };

    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}
