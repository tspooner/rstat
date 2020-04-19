use ndarray::{s, Array1, Array2};

/// Type alias for 1-dimensional arrays.
pub type Vector<T> = Array1<T>;

/// Type alias for 2-dimensional arrays.
pub type Matrix<T> = Array2<T>;

/// Perform a Cholesky decomposition.
///
/// This function is unsafe because it assumes the given matrix is square,
/// symmetric, and positive-definite.
#[allow(non_snake_case)]
pub unsafe fn cholesky(A: &Matrix<f64>) -> Matrix<f64> {
    let n = A.nrows();
    let mut L = Matrix::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let s = (0..j).fold(0.0, |s, k| s + L[(i, k)] * L[(j, k)]);

            L[(i, j)] = if i == j {
                (A[(i, i)] - s).sqrt()
            } else {
                (A[(i, j)] - s) / L[(j, j)]
            };
        }
    }

    L
}

/// Compute the inverse of a nonsingular lower triangular matrix.
///
/// This function is unsafe because it does not check that the matrix is
/// triangular, lower triangular or nonsingular.
#[allow(non_snake_case)]
pub unsafe fn inverse_lt(L: &Matrix<f64>) -> Matrix<f64> {
    let n = L.nrows();
    let mut X = Matrix::zeros((n, n));

    for k in 0..n {
        X[(k, k)] = 1.0 / L[(k, k)];

        for i in (k + 1)..n {
            let z = L
                .slice(s!(i, k..=(i - 1)))
                .dot(&X.slice(s!(k..=(i - 1), k)));

            X[(i, k)] = -z / L[(i, i)];
        }
    }

    X
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_choleksy() {
        let m = arr2(&[
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ]);
        let lt = unsafe { cholesky(&m) };
        let expected = arr2(&[[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]);

        assert!((lt - expected).fold(0.0, |s, x| s + x.abs()) < 1e-5);
    }

    #[test]
    fn test_inverse_lt() {
        let m = arr2(&[
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ]);
        let lt = unsafe { cholesky(&m) };
        let lt_inv = unsafe { inverse_lt(&lt) };
        let expected = arr2(&[
            [1.0 / 2.0, 0.0, 0.0],
            [-3.0, 1.0, 0.0],
            [19.0 / 3.0, -5.0 / 3.0, 1.0 / 3.0],
        ]);

        assert!((lt_inv - expected).fold(0.0, |s, x| s + x.abs()) < 1e-5);
    }
}
