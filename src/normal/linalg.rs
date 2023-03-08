pub(crate) type Matrix<const N: usize, const M: usize> = [[f64; M]; N];

pub(crate) unsafe fn cholesky<const N: usize>(mut mat: Matrix<N, N>) -> Matrix<N, N> {
    let mut info = 0;

    lapack::dpotrf(
        'U' as u8, N as i32,
        &mut *mat.as_mut().as_mut_ptr(),
        N as i32, &mut info
    );

    for i in 0..N {
        for j in (i + 1)..N {
            mat[i][j] = 0.0;
        }
    }

    mat
}

pub(crate) unsafe fn inverse_lt<const N: usize>(mut lt: Matrix<N, N>) -> Matrix<N, N> {
    let mut info = 0;

    lapack::dtrtri(
        'U' as u8,
        'N' as u8,
        N as i32,
        &mut *lt.as_mut_ptr(),
        1i32.max(N as i32),
        &mut info
    );

    lt
}

pub(crate) unsafe fn mm_square<const N: usize>(mat: &Matrix<N, N>) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];

    blas::dgemm(
        b'N', b'T', N as i32, N as i32, N as i32,
        1.0,
        &*mat.as_ptr(), 1i32.max(N as i32),
        &*mat.as_ptr(), 1i32.max(N as i32),
        0.0,
        &mut *out.as_mut_ptr(), 1i32.max(N as i32)
    );

    out
}

pub(crate) unsafe fn mv_mult<const N: usize>(mat: &Matrix<N, N>, mut vec: [f64; N]) -> [f64; N] {
    blas::dtrmv(
        'U' as u8, 'T' as u8, 'N' as u8, N as i32,
        &*mat.as_ref().as_ptr(),
        // std::mem::transmute(mat),
        N as i32,
        vec.as_mut(),
        1
    );

    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate blas_src;

    #[test]
    fn test_cholesky() {
        let res = unsafe {
            cholesky([
                [25.0, 15.0, -5.0],
                [15.0, 18.0,  0.0],
                [-5.0, 0.0, 11.0]
            ])
        };

        assert_eq!(res[0][0], 5.0);
        assert_eq!(res[0][1], 0.0);
        assert_eq!(res[0][2], 0.0);

        assert_eq!(res[1][0], 3.0);
        assert_eq!(res[1][1], 3.0);
        assert_eq!(res[1][2], 0.0);

        assert_eq!(res[2][0], -1.0);
        assert_eq!(res[2][1], 1.0);
        assert_eq!(res[2][2], 3.0);
    }
}
