#![allow(clippy::many_single_char_names)]
#![allow(non_snake_case)]

use crate::utils::{householder_matrix, ComplexOrReal};
use ndarray::{s, Array2, ArrayView1, ArrayView2};
use num_traits::Float;

pub fn outer<T>(x: ArrayView1<T>, y: ArrayView1<T>) -> Array2<T>
where
    T: ComplexOrReal<T> + Float + std::fmt::Debug,
{
    let mut result = unsafe { Array2::uninitialized((x.len(), y.len())) };
    for i in 0..x.len() {
        for j in 0..y.len() {
            result[(i, j)] = x[i] * y[j]
        }
    }
    result
}

pub fn householder_reflection<T, U>(mat: ArrayView2<T>) -> (Array2<T>, Array2<T>)
where
    T: ComplexOrReal<U> + std::fmt::Debug,
    U: Float,
{
    let mut A = mat.to_owned();
    let m = A.nrows();
    let n = A.ncols();
    let mut Q = Array2::eye(m);
    for k in 1..=n {
        let x = A.slice(s![k - 1..m, k - 1]);
        let H = householder_matrix(x);
        let A1 = A.slice(s![k - 1..m, k - 1..n]).to_owned();
        A.slice_mut(s![k - 1..m, k - 1..n]).assign(&H.dot(&A1));
        let Q1 = Q.slice(s![..k - 1, k - 1..m]).to_owned();
        Q.slice_mut(s![..k - 1, k - 1..m]).assign(&Q1.dot(&H));
        let Q1 = Q.slice(s![k - 1..m, k - 1..m]).to_owned();
        Q.slice_mut(s![k - 1..m, k - 1..m]).assign(&Q1.dot(&H));
    }

    for i in 1..m {
        for j in 0..i {
            A[(i, j)] = T::zero();
        }
    }
    (Q, A)
}
