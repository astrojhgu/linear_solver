#![allow(clippy::many_single_char_names)]
#![allow(non_snake_case)]
use crate::qr::givens_rotation;
use crate::utils::Number;
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::Float;

pub fn back_sub<T>(u: ArrayView2<T>, b: ArrayView1<T>) -> Array1<T>
where
    T: Number<T> + Float + std::fmt::Debug,
{
    let mut x = Array1::zeros(b.len());

    let n = x.len();
    x[n - 1] = b[n - 1] / u[(n - 1, n - 1)];

    for i in (0..(n - 1)).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] = x[i] - u[(i, j)] * x[j];
        }
        x[i] = x[i] / u[(i, i)];
    }
    x
}
pub fn solve<T>(A: ArrayView2<T>, b: ArrayView1<T>) -> Array1<T>
where
    T: Number<T> + Float + std::fmt::Debug,
{
    let (q, r) = givens_rotation(A);

    let b1 = q.t().dot(&b);

    back_sub(r.view(), b1.view())
}
