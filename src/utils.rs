#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use num_traits::{Float, Num};

pub fn sprs2dense<T>(s: &sprs::CsMat<T>) -> Array2<T>
where
    T: Copy + std::fmt::Debug + Num,
{
    let mut result = Array2::zeros((s.rows(), s.cols()));
    for (&x, (i, j)) in s.iter() {
        result[(i, j)] = x;
    }
    result
}

pub fn sp_mul_a1<T, I, IptrStorage, IndStorage, DataStorage>(
    A: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage>,
    b: ArrayView1<T>,
) -> Array1<T>
where
    T: Float + Copy + Default + ScalarOperand,
    I: sprs::SpIndex + ndarray::NdIndex<ndarray::Dim<[usize; 1]>>,
    IptrStorage: std::ops::Deref<Target = [I]>,
    IndStorage: std::ops::Deref<Target = [I]>,
    DataStorage: std::ops::Deref<Target = [T]>,
{
    let mut result = Array1::from(vec![T::default(); A.rows()]);
    for (&x, (i, j)) in A.iter() {
        result[(i)] = result[(i)] + x * b[(j)];
    }
    result
}
