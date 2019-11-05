#![allow(clippy::many_single_char_names)]
#![allow(non_snake_case)]

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::Float;

fn copy_sign<T>(x: T, y: T) -> T
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    y.signum() * x.abs()
}

fn norm<T>(x: ArrayView1<T>) -> T
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    x.iter()
        .map(|&x| x.powi(2))
        .fold(T::zero(), |x, y| x + y)
        .sqrt()
}

fn givens_rotation_matrix_entries<T>(a: T, b: T) -> (T, T)
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let r = a.hypot(b);
    let c = a / r;
    let s = -b / r;
    (c, s)
}

fn tril_indices(nrows: usize, ncols: usize) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for i in 1..nrows {
        for j in 0..i.min(ncols) {
            result.push((i, j));
        }
    }
    result
}

pub fn givens_rotation<T>(mat: ArrayView2<T>) -> (Array2<T>, Array2<T>)
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut Q = Array2::eye(nrows);
    let mut R = mat.to_owned();
    for (row, col) in tril_indices(nrows, ncols) {
        if R[(row, col)] != T::zero() {
            let (c, s) = givens_rotation_matrix_entries(R[(col, col)], R[(row, col)]);
            let mut G = Array2::eye(nrows);
            G[(col, col)] = c;
            G[(row, row)] = c;
            G[(row, col)] = s;
            G[(col, row)] = -s;
            R = G.dot(&R);
            Q = Q.dot(&G.t());
        }
    }
    (Q, R)
}

fn outer<T>(x: ArrayView1<T>, y: ArrayView1<T>) -> Array2<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let mut result = unsafe { Array2::uninitialized((x.len(), y.len())) };
    for i in 0..x.len() {
        for j in 0..y.len() {
            result[(i, j)] = x[i] * y[j]
        }
    }
    result
}

pub fn householder_reflection<T>(mat: ArrayView2<T>) -> (Array2<T>, Array2<T>)
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let nrows = mat.nrows();
    //let ncols = mat.ncols();
    let mut Q = Array2::eye(nrows);
    let mut R = mat.to_owned();
    let two = T::one() + T::one();
    for cnt in 0..(nrows - 1) {
        let x = R.slice(s![cnt.., cnt]).to_owned();
        let mut e = Array1::zeros(x.len());
        e[0] = copy_sign(norm(x.view()), -mat[(cnt, cnt)]);
        let u = &x + &e;
        let v = &u / norm(u.view());
        let mut Q_cnt = Array2::eye(nrows);
        let q = Q_cnt.slice(s![cnt.., cnt..]).to_owned();
        Q_cnt
            .slice_mut(s![cnt.., cnt..])
            .assign(&(&q - &(&outer(v.view(), v.view()) * two)));
        R = Q_cnt.dot(&R);
        Q = Q.dot(&Q_cnt.t());
    }
    (Q, R)
}

pub fn gram_schmidt_process<T>(mat: ArrayView2<T>) -> (Array2<T>, Array2<T>)
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut Q = unsafe { Array2::<T>::uninitialized((nrows, ncols)) };
    for (cnt, a) in mat.t().genrows().into_iter().enumerate() {
        let mut u = a.to_owned();
        for i in 0..cnt {
            let proj = &Q.column(i) * (Q.column(i).dot(&a));
            u = &u - &proj;
        }
        let e = &u / norm(u.view());
        Q.column_mut(cnt).assign(&e);
    }
    let R = Q.t().dot(&mat);
    (Q, R)
}
