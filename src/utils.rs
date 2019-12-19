#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, Num};
use std::ops::Neg;
use num_complex::Complex;
pub trait ComplexOrReal<U: Float>:
    HasAbs<Output = U>
    + HasConj
    + ScalarOperand
    + Num
    + Neg<Output = Self>
    + From<U>
    + Copy
    + Clone
    + Default
    + Into<Complex<U>>
{
}

pub trait HasConj {
    fn conj(&self) -> Self;
}

pub trait HasAbs {
    type Output;
    fn abs(&self) -> Self::Output;
}

impl HasConj for f64 {
    fn conj(&self) -> f64 {
        *self
    }
}

impl HasConj for f32 {
    fn conj(&self) -> f32 {
        *self
    }
}

impl<T> HasConj for num_complex::Complex<T>
where
    T: num_traits::Float,
{
    fn conj(&self) -> num_complex::Complex<T> {
        num_complex::Complex::<T>::conj(self)
    }
}

impl HasAbs for f64 {
    type Output = f64;
    fn abs(&self) -> f64 {
        f64::abs(*self)
    }
}

impl HasAbs for f32 {
    type Output = f32;
    fn abs(&self) -> f32 {
        f32::abs(*self)
    }
}

impl<T> HasAbs for num_complex::Complex<T>
where
    T: num_traits::Float,
{
    type Output = T;
    fn abs(&self) -> T {
        num_complex::Complex::<T>::norm(self)
    }
}

impl<T> ComplexOrReal<T> for T where
    T: Float + HasAbs<Output = T> + HasConj + ScalarOperand + Default + Copy + Neg<Output = Self>
{
}

impl<U> ComplexOrReal<U> for num_complex::Complex<U>
where
    U: Float + Default,
    num_complex::Complex<U>: ScalarOperand,
{
}

pub fn norm<T, U>(x: ArrayView1<T>) -> U
where
    T: ComplexOrReal<U>,
    U: Float,
{
    //x.dot(&x).sqrt()
    let mut result = U::zero();
    for &x1 in x.iter() {
        result = result + x1.abs() * x1.abs();
    }
    result.sqrt()
}

pub fn sprs2dense<T>(s: &sprs::CsMat<T>) -> Array2<T>
where
    T: ComplexOrReal<T> + Float,
{
    let mut result = Array2::zeros((s.rows(), s.cols()));
    for (&x, (i, j)) in s.iter() {
        result[(i, j)] = x;
    }
    result
}

pub fn sp_mul_a1<U, T, I, IptrStorage, IndStorage, DataStorage>(
    A: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage>,
    b: ArrayView1<T>,
) -> Array1<T>
where
    T: ComplexOrReal<U>,
    U: Float,
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

pub fn sp_mul_a2<U, T, I, IptrStorage, IndStorage, DataStorage>(
    A: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage>,
    B: ArrayView2<T>,
) -> Array2<T>
where
    T: ComplexOrReal<U>,
    U: Float,
    I: sprs::SpIndex + ndarray::NdIndex<ndarray::Dim<[usize; 1]>> + Into<usize>,
    IptrStorage: std::ops::Deref<Target = [I]>,
    IndStorage: std::ops::Deref<Target = [I]>,
    DataStorage: std::ops::Deref<Target = [T]>,
{
    //let mut result = Array1::from(vec![T::default(); A.rows()]);
    let mut result = Array2::zeros((A.rows(), B.ncols()));

    for (&x, (i, k)) in A.iter() {
        for j in 0..B.ncols() {
            result[(i.into(), j)] = result[(i.into(), j)] + x * B[(k.into(), j)];
        }
    }
    result
}

pub fn csign<T,U>(x: T)->T
where T:ComplexOrReal<U>, 
U: Float{
    if x.abs()==U::zero(){
        T::one()
    }else{
        x/x.abs().into()
    }
}

pub fn hermit<T, U>(a: ArrayView2<T>)->Array2<T>
where T:ComplexOrReal<U>,
U: Float{
    a.map(|&x| x.conj()).t().to_owned()
}

pub fn get_e1<T, U>(n: usize)->Array1<T>
where T: ComplexOrReal<U>,
U: Float,{
    let mut e1=Array1::zeros(n);
    e1[0]=T::one();
    e1   
}

pub fn householder_matrix<T, U>(a: ArrayView1<T>)->Array2<T>
where T: ComplexOrReal<U>,
    U: Float,
{
    let two=T::one()+T::one();
    let n=a.len();
    let I=Array2::eye(n);
    let e1=get_e1(n);
    let norm_a:T=norm(a).into();
    let v=e1*norm_a*csign(a[0])+a;
    let v=&v/T::from(norm(v.view()));
    let v:Array2<_>=v.into_shape((n, 1)).unwrap();
    //let v_star=v.map(|&x|x.conj());
    let v_star=hermit(v.view());
    I-v.dot(&v_star)*two
}

