#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, Num};
use std::ops::Neg;

pub trait Number<U:Float>:
    HasAbs<Output=U> + 
    HasConj + HasSqrt +
    ScalarOperand + 
    Num +
    Neg<Output=Self> +
    From<U> +
    Copy + 
    Clone +
    Default{
}

pub trait HasConj{
    fn conj(&self)->Self;
}

pub trait HasAbs{
    type Output;
    fn abs(&self)->Self::Output;
}

pub trait HasSqrt{
    fn sqrt(&self)->Self;
}

impl HasConj for f64{
    fn conj(&self)->f64{
        *self
    }
}

impl HasConj for f32{
    fn conj(&self)->f32{
        *self
    }
}




impl<T> HasConj for num_complex::Complex<T>
where T:num_traits::Float
{
    fn conj(&self)->num_complex::Complex<T>{
        num_complex::Complex::<T>::conj(self)
    }
}

impl HasAbs for f64{
    type Output=f64;
    fn abs(&self)->f64{
        f64::abs(*self)
    }
}

impl HasAbs for f32{
    type Output=f32;
    fn abs(&self)->f32{
        f32::abs(*self)
    }
}

impl<T> HasAbs for num_complex::Complex<T>
where T:num_traits::Float{
    type Output=T;
    fn abs(&self)->T{
        num_complex::Complex::<T>::norm(self)
    }
}

impl HasSqrt for f64{
    fn sqrt(&self)->f64{
        f64::sqrt(*self)
    }
}

impl HasSqrt for f32{
    fn sqrt(&self)->f32{
        f32::sqrt(*self)
    }
}

impl<T> HasSqrt for num_complex::Complex<T>
where T:num_traits::Float{
    fn sqrt(&self)->num_complex::Complex<T>{
        num_complex::Complex::<T>::sqrt(self)
    }
}

impl<T> Number<T> for T
where T:Float+HasAbs<Output=T>+HasConj+HasSqrt+ScalarOperand+Default+Copy+Neg<Output=Self>
{}

impl Number<f64> for num_complex::Complex<f64>{}
impl Number<f32> for num_complex::Complex<f32>{}


pub fn norm<T,U>(x: ArrayView1<T>) -> U
where
    T: Number<U>,
    U: Float
{
    //x.dot(&x).sqrt()
    let mut result=U::zero();
    for &x1 in x.iter(){
        result=result+x1.abs()*x1.abs();
    }
    result.sqrt()
}

pub fn sprs2dense<T>(s: &sprs::CsMat<T>) -> Array2<T>
where
    T: Number<T> + Float,
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
    T: Number<T>+Float,
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

pub fn sp_mul_a2<T, I, IptrStorage, IndStorage, DataStorage>(
    A: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage>,
    B: ArrayView2<T>,
) -> Array2<T>
where
    T: Number<T>+Float,
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
