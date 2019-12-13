#![allow(clippy::many_single_char_names)]
#![allow(non_snake_case)]
//use crate::qr::householder_reflection as qrdecomp;
use crate::qr::householder_reflection as qrdecomp;
use crate::utils::norm;
use crate::utils::{ComplexOrReal, HasAbs, HasConj};
use ndarray::{s, Array1, Array2, ArrayView2, ScalarOperand};
use num_complex::Complex;
use num_traits::Float;

#[derive(Debug)]
pub enum QREignErr {
    NotConverged,
}

impl std::fmt::Display for QREignErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Not converged")
    }
}

impl std::error::Error for QREignErr {}


pub fn hessenberg_reduction<T>(A: ArrayView2<T>) -> Array2<T>
where
    T:Float + std::fmt::Debug,
{
    let n = A.nrows();
    let mut A = A.to_owned();
 
    for m in 1..n-1{
        let mut x=T::zero();
        let mut i=m;
        for j in m..n{
            if A[(j, m-1)].abs()>x.abs(){
                x=A[(j, m-1)];
                i=j;
            }
        }
        if i!=m{
            for j in m-1..n{
                A.swap((i,j),(m,j));
            }
            for j in 0..n{
                A.swap((j,i), (j,m));
            }
        }
        if x!=T::zero(){
            for i in m+1..n{
                let mut y=A[(i,m-1)];
                if y!=T::zero(){
                    y=y/x;
                    A[(i, m-1)]=y;
                    for j in m..n{
                        A[(i,j)]=A[(i,j)]-y*A[(m,j)];
                    }
                    for j in 0..n{
                        A[(j,m)]=A[(j,m)]+y*A[(j,i)];
                    }
                }
            }
        }
    }

    for j in 2..A.nrows(){
        A.slice_mut(s![j, ..j-1]).fill(T::zero());
    }
    A
}

pub fn wilkinson_shift<T>(a: T, b: T, c: T) -> T
where
    T: Float + std::fmt::Debug,
{
    let two = T::one() + T::one();
    let d = (a - c) / two;
    c - d.signum() * b.powi(2) / (d.abs() + (d.powi(2) + b.powi(2)).sqrt())
}

pub fn qr_iter_with_shift<T>(A: ArrayView2<T>, th: T, iter_max: usize) -> std::result::Result<Array2<T>, QREignErr>
where
    T: ComplexOrReal<T> + Float + std::fmt::Debug,
{
    let n = A.nrows();
    if n == 1 {
        return Ok(A.to_owned());
    }
    let th = th.abs();
    let I = Array2::eye(n);
    let mut A = hessenberg_reduction(A);
    //let mut A=A.to_owned();
    for i in 0..iter_max {
        eprintln!("{:?}", A[(n - 1, n - 2)]);
        let mu = wilkinson_shift(A[(n - 2, n - 2)], A[(n - 1, n - 1)], A[(n - 2, n - 1)]);
        //let mu=T::zero();
        let (q, r) = qrdecomp((&A - &(&I * mu)).view());
        A = r.dot(&q) + &(&I * mu);
        if A[(n - 1, n - 2)].abs() < th{
            return Ok(A)
        }
    }
    Err(QREignErr::NotConverged)
}

pub fn eigvals<T>(A:ArrayView2<T>, th: T, iter_max: usize)->Result<Vec<Complex<T>>, QREignErr>
where T: ComplexOrReal<T> + Float + std::fmt::Debug,
{
    if let Ok(a)=qr_iter_with_shift(A, th, iter_max){
        let mut i=0;
        let mut eig=Vec::new();
        let n=A.nrows();
        while i<n{
            if i==n-1 || a[(i+1, i)]<th{
                eig.push(Complex::from(a[(i,i)]));
                i+=1;
            }else{
                let (l1,l2)=eigv2x2(a.slice(s![i..i+2, i..i+2]));
                eig.push(l1);
                eig.push(l2);
                i+=2;
            }
        }
        if eig.len()<n{
            eig.push(Complex::from(a[(n-1, n-1)]));
        }
        Ok(eig)
    }else{
        Err(QREignErr::NotConverged)
    }
}

pub fn eigv2x2<T>(A: ArrayView2<T>) -> (Complex<T>, Complex<T>)
where
    T: ComplexOrReal<T> + Float + std::fmt::Debug,
{
    let a = A[(0, 0)];
    let b = A[(0, 1)];
    let c = A[(1, 0)];
    let d = A[(1, 1)];
    let T = a + d;
    let D = a * d - b * c;
    let ONE = T::one();
    let TWO = ONE + ONE;
    let FOUR = TWO + TWO;
    let E = Complex::from(T.powi(2) / FOUR - D).sqrt();
    let L1 = Complex::from(T / TWO) + E;
    let L2 = Complex::from(T / TWO) - E;
    (L1, L2)
}

pub fn qr_naive_iter<T>(A: ArrayView2<T>, niter: usize) -> Array2<T>
where
    T: ComplexOrReal<T> + Float + std::fmt::Debug,
{
    let mut A = A.to_owned();
    for _i in 0..niter {
        let (q, r) = qrdecomp(A.view());
        A = r.dot(&q);
    }
    A
}

pub fn qr_naive<T>(A: ArrayView2<T>, niter: usize, tol: T) -> Vec<Complex<T>>
where
    T: ComplexOrReal<T> + Float + std::fmt::Debug,
{
    let (mut q, mut r) = qrdecomp(A);
    for _i in 0..niter {
        let a = r.dot(&q);
        let b = qrdecomp(a.view());
        q = b.0;
        r = b.1;
    }
    let a = r.dot(&q);
    let n = A.nrows();
    let mut i = 0;
    let mut result = Vec::new();
    while i < n {
        eprintln!("{}", i);
        if i >= n - 1 || a[(i + 1, i)].abs() < tol {
            result.push(Complex::from(a[(i, i)]));
            i += 1;
        } else {
            let (l1, l2) = eigv2x2(a.slice(s![i..i + 2, i..i + 2]));
            result.push(l1);
            result.push(l2);
            i += 2;
        }
    }
    if result.len() < n {
        result.push(Complex::from(a[(n - 1, n - 1)]));
    }
    #[allow(clippy::comparison_chain)]
    result.sort_by(|&a, &b| {
        if a.norm() < b.norm() {
            std::cmp::Ordering::Greater
        } else if a.norm() > b.norm() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    });
    result
}
