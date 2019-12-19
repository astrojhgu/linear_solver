#![allow(clippy::many_single_char_names)]
#![allow(non_snake_case)]
//use crate::qr::householder_reflection as qrdecomp;
use crate::qr::householder_reflection as qrdecomp;
use crate::utils::ComplexOrReal;
use crate::utils::{hermit,  householder_matrix, get_e1};
use ndarray::{s, Array2, ArrayView2};
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

pub fn hessenberg<T, U>(A: ArrayView2<T>)->(Array2<T>, Array2<T>)
where T: ComplexOrReal<U> + std::fmt::Debug,
U: Float,
{
    let n=A.nrows();
    let mut A=A.to_owned();
    let mut Q=Array2::eye(n);
    for j in 1..=n-2{
        let U=householder_matrix(A.slice(s![j.., j-1]));
        let A1=A.slice(s![j.., j-1..]).to_owned();
        let U_star=hermit(U.view());
        A.slice_mut(s![j.., j-1..]).assign(&U_star.dot(&A1));
        let A1=A.slice(s![.., j..]).to_owned();
        A.slice_mut(s![.., j..]).assign(&A1.dot(&U));
        let Q1=Q.slice(s![.., j..]).to_owned();
        Q.slice_mut(s![.., j..]).assign(&Q1.dot(&U));
    }
    for j in 2..A.nrows() {
        A.slice_mut(s![j, ..j - 1]).fill(T::zero());
    }
    (A, Q)
}


pub fn implicit_shifted_qr<T, U>(A: ArrayView2<T>, shifts: &[T])->Array2<T>
where T: ComplexOrReal<U> + std::fmt::Debug,
U: Float,
{
    //algorithm 4 of HLA on 43-6
    let n=A.nrows();
    let k=shifts.len();
    let mut A=A.to_owned();
    let mut x=get_e1::<T, U>(n);
    let I=Array2::eye(n);
    
    for &mu in shifts{
        x=(&A-&(&I*mu)).dot(&x);
    }
    let U_star=householder_matrix(x.slice(s![..=k]));
    let U=hermit(U_star.view());
    let A1=A.slice(s![..=k, ..]).to_owned();
    A.slice_mut(s![..=k, ..]).assign(&U_star.dot(&A1));
    let A1=A.slice(s![.., ..=k]).to_owned();
    A.slice_mut(s![.., ..=k]).assign(&A1.dot(&U));
    hessenberg(A.view()).0
}

pub fn hessenberg_reduction<T>(A: ArrayView2<T>) -> Array2<T>
where
    T: Float + std::fmt::Debug,
{
    let n = A.nrows();
    let mut A = A.to_owned();

    for m in 1..n - 1 {
        let mut x = T::zero();
        let mut i = m;
        for j in m..n {
            if A[(j, m - 1)].abs() > x.abs() {
                x = A[(j, m - 1)];
                i = j;
            }
        }
        if i != m {
            for j in m - 1..n {
                A.swap((i, j), (m, j));
            }
            for j in 0..n {
                A.swap((j, i), (j, m));
            }
        }
        if x != T::zero() {
            for i in m + 1..n {
                let mut y = A[(i, m - 1)];
                if y != T::zero() {
                    y = y / x;
                    A[(i, m - 1)] = y;
                    for j in m..n {
                        A[(i, j)] = A[(i, j)] - y * A[(m, j)];
                    }
                    for j in 0..n {
                        A[(j, m)] = A[(j, m)] + y * A[(j, i)];
                    }
                }
            }
        }
    }

    for j in 2..A.nrows() {
        A.slice_mut(s![j, ..j - 1]).fill(T::zero());
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


pub fn eigv2x2<T,U>(A: ArrayView2<T>) -> (Complex<U>, Complex<U>)
where
    T: ComplexOrReal<U> + std::fmt::Debug,
    U: Float,
{
    let a:Complex<U>=A[(0, 0)].into();
    let b:Complex::<U>=A[(0, 1)].into();
    let c:Complex::<U>=A[(1, 0)].into();
    let d:Complex::<U>=A[(1, 1)].into();
    let t = a + d;
    let D = a * d - b * c;
    let ONE = T::one();
    let TWO = ONE + ONE;
    let FOUR = TWO + TWO;
    let E = (t.powi(2) / FOUR.into() - D).sqrt();
    let L1 = t / TWO.into() + E;
    let L2 = t / TWO.into() - E;
    (L1, L2)
}

pub fn qr_naive_iter<T,U>(A: ArrayView2<T>, niter: usize) -> Array2<T>
where
    T: ComplexOrReal<U> + std::fmt::Debug,
    U: Float,
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

pub fn wilkinson_shift1<U>(a: Complex<U>, b: Complex<U>, c: Complex<U>) -> Complex<U>
where
    U: Float,
{
    let two = U::one() + U::one();
    let d = (a - c) / two;
    c - b.powi(2) / (Complex::from(d.norm()) + (d.powi(2) + b.powi(2)).sqrt())
}

pub fn qr_naive_eigvals<T,U>(A:ArrayView2<T>, tol: U)->Vec<Complex<U>>
where T: ComplexOrReal<U> + std::fmt::Debug,
U: Float,{
    let n=A.nrows();
    //let mut A=A.to_owned();
    let (mut A, _)=hessenberg(A);

    loop{
        let mut splits=vec![0];
        for j in 0..n-1{
            if A[(j+1, j)].abs()<tol*(A[(j,j)].abs()+A[(j+1, j+1)].abs()){
                splits.push(j+1);
                A[(j+1, j)]=U::zero().into();
            }
        }
        if splits[splits.len()-1]!=n{
            splits.push(n);
        }

        let mut cnt=0;
        for b in splits.windows(2){
            let j1=b[0];
            let j2=b[1];
            if j2-j1==1 || j2-j1==2{
                continue;
            }
            //println!("{} {}", j1, j2);
            //
            //let s=A[(j2-1, j2-1)];
            let mut s=vec![A[(j2-1, j2-1)]];
            let A1=qr_naive_iter(A.slice(s![j1..j2, j1..j2]), 10);
            //let A1=implicit_shifted_qr(A.slice(s![j1..j2, j1..j2]), &s);
            A.slice_mut(s![j1..j2, j1..j2]).assign(&A1);
            cnt+=1;
        }
        if cnt==0{
            break;
        }
    }

    let mut splits=vec![0];
    for j in 0..n-1{
        if A[(j+1, j)].abs()<tol*(A[(j,j)].abs()+A[(j+1, j+1)].abs()){
            splits.push(j+1);
            A[(j+1, j)]=U::zero().into();
        }
    }
    if splits[splits.len()-1]!=n{
        splits.push(n);
    }
    let mut results=Vec::new();
    for b in splits.windows(2){
        let j1=b[0];
        let j2=b[1];
        if j2-j1==1{
            results.push(A[(j1,j1)].into());
        }else if j2-j1==2{
            let (L1,L2)=eigv2x2(A.slice(s![j1..j2, j1..j2]));
            results.push(L1);
            results.push(L2);
        }else{
            panic!("should never reach here");
        }
    }
    results
}
