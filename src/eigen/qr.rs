#![allow(non_snake_case)]
use crate::qr::givens_rotation as qrdecomp;
use num_traits::Float;
use ndarray::{ArrayView2, ScalarOperand, s};
use num_complex::Complex;

pub fn eigv2x2<T>(A:ArrayView2<T>)->(Complex<T>, Complex<T>)
where 
   T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let a=A[(0,0)];
    let b=A[(0,1)];
    let c=A[(1,0)];
    let d=A[(1,1)];
    let T=a+d;
    let D=a*d-b*c;
    let ONE=T::one();
    let TWO=ONE+ONE;
    let FOUR=TWO+TWO;
    let E=Complex::from(T.powi(2)/FOUR-D).sqrt();
    let L1=Complex::from(T/TWO)+E;
    let L2=Complex::from(T/TWO)-E;
    (L1, L2)
}

pub fn qr_naive<T>(A:ArrayView2<T>, niter:usize, tol: T)->Vec<Complex<T>>
where T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let(mut q,mut r)=qrdecomp(A);
    for i in 0..niter{
        let a=r.dot(&q);
        let b=qrdecomp(a.view());
        q=b.0;
        r=b.1;
    }
    let a=r.dot(&q);
    let n=A.nrows();
    let mut i=0;
    let mut result=Vec::new();
    while i<n{
        println!("{}", i);
        if i>=n-1{
            result.push(Complex::from(a[(i,i)]));
            i+=1;
        }
        else if a[(i+1, i)].abs() < tol {
            result.push(Complex::from(a[(i,i)]));
            i+=1;
        }else
        {
            let (l1, l2)=eigv2x2(a.slice(s![i..i+2, i..i+2]));
            result.push(l1);
            result.push(l2);
            i+=2;
        }
    }
    if result.len()<n{
        result.push(Complex::from(a[(n-1, n-1)]));
    }
    result.sort_by(|&a, &b|{
        if a.norm()<b.norm(){
            std::cmp::Ordering::Greater
        }else if a.norm()>b.norm(){
            std::cmp::Ordering::Less
        }else{
            std::cmp::Ordering::Equal
        }
    });
    result
}
