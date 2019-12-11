#![allow(non_snake_case)]
use crate::qr::givens_rotation as qrdecomp;
use num_traits::Float;
use ndarray::{Array1, Array2, ArrayView2, ScalarOperand, s};
use num_complex::Complex;
use crate::utils::norm;

pub fn hessenberg_reduction<T>(A: ArrayView2<T>)->Array2<T>
where 
T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,{
    let mut n0=0;
    let n=A.nrows();
    let mut A=A.to_owned();
    let two=T::from(2).unwrap();
    for n0 in 0..n-2{
        let a1=A.slice(s![n0+1..,n0]).to_owned();
        let mut e1=Array1::zeros(n-n0-1);
        e1[0]=T::one();
        let sign=a1[0].signum();
        let v=&a1 + &(&e1 * (sign*norm(a1.view())));
        let v=(&v/norm(v.view())).into_shape((n-n0-1, 1)).unwrap();
        //let v=v;
        
        let Q1=Array2::<T>::eye(A.rows()-n0-1) - &(&(v.dot(&v.t()))*two);
        
        let x=A.slice(s![n0+1..n, n0]).to_owned();
        A.slice_mut(s![n0+1..n, n0]).assign(&Q1.dot(&x));
        
        let x=A.slice(s![n0, n0+1..n]).to_owned();
        A.slice_mut(s![n0, n0+1..n]).assign(&Q1.dot(&x));

        let a=A.slice(s![n0+1..n, n0+1..n]).to_owned();
        A.slice_mut(s![n0+1..n, n0+1..n]).assign(&Q1.dot(&(a.dot(&Q1.t()))));
    }
    A
}

pub fn wilkinson_shift<T>(a: T, b: T, c: T)->T
where 
T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,{
    let d=(a-c)/T::from(2).unwrap();
    c-d.signum()*b.powi(2)/(d.abs()+(d.powi(2)+b.powi(2)).sqrt())
}


pub fn qr_with_shift<T>(A: ArrayView2<T>, th: T)->Array2<T>
where 
T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,{
    let n=A.nrows();
    if n==1{
        return A.to_owned();
    }
    let th=th.abs();
    let I=Array2::eye(n);
    let mut A=hessenberg_reduction(A);
    while(A[(n-1, n-2)].abs()>th){
        println!("{:?}", A[(n-1, n-2)]);
        let mu=wilkinson_shift(A[(n-2, n-2)], A[(n-1,n-1)], A[(n-2, n-1)]);
        let (q,r)=qrdecomp((&A-&(&I*mu)).view());
        A=r.dot(&q)+&(&I*mu);
    }
    A
}


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

pub fn qr_naive_iter<T>(A:ArrayView2<T>, niter:usize)->Array2<T>
where T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,{
    let(mut q,mut r)=qrdecomp(A);
    let mut A=A.to_owned();
    for i in 0..niter{
        let (q,r)=qrdecomp(A.view());
        A=r.dot(&q);
    }
    A
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
