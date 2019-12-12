#![allow(non_snake_case)]
#![allow(unused_imports)]

extern crate linear_solver;
extern crate ndarray;
extern crate sprs;

use linear_solver::io::RawMM;
use linear_solver::minres::agmres::agmres;
use linear_solver::minres::agmres::AGmresState;
use linear_solver::utils::sp_mul_a1;
use ndarray::Array1;
use ndarray::{array, ArrayView1};
use num_complex::Complex;
fn main() {
    let a = RawMM::<f64>::from_file("circuit_2.mtx").to_sparse();
    let a=a.map(|&x| 
    //Complex::<f64>::from(x)
    Complex::<f64>::new(x,x)
    );
    //let a = RawMM::<f64>::from_file("bcsstk23.mtx").to_sparse();

    println!("{:?}", a.shape());
    let x0 = Array1::<f64>::from(vec![1.0; a.cols()]).map(|&x| //Complex::<f64>::from(x)
    Complex::<f64>::new(x,-x)
    );
    let b = sp_mul_a1(&a, x0.view());
    println!("{:?}", b);

    let A = |x: ArrayView1<Complex<f64>>| -> Array1<Complex<f64>> {
        //a.dot(&x.to_owned())
        sp_mul_a1(&a, x)
    };
    let mut x = Array1::<Complex<f64>>::from(vec![Complex::from(10.0); a.cols()]);
    let M = |x: ArrayView1<Complex<f64>>| -> Array1<Complex<f64>> { x.to_owned() };
    let tol = 1e-20;

    //let r=agmres(&A, &mut x, b.view(), &M, 925, 10, 1, 1, 0.4, &mut tol);
    let mut ags = AGmresState::<Complex<f64>, f64>::new(&A, x.view(), b.view(), Some(&M), 30, 1, 1, 0.4, tol);

    x.fill(Complex::<f64>::new(100.,0.));
    let mut cnt = 0;
    while !ags.converged {
        cnt += 1;
        if cnt % 100 == 0 {
            println!("{}", ags.resid);
        }
        ags.next(&A, Some(&M));
    }

    //println!("r={}", r);
    //println!("{:?}", x);
    for i in ags.x.iter() {
        println!("{}", i);
    }
}
