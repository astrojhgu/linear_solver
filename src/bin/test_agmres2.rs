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

fn main() {
    let a = RawMM::<f64>::from_file("eye.mtx").to_array2();
    //let a = RawMM::<f64>::from_file("bcsstk23.mtx").to_sparse();

    println!("{:?}", a);
    let x0 = Array1::<f64>::from(vec![0.0; a.ncols()]);
    let b = a.dot(&x0);
    println!("{:?}", b);

    let A = |x: ArrayView1<f64>| -> Array1<f64> {
        //a.dot(&x.to_owned())
        a.dot(&x)
    };
    let x = Array1::<f64>::from(vec![1.0; a.ncols()]);
    let M = |x: ArrayView1<f64>| -> Array1<f64> { x.to_owned() };
    let tol = 1e-20;

    //let r=agmres(&A, &mut x, b.view(), &M, 925, 10, 1, 1, 0.4, &mut tol);
    let mut ags =
        AGmresState::<f64, f64>::new(&A, x.view(), b.view(), Some(&M), 30, 1, 1, 0.4, tol);

    //x.fill(100.);
    let mut cnt = 0;
    while !ags.converged {
        cnt += 1;
        if cnt % 10 == 0 {
            println!("{} {}", ags.tol, ags.resid);
        }
        ags.next(&A, Some(&M));
    }

    //println!("r={}", r);
    //println!("{:?}", x);
    for i in ags.x.iter() {
        println!("{}", i);
    }
}
