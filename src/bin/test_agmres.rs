#![allow(unused_imports)]

extern crate linear_solver;
extern crate ndarray;
extern crate sprs;

use ndarray::{array, ArrayView1};
use linear_solver::agmres::agmres;
use linear_solver::utils::sp_mul_a1;
use linear_solver::io::RawMM;
use ndarray::Array1;

fn main() {
    let a=RawMM::<f64>::from_file("circuit_2.mtx").to_sparse();

    println!("{:?}", a.shape());
    let x0 = Array1::<f64>::from(vec![1.0; a.cols()]);
    let b=sp_mul_a1(&a, x0.view());
    println!("{:?}", b);

    let A=|x: ArrayView1<f64>|->Array1<f64>{
        //a.dot(&x.to_owned())
        sp_mul_a1(&a, x)
    };
    let mut x=Array1::<f64>::from(vec![10.0; a.cols()]);
    let M=|x: ArrayView1<f64>|->Array1<f64>{x.to_owned()};
    let mut tol=1e-20;
    let mut atol=1e-20;

    let r=agmres(&A, &mut x, b.view(), &M, 925, 10, 1, 1, 0.4, &mut tol, &mut atol);
    println!("r={}", r);
    println!("{:?}", x);

}
