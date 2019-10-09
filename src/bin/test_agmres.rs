#![allow(unused_imports)]

extern crate linear_solver;
extern crate ndarray;
extern crate sprs;

use ndarray::{array, ArrayView1};
use linear_solver::agmres::agmres;
use linear_solver::lsqr::sp_mul_a1;
use ndarray::Array1;

fn main() {
    println!("Hello, world!");
    let x0 = Array1::<f64>::from(vec![2., 4., 5.]);
    let a=array![[4., 1., 5.], [1., 3., 4.], [3., 4. , 5.]];
    //let b = sp_mul_a1(&a, &x0);
    let b=a.dot(&x0);
    println!("{:?}", b);

    let A=|x: ArrayView1<f64>|->Array1<f64>{
        a.dot(&x.to_owned())
    };
    let mut x=Array1::<f64>::from(vec![1., 1., 1.]);
    let M=|x: ArrayView1<f64>|->Array1<f64>{x.to_owned()};
    let mut tol=1e-10;
    let mut atol=1e-12;

    let r=agmres(&A, &mut x, b.view(), &M, 100, 3, 1, 1, 0.4, &mut tol, &mut atol);
    println!("{}", r);

    println!("{:?}", x);

}
