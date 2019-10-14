#![allow(unused_imports)]

extern crate linear_solver;
extern crate ndarray;
extern crate sprs;

use ndarray::{array};
use linear_solver::gmres::gmres;
use linear_solver::utils::sp_mul_a1;
use ndarray::Array1;

fn main() {
    println!("Hello, world!");
    let x0 = Array1::<f64>::from(vec![2., 4., 5.]);
    let a=array![[4., 1., 5.], [1., 3., 4.], [3., 4. , 5.]];
    //let b = sp_mul_a1(&a, &x0);
    let b=a.dot(&x0);
    println!("{:?}", b);

    let x=gmres(&|x|{
        a.dot(&x.to_owned())
    }, &b, Array1::<f64>::from(vec![2., 2., 3.]), 2, 1e-8, 100);
    println!("{:?}", x);
}
