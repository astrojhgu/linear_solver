extern crate sparse_solver;
extern crate ndarray;
extern crate sprs;

use sparse_solver::lsqr::lsqr_init;
use sparse_solver::lsqr::sp_mul_a1;
use ndarray::Array1;

fn main() {
    println!("Hello, world!");
    let x0 = Array1::<f64>::from(vec![2., 3., 4., 4.]);
    println!("{:?}", x0);
    let a = sprs::CsMatI::<f64, usize>::new(
        (10, 4),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        vec![1.; 10],
    );
    let b = sp_mul_a1(&a, &x0);
    println!("{:?}", b);
    println!("{}", (x0.dot(&x0)).sqrt());

    let mut aa = lsqr_init(&a, &b);
    for _i in 0..5 {
        aa.next(&a);
        println!("{:?}", aa.x);
    }
}
