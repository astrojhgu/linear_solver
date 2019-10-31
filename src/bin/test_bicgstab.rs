extern crate linear_solver;
extern crate ndarray;
extern crate sprs;

use linear_solver::bicgstab::BiCGStabState;
use linear_solver::utils::sp_mul_a1;
use ndarray::{Array1, ArrayView1};

fn main() {
    println!("Hello, world!");
    let x0 = Array1::<f64>::from(vec![2., 3.]);
    println!("{:?}", x0);
    let a = sprs::CsMatI::<f64, usize>::new(
        (2, 2),
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 1.0, 1.0, 3.0],
    );
    //let b = sp_mul_a1(&a, &x0);
    let b = Array1::<f64>::from(vec![1., 2.]);
    println!("{:?}", b);
    println!("{}", (x0.dot(&x0)).sqrt());

    let mut aa = BiCGStabState::new(
        &|x| sp_mul_a1(&a, x.view()),
        ArrayView1::from(&[1., 1.]),
        b.view(),
        1e-25,
    );

    while !aa.converged {
        let result = aa.next(&|x| sp_mul_a1(&a, x.view()));

        if result.is_none() {
            break;
        }

        println!("{:?}", aa.x);
    }
}
