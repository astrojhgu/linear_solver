extern crate sparse_solver;
extern crate ndarray;
extern crate sprs;

use sparse_solver::bicgstab::BiCGStabState;
use sparse_solver::lsqr::sp_mul_a1;
use ndarray::Array1;

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
    let b=Array1::<f64>::from(vec![1., 2.]);
    println!("{:?}", b);
    println!("{}", (x0.dot(&x0)).sqrt());

    let mut aa=BiCGStabState::new(&|x|{
        sp_mul_a1(&a, x)
    }, Array1::from(vec![1.,1.]), b.clone());

    while !aa.converged(&|x|{
        sp_mul_a1(&a, x)
    }, &b, 1e-25){

    if let Some(_)=aa.next(&|x|{
        sp_mul_a1(&a, x)
    }, 1e-10){
        
    }else{
        break;
    }

    println!("{:?}", aa.x);
    }
}
