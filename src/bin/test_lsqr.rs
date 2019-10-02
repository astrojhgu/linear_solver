extern crate sparse_solver;
extern crate ndarray;
extern crate sprs;

use sparse_solver::lsqr::lsqr_init;
use sparse_solver::lsqr::sp_mul_a1;
use ndarray::Array1;

fn main() {
    println!("Hello, world!");
    let x0 = Array1::<f64>::from(vec![2., 3.]);
    println!("{:?}", x0);
    
    let a = sprs::CsMatI::<f64, usize>::new(
        (3, 2),
        vec![0, 2, 4, 6],
        vec![0, 1, 0, 1, 0, 1],
        vec![1.0, 1.0, -1.0, 1.0, 0.5, 0.7],
    );

    let b = sp_mul_a1(&a, &x0);
    println!("{:?}", b);
    println!("{}", (x0.dot(&x0)).sqrt());

    let mut aa = lsqr_init(&|x|{
        sp_mul_a1(&a.transpose_view(), &x)
    }, a.cols(), &b);

    for _i in 0..5 {
        aa.next(
            &|x|{
                sp_mul_a1(&a, &x)
            },
            &|x|{
                sp_mul_a1(&a.transpose_view(), &x)
            }
        );
        println!("{:?}", aa.x);
    }
}
