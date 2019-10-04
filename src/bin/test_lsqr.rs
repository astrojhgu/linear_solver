extern crate sparse_solver;
extern crate ndarray;
extern crate sprs;

use sparse_solver::lsqr::lsqr_init;
use sparse_solver::lsqr::sp_mul_a1;
use ndarray::Array1;

fn main() {
    let x0 = Array1::<f64>::from(vec![3., 1.]);
    println!("{:?}", x0);
    
    let a = sprs::CsMatI::<f64, usize>::new(
        (2, 2),
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, -1.0, 1.0],
    );

    let b = sp_mul_a1(&a, &x0.view());
    
    let mut aa = lsqr_init(&|x|{
        sp_mul_a1(&a.transpose_view(), &x)
    }, a.cols(), &b);

    for _i in 0..15 {
        if let Some(())=aa.next(
            &|x|{
                sp_mul_a1(&a, &x)
            },
            &|x|{
                sp_mul_a1(&a.transpose_view(), &x)
            }
        ){
            println!("a={:?}", aa.x);
            
        }else{
            println!("a={:?}", aa.x);
            break;
        }
    }
}
