#![allow(unused_imports)]
#![allow(unused_variables)]

extern crate linear_solver;
extern crate ndarray;
extern crate sprs;

use linear_solver::io::RawMM;
use linear_solver::lsqr::lsqr_init;
use linear_solver::utils::sp_mul_a1;
use linear_solver::utils::sprs2dense;
use ndarray::Array1;

fn main() {
    let mm = RawMM::<f64>::from_file("bcsstk01.mtx");
    //println!("{:?}", mm);
    let x = sprs2dense(&mm.to_sparse());
}
