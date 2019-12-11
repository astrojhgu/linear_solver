extern crate linear_solver;
use ndarray::{array};
use linear_solver::eigen::qr::{qr_naive};
use linear_solver::io::RawMM;
pub fn main(){
    let m=RawMM::from_file("A.mtx").to_array2();
    let v=qr_naive(m.view(), 500, 1e-12);
    for i in v{
        println!("{}", i);
    }
}
