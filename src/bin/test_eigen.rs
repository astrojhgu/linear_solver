extern crate linear_solver;
use ndarray::{array};
use linear_solver::eigen::qr::{hessenberg_reduction, qr_with_shift, qr_naive_iter};
use linear_solver::io::RawMM;
use linear_solver::qr::givens_rotation as qrdecomp;

pub fn main(){
    let m=RawMM::<f64>::from_file("A.mtx").to_array2();
    //let m=m.dot(&m.t());
    //let A=qr_with_shift(m.view(), 1e-25);
    let a=qr_naive_iter(m.view(), 100);
    RawMM::from_array2(a.view()).to_file("S.mtx");
    //let a=hessenberg_reduction(m.view());
    //RawMM::from_array2(a.view()).to_file("S.mtx");
}
