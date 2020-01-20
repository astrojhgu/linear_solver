extern crate linear_solver;
use linear_solver::eigen::qr::{hessenberg, implicit_shifted_qr, qr_naive_iter, qr_naive_eigvals};
use linear_solver::qr::householder_reflection;
use ndarray::{Array2, array, s};
use linear_solver::io::RawMM;
use std::fs::File;

use std::io::Write;
pub fn main(){
    let B=RawMM::<f64>::from_file("A.mtx").to_array2();

    let B=qr_naive_eigvals(B.view(), 1e-10);
    let mut outfile=File::create("./a.txt").unwrap();
    for i in B{
        writeln!(&mut outfile, "{} {}", i.re, i.im);
    }
}