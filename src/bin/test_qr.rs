extern crate linear_solver;
use linear_solver::eigen::qr;
use linear_solver::io::RawMM;

pub fn main() {
    let m = RawMM::<f64>::from_file("mat.mtx").to_array2();

    let a = qr::hessenberg_reduction(m.view());

    let a = qr::qr_iter_with_shift(a.view(), 1e-15, 100).unwrap();
    //let v=qr::eigvals(a.view(), 1e-9, 1000);
    //let a=qr::qr_naive_iter(a.view(), 100);

    RawMM::from_array2(a.view()).to_file("mat1.mtx");
    //for i in v.unwrap(){
    //    println!("{} {}", i.re, i.im);
    //}
}
