extern crate linear_solver;
use linear_solver::eigen::qr::qr_naive_iter;
use linear_solver::io::RawMM;

pub fn main() {
    let m = RawMM::<f64>::from_file("A.mtx").to_array2();
    //let m=m.dot(&m.t());
    //let A=qr_with_shift(m.view(), 1e-25);
    let a = qr_naive_iter(m.view(), 100);
    RawMM::from_array2(a.view()).to_file("S.mtx");
    //let a=hessenberg_reduction(m.view());
    //RawMM::from_array2(a.view()).to_file("S.mtx");
}
