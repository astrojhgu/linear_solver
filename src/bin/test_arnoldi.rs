#![allow(non_snake_case)]
extern crate linear_solver;
use linear_solver::arnoldi::ArnoldiSpace;
use linear_solver::io::RawMM;
pub fn main() {
    let Am = RawMM::<f64>::from_file("A.mtx").to_array2();

    let A = |x: ndarray::ArrayView1<f64>| Am.dot(&x);

    let b = RawMM::<f64>::from_file("b.mtx").to_array1();
    let mut arn = ArnoldiSpace::new(b.view());
    //while let Ok(())=arn.iter(&A){
    //}
    for _i in 0..15 {
        arn.iter(&A).unwrap();
    }

    //println!("{:?}",arn.Q);
    //let H=arn.get_H();
    //let Q=arn.get_Q();
    //RawMM::from_array2(H.view()).to_file("H.mtx");
    //RawMM::from_array2(Q.view()).to_file("Q.mtx");

    //let h1=qr::qr_naive_iter(H.slice(s![0..15,..]), 100);
    //let h1=qr::qr_with_shift(H.slice(s![0..15,..]), 1e-15);
    //RawMM::from_array2(h1.view()).to_file("h1.mtx");
}
