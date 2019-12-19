#![allow(non_snake_case)]
extern crate linear_solver;
use linear_solver::arnoldi::ArnoldiSpace;
use linear_solver::io::RawMM;
use num_complex::Complex;
pub fn main() {
    let Am = RawMM::<f64>::from_file("A.mtx").to_array2().map(|&x|{
        Complex::from(x)
    });

    let A = |x: ndarray::ArrayView1<Complex<f64>>| Am.dot(&x);

    let b = RawMM::<f64>::from_file("b.mtx").to_array1().map(|&x|{
        Complex::from(x)
    });
    let mut arn = ArnoldiSpace::new(b.view()).to_complex();
    //while let Ok(())=arn.iter(&A){
    //}
    for _i in 0..10 {
        arn.iter(&A).unwrap();
    }

    //println!("{:?}",arn.Q);
    let H_star=arn.get_H_star();
    let Q_star=arn.get_Q_star();
    let H=arn.get_H();
    let Q=arn.get_Q();
    
    let f=arn.get_f();
    RawMM::from_array2(H.view()).to_file("H.mtx");
    RawMM::from_array2(Q.view()).to_file("Q.mtx");

    RawMM::from_array2(H_star.view()).to_file("H_star.mtx");
    RawMM::from_array2(Q_star.view()).to_file("Q_star.mtx");
    
    RawMM::from_array1(f.view()).to_file("f.mtx");

    for i in 0..1500{
        arn.restart(&A, 10, 1e-6).unwrap();
        let H_star=arn.get_H_star();
        let Q_star=arn.get_Q_star();
        let H=arn.get_H();
        let Q=arn.get_Q();
        
        let f=arn.get_f();
        RawMM::from_array2(H.view()).to_file("H.mtx");
        RawMM::from_array2(Q.view()).to_file("Q.mtx");

        RawMM::from_array2(H_star.view()).to_file("H_star.mtx");
        RawMM::from_array2(Q_star.view()).to_file("Q_star.mtx");
        
        RawMM::from_array1(f.view()).to_file("f.mtx");
    }
    //let h1=qr::qr_naive_iter(H.slice(s![0..15,..]), 100);
    //let h1=qr::qr_with_shift(H.slice(s![0..15,..]), 1e-15);
    //RawMM::from_array2(h1.view()).to_file("h1.mtx");
}
