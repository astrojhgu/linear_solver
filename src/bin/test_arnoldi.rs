extern crate linear_solver;
use linear_solver::io::RawMM;
use linear_solver::arnoldi::ArnoldiSpace;
pub fn main(){
    let Am=RawMM::<f64>::from_file("A.mtx").to_array2();

    let A=|x: ndarray::ArrayView1<f64>| Am.dot(&x);

    let b=RawMM::<f64>::from_file("b.mtx").to_array1();
    let mut arn=ArnoldiSpace::new(b.view());
    //while let Ok(())=arn.iter(&A){
    //}
    for i in 0..15{
        arn.iter(&A).unwrap();
    }
    //println!("{:?}",arn.Q);
    let H=arn.get_H();
    let Q=arn.get_Q();
    RawMM::from_array2(H.view()).to_file("H.mtx");
    RawMM::from_array2(Q.view()).to_file("Q.mtx");
}