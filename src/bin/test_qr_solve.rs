extern crate linear_solver;
use linear_solver::qr_solve::solve;
use ndarray::array;

pub fn main() {
    let arr = array![[1.0, 2.0, 3.0], [3.0, 4.0, 3.0], [4.0, 3.0, 2.0]];
    let b = array![1., 2., 3.];

    let x = solve(arr.view(), b.view());
    println!("{}", x);
}
