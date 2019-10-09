extern crate linear_solver;
extern crate ndarray;
extern crate sprs;

use linear_solver::io::read_file;
use linear_solver::lsqr::lsqr_init;
use linear_solver::lsqr::sp_mul_a1;
use ndarray::Array1;

fn main() {
    read_file("mcca.mtx");
}
