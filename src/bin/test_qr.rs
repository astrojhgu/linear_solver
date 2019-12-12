use linear_solver::qr::gram_schmidt_process;
use ndarray::array;

pub fn main() {
    let arr = array![[1.0, 2.0, 3.0], [3.0, 4.0, 3.0], [4.0, 3.0, 2.0]];
    let (q, r) = gram_schmidt_process(arr.view());
    println!("{:?}", q);
    println!("{:?}", r);
    println!("{:?}", q.dot(&q.t()));
    println!("{:?}", q.dot(&r));
}
