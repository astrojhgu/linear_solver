#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]

use ndarray::Array1;
use ndarray::ScalarOperand;
use num_traits::Float;

pub fn eculid_norm<T>(x: &Array1<T>) -> T
where
    T: Copy + Default + Float + ScalarOperand + 'static,
{
    (x.dot(x)).sqrt()
}

pub struct LsqrState<T>
where
    T: Float + Copy + Default + ScalarOperand,
{
    pub x: Array1<T>,
    pub alpha: T,
    pub u: Array1<T>,
    pub v: Array1<T>,
    pub w: Array1<T>,
    pub phi_bar: T,
    pub rho_bar: T,
}

pub fn sp_mul_a1<T, I, IptrStorage, IndStorage, DataStorage>(
    A: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage>,
    b: &Array1<T>,
) -> Array1<T>
where
    T: Float + Copy + Default + ScalarOperand,
    I: sprs::SpIndex + ndarray::NdIndex<ndarray::Dim<[usize; 1]>>,
    IptrStorage: std::ops::Deref<Target = [I]>,
    IndStorage: std::ops::Deref<Target = [I]>,
    DataStorage: std::ops::Deref<Target = [T]>,
{
    let mut result = Array1::from(vec![T::default(); A.rows()]);
    for (&x, (i, j)) in A.iter() {
        result[(i)] = result[(i)] + x * b[(j)];
    }
    result
}

pub fn lsqr_iter<T, IptrStorage, IndStorage, DataStorage>(A: &sprs::CsMatBase<T, usize, IptrStorage, IndStorage, DataStorage>, s_last: &LsqrState<T>) -> LsqrState<T>
where
    T: Float + Copy + Default + ScalarOperand,
    IptrStorage: std::ops::Deref<Target= [usize]>,
    IndStorage: std::ops::Deref<Target = [usize]>,
    DataStorage: std::ops::Deref<Target = [T]>,
{
    let rhs_beta = sp_mul_a1(A, &s_last.v) - (&s_last.u) * (s_last.alpha);
    let beta = eculid_norm(&rhs_beta);
    let u = rhs_beta / beta;
    let rhs_alpha = sp_mul_a1(&A.transpose_view(), &u) - (&s_last.v) * beta;
    let alpha = eculid_norm(&rhs_alpha);
    let v = &rhs_alpha / alpha;
    let rho = (s_last.rho_bar.powi(2) + beta.powi(2)).sqrt();
    let c = s_last.rho_bar / rho;
    let s = beta / rho;
    let theta = s * alpha;
    let rho_bar = -c * alpha;
    let phi = (s_last.phi_bar) * c;
    let phi_bar = (s_last.phi_bar) * s;
    let x = ((&s_last.w) * (phi / rho)) + (&s_last.x);

    let w = (&v) - &((&s_last.w) * (theta / rho));
    LsqrState {
        x,
        alpha,
        u,
        v,
        w,
        phi_bar,
        rho_bar,
    }
}

#[allow(non_snake_case)]
pub fn lsqr_init<T, IptrStorage, IndStorage, DataStorage >(A: &sprs::CsMatBase<T, usize, IptrStorage, IndStorage, DataStorage>, b: &Array1<T>) -> LsqrState<T>
where
    T: Float + Copy + Default + ScalarOperand,
    IptrStorage: std::ops::Deref<Target= [usize]>,
    IndStorage: std::ops::Deref<Target = [usize]>,
    DataStorage: std::ops::Deref<Target = [T]>,
{
    let x0 = Array1::from(vec![<T as Default>::default(); A.cols()]);
    let beta = eculid_norm(&b);
    let u = (b) / beta;
    let ATu = sp_mul_a1(&A.transpose_view(), &u);
    let alpha = eculid_norm(&ATu);
    let v = ATu / alpha;
    let w = v.clone();
    let phi_bar = beta;
    let rho_bar = alpha;
    LsqrState {
        x: x0,
        alpha,
        u,
        v,
        w,
        phi_bar,
        rho_bar,
    }
}

impl<T> LsqrState<T>
where
    T: Float + Copy + Default + ScalarOperand,
{
    pub fn new<IptrStorage, IndStorage, DataStorage>(A: &sprs::CsMatBase<T, usize, IptrStorage, IndStorage, DataStorage>, b: &Array1<T>) -> LsqrState<T> 
    where IptrStorage: std::ops::Deref<Target= [usize]>,
    IndStorage: std::ops::Deref<Target = [usize]>,
    DataStorage: std::ops::Deref<Target = [T]>,
    {
        lsqr_init(A, b)
    }

    pub fn next<IptrStorage, IndStorage, DataStorage>(&mut self, A: &sprs::CsMatBase<T, usize, IptrStorage, IndStorage, DataStorage>) 
    where IptrStorage: std::ops::Deref<Target= [usize]>,
    IndStorage: std::ops::Deref<Target = [usize]>,
    DataStorage: std::ops::Deref<Target = [T]>,

    {
        let ls = lsqr_iter(A, self);
        *self = ls;
    }

    pub fn calc_resid<IptrStorage, IndStorage, DataStorage>(&self, A: &sprs::CsMatBase<T, usize, IptrStorage, IndStorage, DataStorage>, b: &Array1<T>)->Array1<T>
    where IptrStorage: std::ops::Deref<Target= [usize]>,
    IndStorage: std::ops::Deref<Target = [usize]>,
    DataStorage: std::ops::Deref<Target = [T]>,
    {
        &sp_mul_a1(A, &self.x)-b
    }
}
