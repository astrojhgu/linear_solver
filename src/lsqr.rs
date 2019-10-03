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

#[derive(Clone)]
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

pub fn lsqr_iter<T>(fl: &dyn Fn(&Array1<T>)->Array1<T>, fr: &dyn Fn(&Array1<T>)->Array1<T>, s_last: &LsqrState<T>) -> LsqrState<T>
where
    T: Float + Copy + Default + ScalarOperand+std::fmt::Debug,
{
    let rhs_beta = fl(&s_last.v) - (&s_last.u) * (s_last.alpha);
    let beta = eculid_norm(&rhs_beta);
    let u = rhs_beta / beta;
    //let rhs_alpha = sp_mul_a1(&A.transpose_view(), &u) - (&s_last.v) * beta;
    let rhs_alpha = fr(&u) - (&s_last.v) * beta;
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
pub fn lsqr_init<T>(fr: &dyn Fn(&Array1<T>)->Array1<T>, ncols: usize, b: &Array1<T>) -> LsqrState<T>
where
    T: Float + Copy + Default + ScalarOperand,
{
    //assert!(ncols<b.len());
    let x0 = Array1::from(vec![<T as Default>::default(); ncols]);
    let beta = eculid_norm(&b);
    let u = (b) / beta;
    let ATu = fr(&u);
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
    T: Float + Copy + Default + ScalarOperand+ std::fmt::Debug,
{
    pub fn new(fr: &dyn Fn(&Array1<T>)->Array1<T>,ncols: usize,  b: &Array1<T>) -> LsqrState<T> 
    {
        lsqr_init(fr, ncols,  b)
    }

    pub fn next(&mut self, fl: &dyn Fn(&Array1<T>)->Array1<T>, fr: &dyn Fn(&Array1<T>)->Array1<T>)->Option<()> 
    {
        let ns = lsqr_iter(fl, fr, self);
        println!("{}", ns.valid());
        if ns.valid(){
            *self = ns;
            Some(())
        }else{
            None
        }
    }

    pub fn calc_resid(&self, fl: &dyn Fn(&Array1<T>)->Array1<T>, b: &Array1<T>)->Array1<T>
    {
        b-&fl(&self.x)
    }

    pub fn valid(&self)->bool{
        self.x.iter().all(|x|{x.is_finite()})&&
        self.alpha.is_finite()&&
        self.u.iter().all(|x| x.is_finite()) &&
        self.v.iter().all(|x| x.is_finite()) &&
        self.w.iter().all(|x| x.is_finite()) &&
        self.phi_bar.is_finite()&&
        self.rho_bar.is_finite()
    }
}
