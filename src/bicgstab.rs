#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]

use ndarray::ScalarOperand;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;

pub struct BiCGStabState<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    pub res: Array1<T>,
    pub res_prime: Array1<T>,
    pub p: Array1<T>,
    pub x: Array1<T>,
    pub tol: T,
    pub converged: bool,
}

pub fn bicgstab_iter<T>(
    lhs: &dyn Fn(ArrayView1<T>) -> Array1<T>,
    s_last: &BiCGStabState<T>,
) -> BiCGStabState<T>
where
    T: Float + Copy + Default + ScalarOperand + std::fmt::Debug,
{
    let ap = lhs(s_last.p.view());

    let alpha = s_last.res.dot(&s_last.res_prime) / (ap.dot(&s_last.res_prime));
    let s = (&s_last.res) - &((&ap) * alpha);
    let a_s = lhs(s.view());
    let a_s_norm = a_s.dot(&a_s);
    let w = if a_s_norm == T::zero() {
        T::one()
    } else {
        a_s.dot(&s) / a_s_norm
    };
    let x = &s_last.x + &((&s_last.p) * alpha) + &(&s * w);
    let res = &s - &(&a_s * w);
    let res_res_prime = s_last.res.dot(&s_last.res_prime);
    let beta = (alpha / w)
        * if res_res_prime == T::zero() {
            T::one()
        } else {
            res.dot(&s_last.res_prime) / res_res_prime
        };

    let p = &res + &(&(&s_last.p - &(&ap * w)) * beta);
    let (p, res_prime, converged) = if res.dot(&s_last.res_prime).abs() < s_last.tol.powi(2) {
        (res.clone(), res.clone(), true)
    } else {
        (p, s_last.res_prime.clone(), false)
    };

    BiCGStabState {
        res,
        res_prime,
        p,
        x,
        tol: s_last.tol,
        converged,
    }
}

impl<T> BiCGStabState<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    pub fn new(
        lhs: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        x: ArrayView1<T>,
        b: ArrayView1<T>,
        tol: T,
    ) -> BiCGStabState<T> {
        assert!(x.len() == b.len());
        let res = &b - &lhs(x.view());
        let res_prime = res.clone();
        let p = res.clone();
        BiCGStabState {
            res,
            res_prime,
            p,
            x: x.to_owned(),
            tol,
            converged: false,
        }
    }

    pub fn next(&mut self, lhs: &dyn Fn(ArrayView1<T>) -> Array1<T>) -> std::option::Option<()> {
        let ns = bicgstab_iter(lhs, self);
        if ns.valid() {
            *self = ns;
            Option::Some(())
        } else {
            Option::None
        }
    }

    pub fn calc_resid(&self, lhs: &dyn Fn(ArrayView1<T>) -> Array1<T>, b: &Array1<T>) -> Array1<T> {
        b - &lhs(self.x.view())
    }

    pub fn converged(
        &self,
        lhs: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        b: &Array1<T>,
        th: T,
    ) -> bool {
        let res = self.calc_resid(lhs, b);
        res.dot(&res) < th * th
    }

    pub fn valid(&self) -> bool {
        self.res.iter().all(|x| x.is_finite())
            && self.res_prime.iter().all(|x| x.is_finite())
            && self.p.iter().all(|x| x.is_finite())
            && self.x.iter().all(|x| x.is_finite())
    }
}
