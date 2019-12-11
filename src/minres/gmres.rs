#![allow(clippy::deref_addrof)]
#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;

use super::utils::{apply_plane_rotation, generate_plane_rotation, update2};
use crate::utils::norm;
use crate::arnoldi::{ArnoldiSpace, ArnoldiErr};
use crate::utils::HasConj;

pub struct GmresState<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug + HasConj,
{
    pub m: usize,
    pub tol: T,
    pub x: Array1<T>,
    pub b: Array1<T>,
    //pub H: Array2<T>,
    //s: Array1<T>,
    pub cs: Array1<T>,
    pub sn: Array1<T>,
    //pub av: Array1<T>,
    pub beta: T,
    pub resid: T,
    pub r: Array1<T>,
    //pub v: Vec<Array1<T>>,
    pub converged: bool,
    pub arn: ArnoldiSpace<T>,
}

pub fn gmres1<T>(
    ags: &mut GmresState<T>,
    A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
    M: Option<&dyn Fn(ArrayView1<T>) -> Array1<T>>,
) where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug + HasConj,
{
    //ags.v[0] = &ags.r / ags.beta;
    ags.arn.reset(ags.r.view());
    //println!("v={:?}", v);
    let mut s = Array1::<T>::zeros(ags.m + 1);
    s[0] = ags.beta;

    let mut i = 0;
    while i < ags.m {
        ags.arn.iter(&|x|{
            let av=A(x);
            if let Some(ref M)=M{
                M(av.view())
            }else{
                av
            }
        }).unwrap();

        for k in 0..i {
            let (dx, dy) =
                apply_plane_rotation(ags.arn.H[i][k], ags.arn.H[i][k+1], ags.cs[k], ags.sn[k]);
            //ags.H[(k, i)] = dx;
            //ags.H[(k + 1, i)] = dy;
            ags.arn.H[i][k]=dx;
            ags.arn.H[i][k+1]=dy;
        }

        //let (cs1, sn1) = generate_plane_rotation(ags.H[(i, i)], ags.H[(i + 1, i)]);
        let (cs1, sn1) = generate_plane_rotation(ags.arn.H[i][i], ags.arn.H[i][i+1]);
        ags.cs[i] = cs1;
        ags.sn[i] = sn1;
        {
            let (dx, dy) =
                //apply_plane_rotation(ags.H[(i, i)], ags.H[(i + 1, i)], ags.cs[i], ags.sn[i]);
                apply_plane_rotation(ags.arn.H[i][i], ags.arn.H[i][i+1], ags.cs[i], ags.sn[i]);
            ags.arn.H[i][i] = dx;
            ags.arn.H[i][i+1] = dy;
        }
        {
            let (dx, dy) = apply_plane_rotation(s[i], s[i + 1], ags.cs[i], ags.sn[i]);
            s[i] = dx;
            s[i + 1] = dy;
        }
        //println!("H={:?}", H);
        //println!("s={:?}", s);
        //println!("cn={:?}", cs);
        //println!("sn={:?}", sn);
        //std::process::exit(0);

        ags.resid = s[i + 1].abs();
        if ags.resid.powi(2) < ags.tol {
            //println!("resid={:?}, {:?}", resid, tol);
            update2(&mut ags.x, i, &ags.arn.H, &s, &ags.arn.Q[..]);
            //*tol = ags.resid.powi(2);
            ags.converged = true;
            return;
        }
        i += 1;
    }
    update2(&mut ags.x, i - 1, &ags.arn.H, &s, &ags.arn.Q[..]);
    //ags.r = ;
    let w = &ags.b - &A(ags.x.view());
    ags.r=if let Some(ref M)=M{
        M(w.view())
    }else{
        w
    };
    ags.beta = norm(ags.r.view());
    if ags.resid.powi(2) < ags.tol {
        ags.converged = true;
        return;
    }
    ags.converged = false;
}

impl<T> GmresState<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug + HasConj,
{
    pub fn create(problem_size: usize, m: usize, tol: T) -> GmresState<T> {
        GmresState {
            m,
            tol,
            x: Array1::<T>::zeros(problem_size),
            b: Array1::<T>::zeros(problem_size),
            //H: Array2::<T>::zeros((m + 1, m)),
            //s: Array1::<T>::zeros(m+1),
            cs: Array1::<T>::zeros(m + 1),
            sn: Array1::<T>::zeros(m + 1),
            //av: Array1::<T>::zeros(problem_size),
            beta: T::zero(),
            resid: T::zero(),
            r: Array1::<T>::zeros(problem_size),
            //v: (0..=m).map(|_| Array1::<T>::zeros(problem_size)).collect(),
            converged: false,
            arn: ArnoldiSpace::empty()
        }
    }

    pub fn init(
        &mut self,
        A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        x: ArrayView1<T>,
        b: ArrayView1<T>,
        M: &dyn Fn(ArrayView1<T>) -> Array1<T>,
    ) {
        let w = M(b.view());
        let normb = {
            let nb = norm(w.view());
            if nb == T::zero() {
                T::one()
            } else {
                nb
            }
        };
        self.b = b.to_owned();
        self.x = x.to_owned();
        self.r = A(x.view());
        let w = &b - &self.r;
        self.r = M(w.view());
        self.beta = norm(self.r.view());
        self.resid = self.beta / normb;
        self.converged = false;
        self.arn.reset(self.r.view());
    }

    pub fn new(
        A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        x: ArrayView1<T>,
        b: ArrayView1<T>,
        M: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        m: usize,
        tol: T,
    ) -> Self {
        let mut result = Self::create(b.len(), m, tol);
        result.init(A, x, b, M);
        result
    }

    pub fn next(
        &mut self,
        A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        M: Option<&dyn Fn(ArrayView1<T>) -> Array1<T>>,
    ) {
        if self.converged {
            return;
        }
        gmres1(self, A, M);
    }

    pub fn calc_resid(&self, lhs: &dyn Fn(ArrayView1<T>) -> Array1<T>, b: &Array1<T>) -> Array1<T> {
        b - &lhs(self.x.view())
    }
}
