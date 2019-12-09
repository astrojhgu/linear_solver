#![allow(clippy::deref_addrof)]
#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;

use super::utils::{apply_plane_rotation, generate_plane_rotation, update};
use crate::utils::norm;

pub struct AGmresState<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    pub m: usize,
    pub m_max: usize,
    pub m_min: usize,
    pub m_step: usize,
    pub cf: T,
    pub tol: T,
    pub x: Array1<T>,
    pub b: Array1<T>,
    pub H: Array2<T>,
    //s: Array1<T>,
    pub cs: Array1<T>,
    pub sn: Array1<T>,
    //pub av: Array1<T>,
    pub beta: T,
    pub resid: T,
    pub r: Array1<T>,
    pub v: Vec<Array1<T>>,
    pub converged: bool,
}

pub fn agmres1<T>(
    ags: &mut AGmresState<T>,
    A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
    M: Option<&dyn Fn(ArrayView1<T>) -> Array1<T>>,
) where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    //println!("{:?}", ags.beta);
    if ags.beta == T::zero() {
        ags.converged = true;
        return;
    }
    ags.v[0] = &ags.r / ags.beta;
    let mut s = Array1::<T>::zeros(ags.m + 1);
    s[0] = ags.beta;
    let r1 = ags.beta;

    let mut i = 0;
    while i < ags.m {
        let av = A(ags.v[i].view());
        //let mut w=M.map_or(av,|m|{m(av.view())});
        let mut w=if let Some(M)=M{
            M(av.view())
        }else{
            av
        };
        
        for k in 0..=i {
            ags.H[(k, i)] = w.dot(&ags.v[k]);
            w = w - &ags.v[k] * ags.H[(k, i)];
        }

        ags.H[(i + 1, i)] = norm(w.view());
        ags.v[i + 1] = w / ags.H[(i + 1, i)];

        for k in 0..i {
            let (dx, dy) =
                apply_plane_rotation(ags.H[(k, i)], ags.H[(k + 1, i)], ags.cs[k], ags.sn[k]);
            ags.H[(k, i)] = dx;
            ags.H[(k + 1, i)] = dy;
        }

        let (cs1, sn1) = generate_plane_rotation(ags.H[(i, i)], ags.H[(i + 1, i)]);
        ags.cs[i] = cs1;
        ags.sn[i] = sn1;
        {
            let (dx, dy) =
                apply_plane_rotation(ags.H[(i, i)], ags.H[(i + 1, i)], ags.cs[i], ags.sn[i]);
            ags.H[(i, i)] = dx;
            ags.H[(i + 1, i)] = dy;
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
            update(&mut ags.x, i, &ags.H, &s, &ags.v[..]);
            //*tol = ags.resid.powi(2);
            ags.converged = true;
            return;
        }
        i += 1;
    }
    update(&mut ags.x, i - 1, &ags.H, &s, &ags.v[..]);
    //ags.r = ;
    let w = &ags.b - &A(ags.x.view());
    ags.r=if let Some(M)=M{
        M(w.view())
    }else{
        w
    };
    
    //ags.r = M(w.view());
    ags.beta = norm(ags.r.view());
    if ags.resid.powi(2) < ags.tol {
        ags.converged = true;
        return;
    }

    if ags.beta / r1 > ags.cf {
        if ags.m - ags.m_step > ags.m_min {
            ags.m -= ags.m_step;
        } else {
            ags.m = ags.m_max;
        }
    }
    ags.converged = false;
}

pub fn agmres<T>(
    A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
    x: ArrayView1<T>,
    b: ArrayView1<T>,
    M: Option<&dyn Fn(ArrayView1<T>) -> Array1<T>>,
    max_iter: usize,
    m_max: usize,
    m_min: usize,
    m_step: usize,
    cf: T,
    tol: T,
) -> AGmresState<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    let mut ags = AGmresState::create(b.len(), m_max, m_max, m_min, m_step, cf, tol);

    //let mut H=Array2::<T>::zeros((m+1, m));
    //let mut s=Array1::<T>::zeros(m+1);
    //let mut cs=Array1::<T>::zeros(m+1);
    //let mut sn=Array1::<T>::zeros(m+1);
    //let mut av=Array1::<T>::zeros(n);
    //let mut v:Vec<_>=(0..=m).map(|_|{Array1::<T>::zeros(n)}).collect();

    //let mut r1=T::zero();
    //println!("b={:?}", b);
    ags.init(A, x, b, M);

    //println!("w={:?}", w);
    if ags.resid.powi(2) <= ags.tol {
        ags.converged = true;
        return ags;
    }

    ags.m = ags.m_max;

    //let mut j=1;
    for _j in 0..max_iter {
        agmres1(&mut ags, A, M);
        if ags.converged {
            return ags;
        }
        //j+=1;
    }
    ags
}

impl<T> AGmresState<T>
where
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    pub fn create(
        problem_size: usize,
        m: usize,
        m_max: usize,
        m_min: usize,
        m_step: usize,
        cf: T,
        tol: T,
    ) -> AGmresState<T> {
        let m_max = if m_max > problem_size {
            problem_size
        } else {
            m_max
        };
        AGmresState {
            m,
            m_max,
            m_min,
            m_step,
            cf,
            tol,
            x: Array1::<T>::zeros(problem_size),
            b: Array1::<T>::zeros(problem_size),
            H: Array2::<T>::zeros((m + 1, m)),
            //s: Array1::<T>::zeros(m+1),
            cs: Array1::<T>::zeros(m + 1),
            sn: Array1::<T>::zeros(m + 1),
            beta: T::zero(),
            resid: T::zero(),
            r: Array1::<T>::zeros(problem_size),
            v: (0..=m).map(|_| Array1::<T>::zeros(problem_size)).collect(),
            converged: false,
        }
    }

    pub fn init(
        &mut self,
        A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        x: ArrayView1<T>,
        b: ArrayView1<T>,
        M: Option<&dyn Fn(ArrayView1<T>) -> Array1<T>>,
    ) {
        let w=M.map_or(b.to_owned(),|m|{m(b.view())});
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
        self.r= if let Some(M)=M{
            M(w.view())
        }else{
            w
        };
        
        self.beta = norm(self.r.view());
        self.resid = self.beta / normb;
        self.converged = false;
    }

    pub fn new(
        A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
        x: ArrayView1<T>,
        b: ArrayView1<T>,
        M: Option<&dyn Fn(ArrayView1<T>) -> Array1<T>>,
        m_max: usize,
        m_min: usize,
        m_step: usize,
        cf: T,
        tol: T,
    ) -> Self {
        let mut result = Self::create(b.len(), m_max, m_max, m_min, m_step, cf, tol);
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
        agmres1(self, A, M);
    }

    pub fn calc_resid(&self, lhs: &dyn Fn(ArrayView1<T>) -> Array1<T>, b: &Array1<T>) -> Array1<T> {
        b - &lhs(self.x.view())
    }
}
