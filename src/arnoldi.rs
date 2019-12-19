#![allow(non_snake_case)]

use crate::utils::{norm, ComplexOrReal, HasAbs};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub enum ArnoldiErr {
    HMM1Zero,
    NoMoreBase,
}

impl std::fmt::Display for ArnoldiErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            ArnoldiErr::HMM1Zero => write!(f, "H(m, m+1)=0"),
            ArnoldiErr::NoMoreBase => write!(f, "No more base"),
        }
    }
}

impl std::error::Error for ArnoldiErr {}

#[derive(Clone)]
pub struct ArnoldiSpace<T, U>
where
    T: ComplexOrReal<U> + std::fmt::Debug,
    U: Float + std::fmt::Debug,
{
    pub Q: Vec<Array1<T>>,
    pub H: Vec<Array1<T>>,
    pub phantom: PhantomData<U>,
}

impl<T, U> ArnoldiSpace<T, U>
where
    T: ComplexOrReal<U> + std::fmt::Debug,
    U: Float + std::fmt::Debug,
{
    pub fn new(b: ArrayView1<T>) -> ArnoldiSpace<T, U> {
        let q = &b / <T as From<U>>::from(norm(b));
        ArnoldiSpace {
            Q: vec![q],
            H: Vec::new(),
            phantom: PhantomData,
        }
    }

    pub fn empty() -> ArnoldiSpace<T, U> {
        ArnoldiSpace {
            Q: Vec::new(),
            H: Vec::new(),
            phantom: PhantomData,
        }
    }

    pub fn reset(&mut self, b: ArrayView1<T>) {
        *self = Self::new(b)
    }

    pub fn iter(
        &mut self,
        A: &dyn Fn(ArrayView1<T>) -> Array1<T>,
    ) -> std::result::Result<(), ArnoldiErr> {
        let m = self.Q[0].len();
        if self.Q.len() >= m {
            return Err(ArnoldiErr::NoMoreBase);
        }
        //let mut v=self.A.dot(self.Q.last().unwrap());
        let mut v = A(self.Q.last().unwrap().view());
        let k = self.H.len();
        self.H.push(Array1::zeros(k + 2));
        //self.Q.push(Array1::zeros(self.q.len()));
        for j in 0..=k {
            self.H[k][j] = self.Q[j].map(|&x| x.conj()).dot(&v);
            v = &v - &(&self.Q[j] * self.H[k][j]);
        }
        self.H[k][k + 1] = <T as From<U>>::from(norm(v.view()));
        let q = v / self.H[k][k + 1];
        if q.iter().all(|x| HasAbs::abs(x).is_finite()) {
            self.Q.push(q);
            Result::Ok(())
        } else {
            Result::Err(ArnoldiErr::HMM1Zero)
        }
    }

    pub fn get_H(&self) -> Array2<T> {
        let n = self.H.len();
        let mut result = Array2::zeros([n, n]);
        for (i, h) in self.H.iter().enumerate() {
            //result.column_mut(i).slice(s![]).assign(h);
            let i1=if i == n -1{
                n - 1 
            }else{
                i + 1
            };
            result.slice_mut(s![..=i1, i]).assign(&h.slice(s![0..=i1]));
        }
        result
    }


    pub fn put_H(&mut self, H: ArrayView2<T>){
        let n = self.H.len();
        for (i, h) in self.H.iter_mut().enumerate() {
            let i1=if i == n -1{
                n - 1 
            }else{
                i + 1
            };
            h.slice_mut(s![0..=i1]).assign(&H.slice(s![..=i1, i]));
        }
    }


    pub fn get_H_star(&self) -> Array2<T> {
        let n = self.H.len();
        let mut result = Array2::zeros([n+1, n]);
        for (i, h) in self.H.iter().enumerate() {
            //result.column_mut(i).slice(s![]).assign(h);
            result.slice_mut(s![0..i+2, i]).assign(h);
        }
        result
    }

    pub fn get_Q(&self) -> Array2<T> {
        let n = self.H.len();
        let mut result = Array2::zeros([self.Q[0].len(), n ]);

        for (i, q) in self.Q.iter().take(n).enumerate() {
            result.slice_mut(s![.., i]).assign(q);
        }
        result
    }

    pub fn put_Q(&mut self, Q: ArrayView2<T>) {
        let n = self.H.len();
        for (i, q) in self.Q.iter_mut().take(n).enumerate() {
            //result.slice_mut(s![.., i]).assign(q);
            q.assign(&Q.slice(s![.., i]));
        }
    }

    pub fn get_Q_star(&self) -> Array2<T> {
        let n = self.H.len();
        let mut result = Array2::zeros([self.Q[0].len(), n + 1 ]);

        for (i, q) in self.Q.iter().enumerate() {
            result.slice_mut(s![.., i]).assign(q);
        }
        result
    }

    pub fn get_f(&self) -> Array1<T>{
        let n = self.H.len();
        let f = self.H[n-1][n];
        let v = self.Q.last().unwrap().clone();
        v*f
    }

    pub fn put_f(&mut self, f: ArrayView1<T>){
        let n=self.H.len();
        let nn=norm(f);
        let v=&f/T::from(nn);
        self.Q.last_mut().unwrap().assign(&v);
        self.H[n-1][n]=nn.into();
    }
}
