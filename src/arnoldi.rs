#![allow(non_snake_case)]

use num_traits::Float;
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1, s};
use crate::utils::{norm, Number};

#[derive(Debug)]
pub enum ArnoldiErr{
    HMM1Zero,
    NoMoreBase,
}

impl std::fmt::Display for ArnoldiErr{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error>{
        write!(f, "H(m, m+1)=0")
    }
}


impl std::error::Error for ArnoldiErr{
}

pub struct ArnoldiSpace<T>
where 
    T: Number<T>+Float+std::fmt::Debug

{
    pub Q: Vec<Array1<T>>,
    pub H: Vec<Array1<T>>,
}

impl<T> ArnoldiSpace<T>
where 
    T: Number<T>+Float+std::fmt::Debug
{
    pub fn new(b: ArrayView1<T>)->ArnoldiSpace<T>{
        let q=&b/norm(b);
        ArnoldiSpace{
            Q: vec![q],
            H: Vec::new(),
        }
    }

    pub fn empty()->ArnoldiSpace<T>{
        ArnoldiSpace{
            Q: Vec::new(),
            H: Vec::new()
        }
    }

    pub fn reset(&mut self, b: ArrayView1<T>){
        *self=Self::new(b)
    }

    pub fn iter(&mut self, A: &dyn Fn(ArrayView1<T>) -> Array1<T>)->std::result::Result<(), ArnoldiErr>{
        let m=self.Q[0].len();
        if self.Q.len()>=m{
            return Err(ArnoldiErr::NoMoreBase);
        }
        //let mut v=self.A.dot(self.Q.last().unwrap());
        let mut v=A(self.Q.last().unwrap().view());
        let k=self.H.len();
        self.H.push(Array1::zeros(k+2));
        //self.Q.push(Array1::zeros(self.q.len()));
        for j in 0..=k{
            self.H[k][j]=self.Q[j].map(|&x|{x.conj()}).dot(&v);
            v=&v-&(&self.Q[j]*self.H[k][j]);
        }
        self.H[k][k+1]=norm(v.view());
        let q=v/self.H[k][k+1];
        if q.iter().all(|x|x.is_finite()){
            self.Q.push(q);
            Result::Ok(())
        }else{
            Result::Err(ArnoldiErr::HMM1Zero)
        }
    }

    pub fn get_H(&self)->Array2<T>{
        let n=self.H.len();
        let mut result=Array2::zeros([n+1, n]);
        for (i,h) in self.H.iter().enumerate(){
            //result.column_mut(i).slice(s![]).assign(h);
            result.slice_mut(s![0..i+2, i]).assign(h);
        }
        result
    }

    pub fn get_Q(&self)->Array2<T>{
        let n=self.H.len();
        let mut result=Array2::zeros([self.Q[0].len(),n+1]);

        for (i, q) in self.Q.iter().enumerate(){
            result.slice_mut(s![.., i]).assign(q);
        }

        result
    }
}
