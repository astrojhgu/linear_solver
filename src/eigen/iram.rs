use crate::eigen::qr::qr_naive_eigvals;
use crate::arnoldi::{ArnoldiSpace, ArnoldiErr};
use crate::utils::ComplexOrReal;

use crate::qr::householder_reflection;
use crate::utils::hermit;
use num_traits::Float;
use num_complex::Complex;
use ndarray::{ArrayView1, Array1, ArrayView2, Array2, ScalarOperand, s};
use std::marker::PhantomData;
impl<T, U> ArnoldiSpace<T, U>
where
    T: ComplexOrReal<U> + std::fmt::Debug,
    U: Float + std::fmt::Debug + Default,
    Complex<U>: From<T> + ScalarOperand
{
    pub fn to_complex(&self)->ArnoldiSpace<Complex<U>, U>{
        let H=self.H.iter().map(|arr|{
            arr.map(|&x| Complex::from(x))
        }).collect();
        let Q=self.Q.iter().map(|arr|{
            arr.map(|&x| Complex::from(x))
        }).collect();

        ArnoldiSpace{
            H,
            Q,
            phantom:PhantomData
        }
    }
}

impl<U> ArnoldiSpace<Complex<U>, U>
where
    U: Float + std::fmt::Debug + Default,
    Complex<U>: ScalarOperand,
{
    pub fn restart(&mut self,
    A: &dyn Fn(ArrayView1<Complex<U>>) -> Array1<Complex<U>>,
    p: usize, tol: U)->Result<(), ArnoldiErr>{
        let mut temp_arnoldi=self.clone();
        let k=self.H.len(); 
        for i in 0..p{
            temp_arnoldi.iter(A)?;
        }

        let m=k+p;
        assert!(m==temp_arnoldi.H.len());

        let mut Hm=temp_arnoldi.get_H();

        
        let mut H_eigs:Vec<Complex<U>>=qr_naive_eigvals(Hm.view(), tol);

        H_eigs.sort_by(|a,b|{
            if a.norm()<b.norm(){
                std::cmp::Ordering::Less
            }else if a.norm()>b.norm(){
                std::cmp::Ordering::Greater
            }else{
                std::cmp::Ordering::Equal
            }
        });

        let H_eigs=H_eigs;        
        
        let mut Q=Array2::eye(m);
        let I: Array2<Complex<U>>=Array2::eye(m);
        for &mu in &H_eigs[..p]{
        //for _ in 0..p{
            let shifted_H=&Hm-&(&I*Complex::from(mu.re));
            let (q,_)=householder_reflection(shifted_H.view());
            let q_star=hermit(q.view());
            Hm=q_star.dot(&Hm.dot(&q));
            Q=Q.dot(&q);
        }

        let beta_k=Hm[(k, k-1)];
        println!("{:?}", beta_k);
        let sigma_k=Q[(m-1, k-1)];
        let fm=temp_arnoldi.get_f();

        let fk=&temp_arnoldi.Q[k]*beta_k+fm*sigma_k;
        let Vm=temp_arnoldi.get_Q();
        let Vk=Vm.dot(&Q.slice(s![.., ..k]));
        
        let Hk=Hm.slice(s![..k, ..k]);
        
        self.put_f(fk.view());
        self.put_Q(Vk.view());
        self.put_H(Hk.view());
        Ok(())
    }
}
