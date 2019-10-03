#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]

use ndarray::Array1;
use ndarray::ScalarOperand;
use num_traits::Float;


pub struct BiCGStabState<T>
where    
T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug
{
    pub res: Array1<T>,
    pub res_prime: Array1<T>,
    pub p: Array1<T>,
    pub x: Array1<T>,
}

pub fn bicgstab_iter<T>(lhs: &dyn Fn(&Array1<T>)->Array1<T>, s_last: &BiCGStabState<T>, th: T) -> BiCGStabState<T>
    where T: Float + Copy + Default + ScalarOperand+ std::fmt::Debug,
{
    let ap=lhs(&s_last.p);

    let alpha=s_last.res.dot(&s_last.res_prime)/(ap.dot(&s_last.res_prime));
    let s=(&s_last.res)-&((&ap)*alpha);
    let a_s=lhs(&s);
    let a_s_norm=a_s.dot(&a_s);
    let w=if a_s_norm == T::zero(){
        T::one()
    }else{
        a_s.dot(&s)/a_s_norm
    };
    let x=&s_last.x+&((&s_last.p)*alpha)+&(&s*w);
    let res=&s-&(&a_s*w);
    let res_res_prime=s_last.res.dot(&s_last.res_prime);
    let beta=(alpha/w)*
    if res_res_prime==T::zero(){
        T::one()
    }else{
        res.dot(&s_last.res_prime)/res_res_prime
    };
    
    let p=&res+&(&(&s_last.p-&(&ap*w))*beta);
    let (p, res_prime)=if res.dot(&s_last.res_prime)<th.powi(2){
        (res.clone(),res.clone())
    }else{
        (p, s_last.res_prime.clone())
    };

    BiCGStabState{
        res, 
        res_prime, 
        p,
        x,
    }
}


impl<T> BiCGStabState<T>
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug,
{
    pub fn new(lhs: &dyn Fn(&Array1<T>)->Array1<T>, x: Array1<T>, b: Array1<T>, )->BiCGStabState<T>{
        assert!(x.len()==b.len());
        let res=&b-&lhs(&x);
        let res_prime=res.clone();
        let p=res.clone();
        BiCGStabState{
            res, 
            res_prime,
            p, 
            x
        }
    }

    pub fn next(&mut self, lhs: &dyn Fn(&Array1<T>)->Array1<T>, th: T)->std::option::Option<()>{
        let ns=bicgstab_iter(lhs, self, th);
        if ns.valid(){
            *self=ns;
            Option::Some(())
        }else{
            Option::None
        }
    }

    pub fn calc_resid(&self, lhs: &dyn Fn(&Array1<T>)->Array1<T>, b:&Array1<T>)->Array1<T>{
        b-&lhs(&self.x)
    }

    pub fn converged(&self, lhs: &dyn Fn(&Array1<T>)->Array1<T>, b:&Array1<T>, th: T)->bool{
        let res=self.calc_resid(lhs, b);
        res.dot(&res)<th*th
    }

    pub fn valid(&self)->bool{
        self.res.iter().all(|x|{x.is_finite()}) &&
        self.res_prime.iter().all(|x|{x.is_finite()}) &&
        self.p.iter().all(|x|{x.is_finite()}) && 
        self.x.iter().all(|x|{x.is_finite()})
    }
}
