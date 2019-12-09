use num_traits::Float;
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2,s};
use crate::utils::norm;
pub struct ArnoldiSpace<T>
where 
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,

{
    pub Q: Vec<Array1<T>>,
    pub H: Vec<Array1<T>>,
}

impl<T> ArnoldiSpace<T>
where 
    T: Copy + Default + Float + ScalarOperand + 'static + std::fmt::Debug,
{
    pub fn new(b: ArrayView1<T>)->ArnoldiSpace<T>{
        let q=&b/norm(b);
        ArnoldiSpace{
            Q: vec![q],
            H: Vec::new(),
        }
    }

    pub fn iter(&mut self, A: &dyn Fn(ArrayView1<T>) -> Array1<T>)->Option<()>{
        let m=self.Q[0].len();
        //let mut v=self.A.dot(self.Q.last().unwrap());
        let mut v=A(self.Q.last().unwrap().view());
        let k=self.H.len();
        self.H.push(Array1::zeros(k+2));
        //self.Q.push(Array1::zeros(self.q.len()));
        for j in 0..=k{
            self.H[k][j]=self.Q[j].dot(&v);
            v=&v-&(&self.Q[j]*self.H[k][j]);
        }
        self.H[k][k+1]=norm(v.view());
        let q=v/self.H[k][k+1];
        if q.iter().all(|x|x.is_finite()){
            self.Q.push(q);
            Some(())
        }else{
            None
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
