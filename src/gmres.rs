#![allow(clippy::deref_addrof)] 
#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]
use ndarray::{Array1, Array2, ArrayView2, ArrayView1, s};
use ndarray::ScalarOperand;
use num_traits::Float;

pub fn criteria<T>(v2: &Array1<T>, v1: &Array1<T>)->T
where 
T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug
{
    let w=v2-v1;
    w.dot(&w)/v2.dot(v2)
}

pub fn init<T>(A: &dyn Fn(&ArrayView1<T>)->Array1<T>, x: &ArrayView1<T>, b: &Array1<T>)->(T, Array1<T>)
where 
T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug{
    let r=b-&A(x);
    let beta=r.dot(&r).sqrt();
    let v1=&r/beta;
    (beta, v1)
}

pub fn solve_u<T>(U: &ArrayView2<T>, mut x: Array1<T>)->Array1<T>
where
T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug
{
    let n=x.len();
    for i in (0..n).rev(){
        for j in (i+1..n).rev(){
            x[i]=x[i]-U[(i,j)]*x[j];
        }
        x[i]=x[i]/U[(i,i)];
    }
    x
}

pub fn least_square<T>(Hess: &mut Array2<T>, beta: T)->Array1<T>
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug
{
    let m=Hess.ncols();
    let mut b=Array1::<T>::zeros(m+1);
    b[0]=beta;
    for i in 0..m{
        let t=(Hess[(i,i)].powi(2)+Hess[(i+1, i)].powi(2)).sqrt();
        let c=Hess[(i,i)]/t;
        let s=Hess[(i+1, i)]/t;
        let u=b[i];
        b[i]=c*u;
        b[i+1]=-s*u;
        for j in i..m{
            let (u,v)=(Hess[(i,j)], Hess[(i+1, j)]);
            Hess[(i,j)]= c*u+s*v;
            Hess[(i+1, j)]=-s*u+c*v;
        }
        Hess[(i+1, i)]=T::zero();
    }
    //println!("Hess1={:?}", Hess);
    //println!("b1={:?}", b);
    solve_u(&Hess.slice(s![0..m, 0..m]), b.slice(s![0..m]).to_owned())
}

pub fn hessenberg<T>(A: &dyn Fn(&ArrayView1<T>)->Array1<T>, v1: &Array1<T>, n: usize, m: usize)->(Array2<T>, Array2<T>)
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug,
{
    let mut Hess=Array2::<T>::zeros((m+2, m+1));
    let mut Vm=Array2::<T>::zeros((n, m+1));
    Vm.column_mut(0).assign(v1);
    for j in 0..=m{
        let mut w=A(&Vm.column(j));
        for i in 0..=j{
            Hess[(i,j)]=w.dot(&Vm.column(i));
            w=w-&Vm.column(i)*Hess[(i,j)];
        }
        Hess[(j+1, j)]=w.dot(&w).sqrt();
        if Hess[(j+1, j)]==T::zero() || j == m{
            //println!("j={} ss={:?}", j, Vm.slice(s![.., 0..=j]));
            return (Hess.slice(s![0..=j+1, 0..=j]).to_owned(), Vm.slice(s![.., 0..=j]).to_owned());
        }
        Vm.column_mut(j+1).assign(&(w/Hess[(j+1, j)]));
    }
    (Hess.slice(s![0..=m+1, 0..=m]).to_owned(), Vm.slice(s![.., 0..=m]).to_owned())
}

pub fn gmres<T>(A: &dyn Fn(&ArrayView1<T>)->Array1<T>, b: &Array1<T>, mut x: Array1<T>, restart: usize, tol: T, max_iter: usize)->Array1<T>
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug,
{
    let restart=if restart >b.len(){
        b.len()
    }else{
        restart
    };
    assert!(b.len()==x.len());
    let n=b.len();
    for _iter in 0..max_iter{
        for m in 0..restart{
            //println!("======iter={} m={}=======", iter, m);
            let (beta, v1)=init(A, &x.view(), b);
            //println!("beta={:?}", beta);
            //println!("v1={:?}", v1);
            let (mut Hess, Vm)=hessenberg(A, &v1, n, m);
            //println!("vm={:?}", Vm);
            //println!("Hess={:?}", Hess);
            let y = least_square(&mut Hess, beta);
            //println!("y={:?}", y);
            x=&x+&(Vm.dot(&y));
            let res=b-&A(&x.view());
            if res.dot(&res)<tol*tol{
                return x;
            }
        }
    }
    x
}
