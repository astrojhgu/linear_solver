#![allow(clippy::deref_addrof)] 
#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]
use ndarray::{Array1, Array2, ArrayView2, ArrayView1, s};
use ndarray::ScalarOperand;
use num_traits::Float;


pub fn apply_plane_rotation<T> (mut dx: T, mut dy: T, cs: T,sn :T)->(T, T)
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug
{
   let temp  =  cs * dx + sn * dy;
   dy = -sn * dx + cs * dy;
   dx = temp;
   (dx, dy)
}

pub fn generate_plane_rotation<T> (dx:T, dy: T)->(T, T)
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug
{
    if dy == T::zero() {
        (T::one(), T::zero())
    } 
    else if dy.abs() > dx.abs() {
        let temp = dx / dy;
        let sn=T::one() / ( T::one() + temp*temp ).sqrt();
        let cs = temp * sn;
        (cs, sn)
    } else {
        let temp = dy / dx;
        let cs = T::one() / ( T::one() + temp*temp ).sqrt();
        let sn = temp * cs;
        (cs, sn)
    }
}

pub fn update<T>(x: &mut Array1<T>, k: usize, h: &Array2<T>, s: &Array1<T>, v: &[Array1<T>])
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug
{
    let mut y=s.to_owned();
    for i in (0..=k).rev(){
        y[i]=y[i]/h[(i,i)];
        if i>0{
            for j in (0..=i-1).rev(){
                y[j]=y[j]-h[(j,i)]*y[i];
            }
        }
    }

    for j in 0..=k{
        *x=&(*x)+&(&v[j]*y[j]);
    }
}

pub fn norm<T>(x: &ArrayView1<T>)->T
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug{
    x.dot(x).sqrt()
}

pub fn agmres<T>(A: &dyn Fn(ArrayView1<T>)->Array1<T>, x: &mut Array1<T>, b: ArrayView1<T>,M: &dyn Fn(ArrayView1<T>)->Array1<T>, max_iter: usize, m_max: usize, m_min: usize, m_step: usize, cf: T, tol: &mut T, atol: &mut T)->i32
where T: Copy + Default + Float + ScalarOperand + 'static+ std::fmt::Debug{
    let n = b.len();
    let mut m = m_max;
    let mut H=Array2::<T>::zeros((m+1, m));
    let mut s=Array1::<T>::zeros(m+1);
    let mut cs=Array1::<T>::zeros(m+1);
    let mut sn=Array1::<T>::zeros(m+1);
    let av=Array1::<T>::zeros(n);

    let mut r1=T::zero();
    //println!("b={:?}", b);
    let mut w=M(b.view());
    //println!("w={:?}", w);
    let normb={
        let nb=norm(&w.view());
        if nb==T::zero(){
            T::one()
        }else{
            nb
        }
    };
    //println!("normb={:?}", normb);
    let r=A(x.view());
    //println!("r={:?}", r);
    let w=&b-&r;
    //println!("w={:?}", w);
    let r=M(w.view());
    let beta=norm(&r.view());

    let resid=beta/normb;
    //println!("resid={:?}", resid);

    if resid.powi(2)<=*tol{
        *tol=resid.powi(2);
        return 0;
    }

    *tol=(*tol)*normb.powi(2);
    *tol=if *atol>*tol{
        *atol
    }else{
        *tol
    };

    m=m_max;
    let mut v:Vec<_>=(0..=m).map(|_|{Array1::<T>::zeros(n)}).collect();

    let mut j=1;
    while j<=max_iter{
        v[0]=(&r/beta);
        //println!("v={:?}", v);
        s.fill(T::zero());
        s[0]=beta;
        r1=beta;

        let mut i=0;
        while i< m && j<=max_iter{
            let av=A(v[i].view());
            let mut w=M(av.view());
            for k in 0..=i{
                H[(k, i)]=w.dot(&v[k]);
                w=(&w)-&(&v[k]*H[(k,i)]);
            }

            H[(i+1, i)]=norm(&w.view());
            v[i+1]=(&w)/H[(i+1, i)];

            for k in 0..i{
                let (dx, dy)=apply_plane_rotation(H[(k, i)], H[(k+1, i)], cs[k], sn[k]);
                H[(k, i)]=dx;
                H[(k+1, i)]=dy;
            }

            let (cs1, sn1)=generate_plane_rotation(H[(i, i)], H[(i+1, i)]);
            cs[i]=cs1;
            sn[i]=sn1;
            {
                let (dx, dy)=apply_plane_rotation(H[(i, i)], H[(i+1, i)], cs[i], sn[i]);
                H[(i, i)]=dx;
                H[(i+1, i)]=dy;
            }
            {
                let (dx, dy)=apply_plane_rotation(s[i], s[i+1], cs[i], sn[i]);
                s[i]=dx;
                s[i+1]=dy;
            }
            //println!("H={:?}", H);
            //println!("s={:?}", s);
            //println!("cn={:?}", cs);
            //println!("sn={:?}", sn);
            //std::process::exit(0);

            let resid=s[i+1].abs();
            if resid.powi(2)<*tol{
                //println!("resid={:?}, {:?}", resid, tol);
                update(x, i, &H, &s, &v[..]);
                *tol = resid.powi(2);
                return 0;               
            }
            i+=1;
        }
        update(x, i-1, &H, &s, &v[..]);
        let r=A(x.view());
        let w=&b-&r;
        let r=M(w.view());
        let beta=norm(&r.view());
        if resid.powi(2)<*tol{
            *tol=resid.powi(2);
            return 0;
        }

        if beta/r1 > cf{
            if m-m_step > m_min{
                m-=m_step;
            }else{
                m=m_max;
            }
        }

        j+=1;
    }
    *tol=resid.powi(2);
    return 1;
}
