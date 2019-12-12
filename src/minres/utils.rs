#![allow(clippy::deref_addrof)]
#![allow(non_snake_case)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2};
use num_traits::Float;
use crate::utils::Number;
pub fn apply_plane_rotation<T,U>(mut dx: T, mut dy: T, cs: T, sn: T) -> (T, T)
where
    T: Number<U>,
    U: Float
{
    let temp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
    (dx, dy)
}

pub fn generate_plane_rotation<T,U>(dx: T, dy: T) -> (T, T)
where
    T: Number<U>,
    U: Float
{
    if dy == T::zero() {
        (T::one(), T::zero())
    } else if dy.abs() > dx.abs() {
        let temp = dx / dy;
        let sn = T::one() / (T::one() + temp * temp).sqrt();
        let cs = temp * sn;
        (cs, sn)
    } else {
        let temp = dy / dx;
        let cs = T::one() / (T::one() + temp * temp).sqrt();
        let sn = temp * cs;
        (cs, sn)
    }
}

pub fn update<T,U>(x: &mut Array1<T>, k: usize, h: &Array2<T>, s: &Array1<T>, v: &[Array1<T>])
where
    T: Number<U>,
    U: Float
{
    let mut y = s.to_owned();
    for i in (0..=k).rev() {
        y[i] = y[i] / h[(i, i)];
        if i > 0 {
            for j in (0..i).rev() {
                y[j] = y[j] - h[(j, i)] * y[i];
            }
        }
    }

    for j in 0..=k {
        *x = &(*x) + &(&v[j] * y[j]);
    }
}

pub fn update2<T,U>(x: &mut Array1<T>, k: usize, h: &[Array1<T>], s: &Array1<T>, v: &[Array1<T>])
where
    T: Number<U>,
    U: Float
{
    let mut y = s.to_owned();
    for i in (0..=k).rev() {
        y[i] = y[i] / h[i][i];
        if i > 0 {
            for j in (0..i).rev() {
                y[j] = y[j] - h[i][j] * y[i];
            }
        }
    }

    for j in 0..=k {
        *x = &(*x) + &(&v[j] * y[j]);
    }
}

