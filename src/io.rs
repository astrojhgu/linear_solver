use ndarray::{Array1, Array2};
use num_complex::{Complex, Complex64};
use num_traits::Num;
use sprs::CsMat;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
pub enum MM {
    SparseReal(CsMat<f64>),
    DenseReal(Array2<f64>),
    SparseComplex(CsMat<Complex64>),
    DenseComplex(Array2<Complex64>),
}

#[derive(Debug, Clone, Copy)]
pub enum Storage {
    Dense,
    Sparse,
}

impl Storage {
    pub fn stringfy(self) -> String {
        match self {
            Storage::Dense => "array",
            Storage::Sparse => "coordinate",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Qualifier {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

impl Qualifier {
    pub fn expand_items<T>(self, e: &RawEntry<T>) -> Vec<RawEntry<T>>
    where
        T: Num + Copy + std::fmt::Debug,
    {
        if e.i == e.j {
            vec![*e]
        } else {
            match self {
                Qualifier::General => vec![*e],
                Qualifier::Symmetric => vec![*e, e.symm()],
                Qualifier::SkewSymmetric => vec![*e, e.skewsymm()],
                Qualifier::Hermitian => {
                    unimplemented!()
                    //vec![e.hermit()]
                }
            }
        }
    }

    pub fn stringfy(self) -> String {
        match self {
            Qualifier::General => "general",
            Qualifier::Symmetric => "symmetric",
            Qualifier::SkewSymmetric => "skewsymmetric",
            Qualifier::Hermitian => "hermitian",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Array,
    Coordinate,
}

#[derive(Debug, Clone, Copy)]
pub struct RawEntry<T>
where
    T: Num + Copy + std::fmt::Debug,
{
    pub i: usize,
    pub j: usize,
    pub value: T,
}

impl<T> RawEntry<T>
where
    T: Num + Copy + std::fmt::Debug,
{
    pub fn new(i: usize, j: usize, value: T) -> RawEntry<T> {
        RawEntry { i, j, value }
    }
    pub fn symm(&self) -> RawEntry<T> {
        RawEntry {
            i: self.j,
            j: self.i,
            value: self.value,
        }
    }

    pub fn skewsymm(&self) -> RawEntry<T> {
        RawEntry {
            i: self.j,
            j: self.i,
            value: T::zero() - self.value,
        }
    }
}

impl<T> RawEntry<Complex<T>>
where
    T: Num + std::ops::Neg<Output = T> + Copy + std::fmt::Debug,
{
    pub fn hermit(&self) -> RawEntry<Complex<T>> {
        RawEntry {
            i: self.j,
            j: self.i,
            value: self.value.conj(),
        }
    }
}

#[derive(Debug)]
pub struct RawMM<T>
where
    T: Num + Copy + std::fmt::Debug,
{
    pub height: usize,
    pub width: usize,
    pub storage: Storage,
    pub qual: Qualifier,
    pub entries: Vec<RawEntry<T>>,
}

pub trait Parseable {
    fn parse(s: &[String]) -> Self;
    fn stringfy(&self) -> Vec<String>;
    fn type_string() -> &'static str;
}

impl Parseable for f64 {
    fn parse(s: &[String]) -> f64 {
        s[0].parse::<f64>().unwrap()
    }

    fn stringfy(&self) -> Vec<String> {
        vec![std::format!("{:e}", self)]
    }

    fn type_string() -> &'static str {
        "real"
    }
}

impl Parseable for i32 {
    fn parse(s: &[String]) -> i32 {
        s[0].parse::<i32>().unwrap()
    }

    fn stringfy(&self) -> Vec<String> {
        vec![std::format!("{}", self)]
    }

    fn type_string() -> &'static str {
        "integer"
    }
}

impl Parseable for i64 {
    fn parse(s: &[String]) -> i64 {
        s[0].parse::<i64>().unwrap()
    }

    fn stringfy(&self) -> Vec<String> {
        vec![std::format!("{}", self)]
    }

    fn type_string() -> &'static str {
        "integer"
    }
}

impl Parseable for isize {
    fn parse(s: &[String]) -> isize {
        s[0].parse::<isize>().unwrap()
    }

    fn stringfy(&self) -> Vec<String> {
        vec![std::format!("{}", self)]
    }

    fn type_string() -> &'static str {
        "integer"
    }
}

impl Parseable for Complex64 {
    fn parse(s: &[String]) -> Complex64 {
        let r = s[0].parse::<f64>().unwrap();
        let i = if s.len() > 1 {
            s[1].parse::<f64>().unwrap()
        } else {
            0.0_f64
        };
        Complex64::new(r, i)
    }

    fn stringfy(&self) -> Vec<String> {
        vec![std::format!("{:e}", self.re), std::format!("{:e}", self.im)]
    }

    fn type_string() -> &'static str {
        "complex"
    }
}

impl<T> RawMM<T>
where
    T: Num + std::ops::Neg<Output = T> + Copy + std::fmt::Debug + Parseable,
{
    pub fn from_file(fname: &str) -> RawMM<T> {
        let file = File::open(fname).unwrap();
        let reader = BufReader::new(file);

        let mut ii = reader.lines();
        //parse 1st line
        let fst_line: Vec<_> = if let Some(line) = ii.next() {
            let xx = line.unwrap();
            xx.split_ascii_whitespace().map(|x| x.to_string()).collect()
        } else {
            panic!()
        };
        let mut ii = ii.filter(|l| {
            if let Ok(ref a) = l {
                if a.starts_with('%') {
                    return false;
                }
            } else {
                panic!("fdsfads")
            }
            true
        });

        assert!(fst_line[0] == "%%MatrixMarket");
        assert!(fst_line[1] == "matrix");
        let line_format = if fst_line[2] == "coordinate" {
            Format::Coordinate
        } else if fst_line[2] == "array" {
            Format::Array
        } else {
            panic!()
        };

        assert!(T::type_string() == fst_line[3]);

        let qual = if fst_line[4] == "general" {
            Qualifier::General
        } else if fst_line[4] == "symmetric" {
            Qualifier::Symmetric
        } else if fst_line[4] == "skew-symmetric" {
            Qualifier::SkewSymmetric
        } else if fst_line[4] == "hermitian" {
            Qualifier::Hermitian
        } else {
            panic!()
        };

        let snd_line: Vec<_> = if let Some(line) = ii.next() {
            let xx = line.unwrap();
            xx.split_ascii_whitespace()
                .map(|x| x.parse::<usize>().unwrap())
                .collect()
        } else {
            panic!()
        };

        //println!("{:?}", snd_line);

        //println!("{:?}", fst_line);

        let width = snd_line[1];
        let height = snd_line[0];
        let mut raw_data = RawMM {
            height,
            width,
            storage: match line_format {
                Format::Coordinate => Storage::Sparse,
                Format::Array => Storage::Dense,
            },
            qual,
            entries: vec![],
        };

        let mut col_num = 0;
        for (n, e) in ii.enumerate() {
            let line = e
                .unwrap()
                .split_ascii_whitespace()
                .map(|x| x.to_string())
                .collect::<Vec<_>>();
            let (i, j) = match line_format {
                Format::Array => {
                    match qual {
                        Qualifier::General => {
                            let j = n / height;
                            let i = n - j * height;
                            (i, j)
                        }
                        Qualifier::Hermitian | Qualifier::Symmetric | Qualifier::SkewSymmetric => {
                            //let nelements_before=(2*height-col_num+1)*col_num/2;
                            let nelements = (2 * height - col_num) * (col_num + 1) / 2;
                            if n >= nelements {
                                col_num += 1;
                            }

                            let nelements_before = (2 * height - col_num + 1) * col_num / 2;
                            let i = col_num;
                            let j = col_num + n - nelements_before;
                            //println!("{:?}", col_num);
                            (i, j)
                        }
                    }
                }
                Format::Coordinate => (
                    line[0].parse::<usize>().unwrap() - 1,
                    line[1].parse::<usize>().unwrap() - 1,
                ),
            };

            raw_data.entries.push(RawEntry {
                i,
                j,
                value: match line_format {
                    Format::Array => T::parse(&line[..]),
                    Format::Coordinate => T::parse(&line[2..]),
                },
            });
        }
        raw_data
    }

    pub fn to_file(&self, fname: &str) {
        let mut f = std::fs::File::create(fname).unwrap();
        writeln!(
            &mut f,
            "%%MatrixMarket matrix {} {} {}",
            self.storage.stringfy(),
            T::type_string(),
            self.qual.stringfy()
        )
        .unwrap();
        write!(&mut f, "{} {}", self.height, self.width).unwrap();
        if let Storage::Sparse = self.storage {
            writeln!(&mut f, " {}", self.entries.len()).unwrap();
        } else {
            writeln!(&mut f).unwrap();
        }
        let mut entries = self.entries.clone();
        entries[..].sort_by(|&a, &b| (a.j, a.i).cmp(&(b.j, b.i)));

        for RawEntry { i, j, value: v } in entries {
            if let Storage::Sparse = self.storage {
                write!(&mut f, "{} {}", i + 1, j + 1).unwrap();
            }
            for s in v.stringfy() {
                write!(&mut f, " {}", s).unwrap();
            }
            writeln!(&mut f).unwrap();
        }
    }

    pub fn to_sparse(&self) -> sprs::CsMat<T> {
        let mut entries: Vec<_> = self
            .entries
            .iter()
            .map(|x| self.qual.expand_items(x))
            .flatten()
            .collect();
        entries[..].sort_by(|&a, &b| (a.i, a.j).cmp(&(b.i, b.j)));

        let mut indptr = vec![0];
        let mut indices = vec![];
        let mut data = vec![];

        for RawEntry { i, j, value: v } in entries {
            while i + 2 != indptr.len() {
                indptr.push(*indptr.last().unwrap());
            }
            indices.push(j);
            data.push(v);
            //data.push(T::one());
            if let Some(x) = indptr.last_mut() {
                *x += 1;
            } else {
                panic!()
            }
        }

        while self.height+1!=indptr.len(){
            indptr.push(*indptr.last().unwrap());
        }

        assert!(self.height+1==indptr.len());

        sprs::CsMat::new((self.height, self.width), indptr, indices, data)
    }

    pub fn from_sparse(mat: &sprs::CsMat<T>) -> RawMM<T> {
        let entries: Vec<_> = mat
            .iter()
            .map(|(&v, (i, j))| RawEntry { i, j, value: v })
            .collect();

        RawMM {
            height: mat.rows(),
            width: mat.cols(),
            qual: Qualifier::General,
            storage: Storage::Sparse,
            entries,
        }
    }

    pub fn to_array1(&self) -> ndarray::Array1<T> {
        assert!(self.width == 1);
        let entries: Vec<_> = self
            .entries
            .iter()
            .map(|x| self.qual.expand_items(x))
            .flatten()
            .collect();

        let mut result = Array1::zeros(self.height);
        for RawEntry { i, value: v, .. } in entries {
            result[i] = v;
        }

        result
    }

    pub fn from_array1(data: ndarray::ArrayView1<T>) -> RawMM<T> {
        let entries: Vec<_> = (0..data.len())
            .map(|i| RawEntry {
                i,
                j: 0,
                value: data[i],
            })
            .collect();

        RawMM {
            height: data.len(),
            width: 1,
            qual: Qualifier::General,
            storage: Storage::Dense,
            entries,
        }
    }

    pub fn to_array2(&self) -> ndarray::Array2<T> {
        let entries: Vec<_> = self
            .entries
            .iter()
            .map(|x| self.qual.expand_items(x))
            .flatten()
            .collect();

        let mut result = Array2::zeros((self.height, self.width));
        for RawEntry { i, j, value: v } in entries {
            result[(i, j)] = v;
        }

        result
    }

    pub fn from_array2(data: ndarray::ArrayView2<T>) -> RawMM<T> {
        let mut entries = Vec::new();
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                //eprintln!("{} {} {:?}", i, j, data[(i, j)]);
                entries.push(RawEntry {
                    i,
                    j,
                    value: data[(i, j)],
                })
            }
        }

        RawMM {
            height: data.nrows(),
            width: data.ncols(),
            qual: Qualifier::General,
            storage: Storage::Dense,
            entries,
        }
    }
}
