use std::fs::File;
use std::io::{BufRead, BufReader};
use ndarray::Array2;
use sprs::CsMat;
use num_complex::Complex64;

pub enum MM{
    SparseReal(CsMat<f64>),
    DenseReal(Array2<f64>),
    SparseComplex(CsMat<Complex64>),
    DenseComplex(Array2<Complex64>),
}


#[derive(Debug)]
pub enum Qualifier{
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

#[derive(Debug)]
pub enum Format{
    Array,
    Coordinate,
}

#[derive(Debug)]
pub enum MatrixValue{
    Real(f64),
    Complex(Complex64),
}

#[derive(Debug)]
pub struct RawEntry{
    i: usize,
    j: usize,
    value: MatrixValue,
}

#[derive(Debug)]
pub struct LineProcessor{
    qual: Qualifier, 
    fmt: Format,
    entry: RawEntry,
}

pub fn read_file(fname: &str){
    let file = File::open(fname).unwrap();
    let reader = BufReader::new(file);

    let mut ii=reader.lines();
    //parse 1st line
    let fst_line:Vec<_>=if let Some(line)=ii.next(){
        let xx=line.unwrap();
        xx.split_ascii_whitespace().map(|x|{x.to_string()}).collect()
    }else{
        panic!()
    };

    assert!(fst_line[0]=="%%MatrixMarket");
    assert!(fst_line[1]=="matrix");
    let line_format=
    if fst_line[2]=="coordinate" {
        Format::Coordinate
    }else if fst_line[2]=="array"{
        Format::Array
    }else{
        panic!()
    };

    let data_type=if fst_line[3]=="real"{
        MatrixValue::Real(0.0)
    }else if fst_line[3]=="complex"{
        MatrixValue::Complex(Complex64::new(0.0, 0.0))
    }else{
        panic!()
    };

    let qual=if fst_line[4]=="general"{
        Qualifier::General
    }else if fst_line[4]=="symmetric" {
        Qualifier::Symmetric
    }else if fst_line[4]=="skew-symmetric"{
        Qualifier::SkewSymmetric
    }else if fst_line[4]=="hermitian"{
        Qualifier::Hermitian
    }else{
        panic!()
    };

    let line_processor=LineProcessor{
        qual, 
        fmt: line_format,
        entry: RawEntry{
            i: 0, j: 0, value: data_type,
        },
    };

    println!("{:?}", fst_line);
    while let Some(line)=ii.next(){
        for i in line.unwrap().split_whitespace(){
            print!("{:?}", i);
        }
        println!();
    }
}
