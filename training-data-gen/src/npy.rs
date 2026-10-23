//! Minimal `.npy` (format 1.0) writer for little-endian arrays.

use std::io::Write;
use std::path::Path;

pub trait Elem: Copy {
    const DESCR: &'static str;
    fn put(self, out: &mut Vec<u8>);
}

impl Elem for f32 {
    const DESCR: &'static str = "<f4";
    fn put(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Elem for i64 {
    const DESCR: &'static str = "<i8";
    fn put(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Elem for u8 {
    const DESCR: &'static str = "|u1";
    fn put(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}

pub fn write<T: Elem>(path: &Path, shape: &[usize], data: &[T]) -> std::io::Result<()> {
    assert_eq!(shape.iter().product::<usize>(), data.len(), "npy shape mismatch for {}", path.display());
    let dims = match shape {
        [n] => format!("{n},"),
        _ => shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({dims}), }}", T::DESCR);
    // Magic (6) + version (2) + length (2) + header + '\n' must be a multiple of 64.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &x in data {
        x.put(&mut out);
    }
    std::fs::File::create(path)?.write_all(&out)
}
