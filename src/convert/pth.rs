//! Reads a PyTorch `.pth` state dict (a zip holding a pickle plus raw tensor storages) without Python.
//!
//! Only what `torch.save(state_dict)` produces is understood: a minimal pickle interpreter that builds the
//! `OrderedDict` of `_rebuild_tensor_v2` tensors, and a zip reader for the stored (uncompressed) entries.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::rc::Rc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;

/// A dense f32 matrix/tensor.
#[derive(Clone)]
pub struct M {
    pub shape: Vec<usize>,
    pub d: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Dt {
    F32,
    F16,
    Bf16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dt {
    fn size(self) -> usize {
        match self {
            Dt::F64 | Dt::I64 => 8,
            Dt::F32 | Dt::I32 => 4,
            Dt::F16 | Dt::Bf16 | Dt::I16 => 2,
            Dt::I8 | Dt::U8 | Dt::Bool => 1,
        }
    }
    fn from_storage(class: &str) -> Option<Self> {
        Some(match class.rsplit('.').next()? {
            "FloatStorage" => Dt::F32,
            "HalfStorage" => Dt::F16,
            "BFloat16Storage" => Dt::Bf16,
            "DoubleStorage" => Dt::F64,
            "LongStorage" => Dt::I64,
            "IntStorage" => Dt::I32,
            "ShortStorage" => Dt::I16,
            "CharStorage" => Dt::I8,
            "ByteStorage" => Dt::U8,
            "BoolStorage" => Dt::Bool,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
struct TensorRef {
    dt: Dt,
    key: String,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

// ---------------------------------------------------------------------------------------------------- zip

struct Zip {
    map: Mmap,
    /// entry name -> (data offset, size)
    entries: HashMap<String, (usize, usize)>,
}

fn u16le(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) as usize
}
fn u32le(b: &[u8], o: usize) -> u64 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as u64
}
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

impl Zip {
    fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let map = unsafe { Mmap::map(&f)? };
        let b = &map[..];
        ensure!(b.len() > 22, "{} is not a zip file", path.display());
        let lo = b.len().saturating_sub(22 + 65535);
        let eocd = (lo..=b.len() - 22).rev().find(|&i| b[i..i + 4] == [0x50, 0x4b, 0x05, 0x06]).context("zip: no end-of-central-directory record")?;
        let (mut count, mut cd) = (u16le(b, eocd + 10) as u64, u32le(b, eocd + 16));
        if cd == 0xFFFF_FFFF || count == 0xFFFF {
            ensure!(eocd >= 20 && b[eocd - 20..eocd - 16] == [0x50, 0x4b, 0x06, 0x07], "zip64 locator missing");
            let rec = u64le(b, eocd - 20 + 8) as usize;
            ensure!(b[rec..rec + 4] == [0x50, 0x4b, 0x06, 0x06], "zip64 end record missing");
            count = u64le(b, rec + 32);
            cd = u64le(b, rec + 48);
        }
        let mut entries = HashMap::new();
        let mut p = cd as usize;
        for _ in 0..count {
            ensure!(b[p..p + 4] == [0x50, 0x4b, 0x01, 0x02], "zip: bad central directory entry");
            let method = u16le(b, p + 10);
            let (mut csize, mut usize_, mut off) = (u32le(b, p + 20), u32le(b, p + 24), u32le(b, p + 42));
            let (n, e, c) = (u16le(b, p + 28), u16le(b, p + 30), u16le(b, p + 32));
            let name = String::from_utf8_lossy(&b[p + 46..p + 46 + n]).into_owned();
            let mut x = p + 46 + n;
            let end = x + e;
            while x + 4 <= end {
                let (id, len) = (u16le(b, x), u16le(b, x + 2));
                if id == 1 {
                    let mut q = x + 4;
                    for field in [&mut usize_, &mut csize, &mut off] {
                        if *field == 0xFFFF_FFFF {
                            *field = u64le(b, q);
                            q += 8;
                        }
                    }
                }
                x += 4 + len;
            }
            ensure!(method == 0, "zip entry {name} is compressed; only stored entries are supported");
            let lh = off as usize;
            ensure!(b[lh..lh + 4] == [0x50, 0x4b, 0x03, 0x04], "zip: bad local header for {name}");
            let data = lh + 30 + u16le(b, lh + 26) + u16le(b, lh + 28);
            entries.insert(name, (data, csize as usize));
            p += 46 + n + e + c;
        }
        Ok(Self { map, entries })
    }
    fn get(&self, name: &str) -> Result<&[u8]> {
        let &(o, n) = self.entries.get(name).with_context(|| format!("zip entry {name} not found"))?;
        Ok(&self.map[o..o + n])
    }
}

// -------------------------------------------------------------------------------------------------- pickle

#[derive(Clone)]
#[allow(dead_code)]
enum V {
    None,
    Mark,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<V>),
    List(Rc<RefCell<Vec<V>>>),
    Dict(Rc<RefCell<Vec<(V, V)>>>),
    Global(String),
    Storage(Dt, String),
    Tensor(TensorRef),
    Obj,
}

fn tuple_ints(v: &V) -> Result<Vec<usize>> {
    match v {
        V::Tuple(t) => t.iter().map(|x| if let V::Int(i) = x { Ok(*i as usize) } else { bail!("pickle: expected an int") }).collect(),
        _ => bail!("pickle: expected a tuple"),
    }
}

fn pop_mark(st: &mut Vec<V>) -> Result<Vec<V>> {
    let i = st.iter().rposition(|v| matches!(v, V::Mark)).context("pickle: no mark")?;
    let items = st.split_off(i + 1);
    st.pop();
    Ok(items)
}

fn unpickle(b: &[u8]) -> Result<Vec<(String, TensorRef)>> {
    let mut st: Vec<V> = Vec::new();
    let mut memo: HashMap<usize, V> = HashMap::new();
    let mut p = 0usize;
    let rd = |p: &mut usize, n: usize| -> Result<&[u8]> {
        let s = b.get(*p..*p + n).context("pickle: truncated")?;
        *p += n;
        Ok(s)
    };
    let line = |p: &mut usize| -> Result<String> {
        let e = b[*p..].iter().position(|&c| c == b'\n').context("pickle: unterminated line")?;
        let s = String::from_utf8_lossy(&b[*p..*p + e]).into_owned();
        *p += e + 1;
        Ok(s)
    };
    loop {
        let op = *b.get(p).context("pickle: ran off the end")?;
        p += 1;
        match op {
            0x80 => p += 1,                              // PROTO
            0x95 => p += 8,                              // FRAME
            b'.' => break,                               // STOP
            b'(' => st.push(V::Mark),
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'K' => st.push(V::Int(rd(&mut p, 1)?[0] as i64)),
            b'M' => st.push(V::Int(u16::from_le_bytes(rd(&mut p, 2)?.try_into()?) as i64)),
            b'J' => st.push(V::Int(i32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as i64)),
            0x8a => {
                let n = rd(&mut p, 1)?[0] as usize;
                let bytes = rd(&mut p, n)?;
                let mut v = 0i64;
                for (i, &c) in bytes.iter().enumerate().take(8) {
                    v |= (c as i64) << (8 * i);
                }
                if n > 0 && n < 8 && bytes[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                st.push(V::Int(v));
            }
            b'G' => st.push(V::Float(f64::from_be_bytes(rd(&mut p, 8)?.try_into()?))),
            0x8c => {
                let n = rd(&mut p, 1)?[0] as usize;
                st.push(V::Str(String::from_utf8_lossy(rd(&mut p, n)?).into_owned()));
            }
            b'X' => {
                let n = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as usize;
                st.push(V::Str(String::from_utf8_lossy(rd(&mut p, n)?).into_owned()));
            }
            0x8d => {
                let n = u64::from_le_bytes(rd(&mut p, 8)?.try_into()?) as usize;
                st.push(V::Str(String::from_utf8_lossy(rd(&mut p, n)?).into_owned()));
            }
            b'U' | b'C' => {
                let n = rd(&mut p, 1)?[0] as usize;
                rd(&mut p, n)?;
                st.push(V::Obj);
            }
            b'T' | b'B' => {
                let n = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as usize;
                rd(&mut p, n)?;
                st.push(V::Obj);
            }
            b'}' => st.push(V::Dict(Default::default())),
            b']' => st.push(V::List(Default::default())),
            b')' => st.push(V::Tuple(vec![])),
            0x85 => {
                let a = st.pop().context("pickle: stack")?;
                st.push(V::Tuple(vec![a]));
            }
            0x86 => {
                let b2 = st.pop().context("pickle: stack")?;
                let a = st.pop().context("pickle: stack")?;
                st.push(V::Tuple(vec![a, b2]));
            }
            0x87 => {
                let c = st.pop().context("pickle: stack")?;
                let b2 = st.pop().context("pickle: stack")?;
                let a = st.pop().context("pickle: stack")?;
                st.push(V::Tuple(vec![a, b2, c]));
            }
            b't' => {
                let items = pop_mark(&mut st)?;
                st.push(V::Tuple(items));
            }
            b'l' => {
                let items = pop_mark(&mut st)?;
                st.push(V::List(Rc::new(RefCell::new(items))));
            }
            b'd' => {
                let items = pop_mark(&mut st)?;
                st.push(V::Dict(Rc::new(RefCell::new(items.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect()))));
            }
            b'a' | b'e' => {
                let items = if op == b'a' { vec![st.pop().context("pickle: stack")?] } else { pop_mark(&mut st)? };
                match st.last() {
                    Some(V::List(l)) => l.borrow_mut().extend(items),
                    _ => bail!("pickle: APPEND to a non-list"),
                }
            }
            b's' | b'u' => {
                let items = if op == b's' {
                    let v = st.pop().context("pickle: stack")?;
                    let k = st.pop().context("pickle: stack")?;
                    vec![k, v]
                } else {
                    pop_mark(&mut st)?
                };
                match st.last() {
                    Some(V::Dict(d)) => d.borrow_mut().extend(items.chunks(2).map(|c| (c[0].clone(), c[1].clone()))),
                    _ => bail!("pickle: SETITEM on a non-dict"),
                }
            }
            b'0' => {
                st.pop();
            }
            b'1' => {
                pop_mark(&mut st)?;
            }
            b'2' => {
                let t = st.last().cloned().context("pickle: stack")?;
                st.push(t);
            }
            b'c' => {
                let m = line(&mut p)?;
                let n = line(&mut p)?;
                st.push(V::Global(format!("{m}.{n}")));
            }
            0x93 => {
                let n = st.pop().context("pickle: stack")?;
                let m = st.pop().context("pickle: stack")?;
                match (m, n) {
                    (V::Str(m), V::Str(n)) => st.push(V::Global(format!("{m}.{n}"))),
                    _ => bail!("pickle: STACK_GLOBAL with non-strings"),
                }
            }
            0x94 => {
                let t = st.last().cloned().context("pickle: stack")?;
                memo.insert(memo.len(), t);
            }
            b'q' => {
                let i = rd(&mut p, 1)?[0] as usize;
                memo.insert(i, st.last().cloned().context("pickle: stack")?);
            }
            b'r' => {
                let i = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as usize;
                memo.insert(i, st.last().cloned().context("pickle: stack")?);
            }
            b'h' => {
                let i = rd(&mut p, 1)?[0] as usize;
                st.push(memo.get(&i).cloned().context("pickle: bad memo index")?);
            }
            b'j' => {
                let i = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as usize;
                st.push(memo.get(&i).cloned().context("pickle: bad memo index")?);
            }
            0x81 => {
                st.pop();
                st.pop();
                st.push(V::Obj);
            }
            b'R' => {
                let args = st.pop().context("pickle: stack")?;
                let f = st.pop().context("pickle: stack")?;
                let V::Tuple(args) = args else { bail!("pickle: REDUCE args are not a tuple") };
                st.push(match f {
                    V::Global(g) if g == "collections.OrderedDict" => V::Dict(Default::default()),
                    V::Global(g) if g == "torch._utils._rebuild_tensor_v2" || g == "torch._utils._rebuild_tensor" => {
                        let V::Storage(dt, key) = &args[0] else { bail!("pickle: tensor without storage") };
                        let V::Int(offset) = args[1] else { bail!("pickle: bad tensor offset") };
                        V::Tensor(TensorRef { dt: *dt, key: key.clone(), offset: offset as usize, shape: tuple_ints(&args[2])?, stride: tuple_ints(&args[3])? })
                    }
                    V::Global(g) if g == "torch._utils._rebuild_parameter" => args.into_iter().next().unwrap_or(V::None),
                    _ => V::Obj,
                });
            }
            b'b' => {
                st.pop(); // BUILD: the state (e.g. `_metadata`) is not needed
            }
            b'Q' => {
                let pid = st.pop().context("pickle: stack")?;
                let V::Tuple(t) = pid else { bail!("pickle: bad persistent id") };
                match (t.first(), t.get(1), t.get(2)) {
                    (Some(V::Str(s)), Some(V::Global(class)), Some(V::Str(key))) if s == "storage" => {
                        let dt = Dt::from_storage(class).ok_or_else(|| anyhow!("pickle: unsupported storage type {class}"))?;
                        st.push(V::Storage(dt, key.clone()));
                    }
                    _ => bail!("pickle: unsupported persistent id"),
                }
            }
            other => bail!("pickle: unsupported opcode {:#x} ({:?})", other, other as char),
        }
    }
    let Some(V::Dict(d)) = st.pop() else { bail!("pickle: the file is not a state dict") };
    let d = d.borrow();
    let mut out = Vec::new();
    for (k, v) in d.iter() {
        if let (V::Str(k), V::Tensor(t)) = (k, v) {
            out.push((k.clone(), t.clone()));
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------------------------- the file

/// A `.pth` state dict whose tensors are read (as f32) on demand.
pub struct Pth {
    zip: Zip,
    prefix: String,
    tensors: HashMap<String, TensorRef>,
}

impl Pth {
    pub fn open(path: &Path) -> Result<Self> {
        let zip = Zip::open(path)?;
        let pkl = zip.entries.keys().find(|k| k.ends_with("data.pkl")).context("not a PyTorch zip checkpoint (no data.pkl)")?.clone();
        let prefix = pkl.trim_end_matches("data.pkl").to_string();
        let tensors = unpickle(zip.get(&pkl)?)?.into_iter().collect();
        Ok(Self { zip, prefix, tensors })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<M> {
        let t = self.tensors.get(name).with_context(|| format!("codec checkpoint has no tensor {name}"))?;
        let n: usize = t.shape.iter().product();
        let mut expect = 1usize;
        for (&s, &st) in t.shape.iter().zip(&t.stride).rev() {
            ensure!(s == 1 || st == expect, "{name}: non-contiguous tensors are not supported");
            expect *= s;
        }
        let raw = self.zip.get(&format!("{}data/{}", self.prefix, t.key))?;
        let sz = t.dt.size();
        let bytes = raw.get(t.offset * sz..(t.offset + n) * sz).with_context(|| format!("{name}: storage too small"))?;
        let d: Vec<f32> = match t.dt {
            Dt::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Dt::F16 => bytes.chunks_exact(2).map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
            Dt::Bf16 => bytes.chunks_exact(2).map(|c| bf16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
            Dt::F64 => bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            _ => bail!("{name}: unsupported dtype {:?}", t.dt),
        };
        Ok(M { shape: t.shape.clone(), d })
    }
}
