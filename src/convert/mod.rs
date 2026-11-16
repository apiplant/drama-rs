//! Builds the dense half of a model directory from the original `fishaudio/drama-3` checkpoint, in pure Rust:
//! `globals`, `tower` and per-layer dense weights (int8, group 128), the codec (`codec.pth` ->
//! `codec.safetensors`), `drama.json` and the tokenizer. The routed experts stay in the original (FP8), so the
//! result is a "dense-only" directory. The dense part ports `tools/convert.py` (which still
//! quantizes the int4 experts); the codec part replaces the old `convert_codec.py`.

mod pth;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use pth::{Pth, M};

const Q8_GROUP: usize = 128;
const Q4_GROUP: usize = 64;
const FP8_BLOCK: usize = 128;

// ------------------------------------------------------------------------------------------------ helpers

/// Runs `f(row_index, row)` over the `row_len`-sized rows of `out`, spread over the available cores.
fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len() / row_len.max(1);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(rows.max(1));
    let per = rows.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * per + i, row);
                }
            });
        }
    });
}

/// Tensors to be written to one safetensors file.
#[derive(Default)]
struct Out {
    items: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
}

impl Out {
    fn f32(&mut self, name: &str, shape: &[usize], d: &[f32]) {
        self.items.push((name.into(), Dtype::F32, shape.into(), d.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    fn f16(&mut self, name: &str, shape: &[usize], d: &[f32]) {
        self.items.push((name.into(), Dtype::F16, shape.into(), d.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect()));
    }
    fn bf16(&mut self, name: &str, shape: &[usize], d: &[f32]) {
        self.items.push((name.into(), Dtype::BF16, shape.into(), d.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect()));
    }
    fn raw(&mut self, name: &str, dt: Dtype, shape: &[usize], bytes: Vec<u8>) {
        self.items.push((name.into(), dt, shape.into(), bytes));
    }

    fn write(&self, path: &Path, format: &str) -> Result<()> {
        let tmp = path.with_extension("safetensors.tmp");
        let views = self
            .items
            .iter()
            .map(|(n, dt, sh, b)| Ok((n.clone(), TensorView::new(*dt, sh.clone(), b)?)))
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        let meta = HashMap::from([("format".to_string(), format.to_string())]);
        safetensors::serialize_to_file(views, &Some(meta), &tmp)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Symmetric int8, one f16 scale per 128 inputs of a row; stored as `name.q8` [n, k] and `name.s` [n, k/128].
fn put_q8(out: &mut Out, name: &str, w: &M) -> Result<()> {
    ensure!(w.shape.len() == 2, "{name}: expected a matrix");
    let (n, k) = (w.shape[0], w.shape[1]);
    ensure!(k % Q8_GROUP == 0, "{name}: {k} is not a multiple of {Q8_GROUP}");
    let groups = k / Q8_GROUP;
    let mut q = vec![0i8; n * k];
    let mut s = vec![f16::ZERO; n * groups];
    {
        // one row of q and one row of scales per index; split them together
        let sp = s.as_mut_ptr() as usize;
        par_rows(&mut q, k, |r, qrow| {
            let src = &w.d[r * k..(r + 1) * k];
            // SAFETY: every row index writes its own disjoint `groups`-long slice of `s`
            let srow = unsafe { std::slice::from_raw_parts_mut((sp as *mut f16).add(r * groups), groups) };
            for g in 0..groups {
                let x = &src[g * Q8_GROUP..(g + 1) * Q8_GROUP];
                let amax = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
                let scale = f16::from_f32(amax / 127.0).to_f32().max(1e-12);
                for (o, v) in qrow[g * Q8_GROUP..(g + 1) * Q8_GROUP].iter_mut().zip(x) {
                    *o = (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                }
                srow[g] = f16::from_f32(scale);
            }
        });
    }
    out.raw(&format!("{name}.q8"), Dtype::I8, &[n, k], q.iter().map(|&v| v as u8).collect());
    out.raw(&format!("{name}.s"), Dtype::F16, &[n, groups], s.iter().flat_map(|v| v.to_le_bytes()).collect());
    Ok(())
}

fn cat_rows(parts: &[&M]) -> M {
    let cols: usize = parts[0].shape[1..].iter().product();
    let rows: usize = parts.iter().map(|p| p.shape[0]).sum();
    let mut d = Vec::with_capacity(rows * cols);
    for p in parts {
        d.extend_from_slice(&p.d);
    }
    M { shape: vec![rows, cols], d }
}

fn plus_one(w: M) -> M {
    M { d: w.d.iter().map(|x| x + 1.0).collect(), ..w }
}

// ------------------------------------------------------------------------------- the original checkpoint

fn fp8_lut() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        let (sign, e, m) = ((i >> 7) & 1, (i >> 3) & 15, i & 7);
        let mag = if e == 15 && m == 7 {
            f32::NAN
        } else if e == 0 {
            m as f32 / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
        };
        *v = if sign == 1 { -mag } else { mag };
    }
    t
}

struct Checkpoint<'a> {
    weight_map: HashMap<String, usize>,
    shards: Vec<SafeTensors<'a>>,
    lut: [f32; 256],
}

impl<'a> Checkpoint<'a> {
    fn new(dir: &Path, maps: &'a [Mmap], names: &[String]) -> Result<Self> {
        let idx: Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).context("read model.safetensors.index.json")?)?;
        let mut weight_map = HashMap::new();
        for (k, v) in idx["weight_map"].as_object().context("index has no weight_map")? {
            let file = v.as_str().context("bad weight_map")?;
            weight_map.insert(k.clone(), names.iter().position(|n| n == file).with_context(|| format!("shard {file} missing"))?);
        }
        let shards = maps.iter().map(|m| SafeTensors::deserialize(m)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { weight_map, shards, lut: fp8_lut() })
    }

    fn get(&self, name: &str) -> Result<M> {
        let &s = self.weight_map.get(name).with_context(|| format!("checkpoint has no tensor {name}"))?;
        let t = self.shards[s].tensor(name)?;
        let b = t.data();
        let d: Vec<f32> = match t.dtype() {
            Dtype::BF16 => b.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Dtype::F16 => b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::F8_E4M3 => b.iter().map(|&x| self.lut[x as usize]).collect(),
            other => bail!("{name}: unsupported dtype {other:?}"),
        };
        Ok(M { shape: t.shape().to_vec(), d })
    }

    fn is_fp8(&self, name: &str) -> bool {
        self.weight_map.get(name).is_some_and(|&s| self.shards[s].tensor(name).is_ok_and(|t| t.dtype() == Dtype::F8_E4M3))
    }

    /// A weight in f32; block-FP8 weights are multiplied out by their `_scale_inv`.
    fn dequant(&self, name: &str) -> Result<M> {
        let mut w = self.get(name)?;
        if self.is_fp8(name) {
            let s = self.get(&format!("{name}_scale_inv"))?;
            let (n, k) = (w.shape[0], w.shape[1]);
            let sc = k.div_ceil(FP8_BLOCK);
            ensure!(s.d.len() >= n.div_ceil(FP8_BLOCK) * sc, "{name}: scale_inv too small");
            par_rows(&mut w.d, k, |r, row| {
                let srow = &s.d[(r / FP8_BLOCK) * sc..];
                for (j, v) in row.iter_mut().enumerate() {
                    *v *= srow[j / FP8_BLOCK];
                }
            });
        }
        Ok(w)
    }
}

// ---------------------------------------------------------------------------------------------- converters

struct Cfg<'a>(&'a Value);

impl Cfg<'_> {
    fn n(&self, path: &[&str]) -> Result<usize> {
        let mut v = self.0;
        for p in path {
            v = &v[*p];
        }
        v.as_u64().map(|x| x as usize).with_context(|| format!("config.json: missing {}", path.join(".")))
    }
}

fn convert_globals(ck: &Checkpoint, cfg: &Cfg) -> Result<Out> {
    let mut out = Out::default();
    let emb = ck.get("model.language_model.embed_tokens.weight")?;
    out.bf16("embed_tokens", &emb.shape, &emb.d);
    let lm = ck.get("lm_head.weight")?;
    ensure!(lm.shape[0] == cfg.n(&["text_config", "vocab_size"])?, "lm_head rows != vocab_size");
    let (eos, s0, s1) = (cfg.n(&["eos_token_id"])?, cfg.n(&["semantic_start_token_id"])?, cfg.n(&["semantic_end_token_id"])?);
    let mut cand: Vec<usize> = std::iter::once(eos).chain(s0..=s1).collect();
    cand.sort_unstable();
    cand.dedup();
    let k = lm.shape[1];
    let mut rows = Vec::with_capacity(cand.len() * k);
    for &c in &cand {
        rows.extend_from_slice(&lm.d[c * k..(c + 1) * k]);
    }
    out.bf16("lm_head_cand", &[cand.len(), k], &rows);
    out.raw("lm_head_cand_ids", Dtype::I32, &[cand.len()], cand.iter().flat_map(|&c| (c as i32).to_le_bytes()).collect());
    let n = plus_one(ck.get("model.language_model.norm.weight")?);
    out.f32("final_norm", &n.shape, &n.d);
    Ok(out)
}

fn convert_tower(ck: &Checkpoint, cfg: &Cfg) -> Result<Out> {
    let mut out = Out::default();
    let emb = ck.get("audio_decoder.codebook_embeddings.weight")?;
    out.bf16("codebook_emb", &emb.shape, &emb.d);
    let dc = |k: &str| cfg.n(&["audio_decoder_config", k]);
    let mut names: Vec<String> = (0..dc("shared_enrichment_layers")?).map(|j| format!("audio_decoder_mtp_shared.layers.{j}.")).collect();
    names.extend((0..dc("num_codebooks")? - 1).map(|j| format!("audio_decoder_mtp.layers.{j}.")));
    for (t, p) in names.iter().enumerate() {
        let o = format!("t{t}.");
        let emb_name = format!("{p}{}", if p.contains("shared") { "semantic_embedding.weight" } else { "codebook_embedding.weight" });
        let e = ck.get(&emb_name)?;
        out.bf16(&format!("{o}tok_emb"), &e.shape, &e.d);
        let tl = format!("{p}transformer_layer.");
        for (dst, src) in [
            ("enorm", format!("{p}enorm.weight")),
            ("hnorm", format!("{p}hnorm.weight")),
            ("final_norm", format!("{p}final_layernorm.weight")),
            ("input_norm", format!("{tl}attention_norm.weight")),
            ("post_norm", format!("{tl}ffn_norm.weight")),
            ("q_norm", format!("{tl}attention.q_norm.weight")),
            ("k_norm", format!("{tl}attention.k_norm.weight")),
        ] {
            let w = plus_one(ck.get(&src)?);
            out.f32(&format!("{o}{dst}"), &w.shape, &w.d);
        }
        put_q8(&mut out, &format!("{o}eh_proj"), &ck.dequant(&format!("{p}eh_proj.weight"))?)?;
        put_q8(&mut out, &format!("{o}qkv"), &ck.dequant(&format!("{tl}attention.wqkv.weight"))?)?;
        put_q8(&mut out, &format!("{o}o"), &ck.dequant(&format!("{tl}attention.wo.weight"))?)?;
        let gu = cat_rows(&[&ck.dequant(&format!("{tl}feed_forward.w1.weight"))?, &ck.dequant(&format!("{tl}feed_forward.w3.weight"))?]);
        put_q8(&mut out, &format!("{o}gate_up"), &gu)?;
        put_q8(&mut out, &format!("{o}down"), &ck.dequant(&format!("{tl}feed_forward.w2.weight"))?)?;
    }
    let heads: Vec<M> = (0..dc("num_codebooks")? - 1).map(|j| ck.dequant(&format!("audio_decoder_mtp.heads.{j}.weight"))).collect::<Result<_>>()?;
    put_q8(&mut out, "heads", &cat_rows(&heads.iter().collect::<Vec<_>>()))?;
    Ok(out)
}

/// One backbone layer without its routed experts.
fn convert_layer(ck: &Checkpoint, cfg: &Cfg, i: usize) -> Result<Out> {
    let tc = &cfg.0["text_config"];
    let p = format!("model.language_model.layers.{i}.");
    let mut out = Out::default();
    for (dst, src) in [("input_norm", "input_layernorm.weight"), ("post_norm", "post_attention_layernorm.weight")] {
        let w = plus_one(ck.get(&format!("{p}{src}"))?);
        out.f32(dst, &w.shape, &w.d);
    }
    let layer_type = tc["layer_types"][i].as_str().context("config.json: bad layer_types")?;
    if layer_type == "linear_attention" {
        let la = format!("{p}linear_attn.");
        // one fused input projection: [qkv (conv channels) | z | b | a]
        let w = cat_rows(&[
            &ck.dequant(&format!("{la}in_proj_qkv.weight"))?,
            &ck.dequant(&format!("{la}in_proj_z.weight"))?,
            &ck.dequant(&format!("{la}in_proj_b.weight"))?,
            &ck.dequant(&format!("{la}in_proj_a.weight"))?,
        ]);
        put_q8(&mut out, "la.in_proj", &w)?;
        put_q8(&mut out, "la.out_proj", &ck.dequant(&format!("{la}out_proj.weight"))?)?;
        let conv = ck.get(&format!("{la}conv1d.weight"))?; // [C, 1, 4]
        out.f32("la.conv_w", &[conv.shape[0], conv.shape[2]], &conv.d);
        for (dst, src) in [("la.A_log", "A_log"), ("la.dt_bias", "dt_bias"), ("la.norm", "norm.weight")] {
            let w = ck.get(&format!("{la}{src}"))?;
            out.f32(dst, &w.shape, &w.d);
        }
    } else {
        let sa = format!("{p}self_attn.");
        let (heads, hd) = (cfg.n(&["text_config", "num_attention_heads"])?, cfg.n(&["text_config", "head_dim"])?);
        let q = ck.dequant(&format!("{sa}q_proj.weight"))?; // per head [query | gate]
        let cols = q.shape[1];
        let (mut qq, mut qg) = (Vec::with_capacity(heads * hd * cols), Vec::with_capacity(heads * hd * cols));
        for h in 0..heads {
            let base = h * 2 * hd * cols;
            qq.extend_from_slice(&q.d[base..base + hd * cols]);
            qg.extend_from_slice(&q.d[base + hd * cols..base + 2 * hd * cols]);
        }
        let (qq, qg) = (M { shape: vec![heads * hd, cols], d: qq }, M { shape: vec![heads * hd, cols], d: qg });
        let w = cat_rows(&[&qq, &qg, &ck.dequant(&format!("{sa}k_proj.weight"))?, &ck.dequant(&format!("{sa}v_proj.weight"))?]);
        put_q8(&mut out, "attn.qkv", &w)?;
        put_q8(&mut out, "attn.o", &ck.dequant(&format!("{sa}o_proj.weight"))?)?;
        for (dst, src) in [("attn.q_norm", "q_norm.weight"), ("attn.k_norm", "k_norm.weight")] {
            let w = plus_one(ck.get(&format!("{sa}{src}"))?);
            out.f32(dst, &w.shape, &w.d);
        }
    }
    let mp = format!("{p}mlp.");
    // router rows 0..E-1 plus the shared-expert gate as row E
    let router = cat_rows(&[&ck.dequant(&format!("{mp}gate.weight"))?, &ck.dequant(&format!("{mp}shared_expert_gate.weight"))?]);
    out.bf16("moe.router", &router.shape, &router.d);
    let gu = cat_rows(&[&ck.dequant(&format!("{mp}shared_expert.gate_proj.weight"))?, &ck.dequant(&format!("{mp}shared_expert.up_proj.weight"))?]);
    put_q8(&mut out, "moe.shared_gate_up", &gu)?;
    put_q8(&mut out, "moe.shared_down", &ck.dequant(&format!("{mp}shared_expert.down_proj.weight"))?)?;
    Ok(out)
}

/// Writes the dense half of a model directory (everything but the routed experts) into `dst`, from the original
/// checkpoint in `src`. Existing files are kept, so an interrupted run resumes; `drama.json` is written last.
pub fn dense(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let cfg_json: Value = serde_json::from_slice(&std::fs::read(src.join("config.json")).context("read config.json")?)?;
    let cfg = Cfg(&cfg_json);
    let layers = cfg.n(&["text_config", "num_hidden_layers"])?;

    let idx: Value = serde_json::from_slice(&std::fs::read(src.join("model.safetensors.index.json")).context("read model.safetensors.index.json")?)?;
    let mut names: Vec<String> = idx["weight_map"].as_object().context("bad index")?.values().filter_map(|v| v.as_str().map(String::from)).collect();
    names.sort();
    names.dedup();
    let maps: Vec<Mmap> = names
        .iter()
        .map(|n| File::open(src.join(n)).and_then(|f| unsafe { Mmap::map(&f) }).with_context(|| format!("open shard {n}")))
        .collect::<Result<_>>()?;
    let ck = Checkpoint::new(src, &maps, &names)?;

    let mut jobs: Vec<(String, Box<dyn Fn() -> Result<Out> + '_>)> = vec![
        ("globals.safetensors".into(), Box::new(|| convert_globals(&ck, &cfg))),
        ("tower.safetensors".into(), Box::new(|| convert_tower(&ck, &cfg))),
    ];
    for i in 0..layers {
        let (ck, cfg) = (&ck, &cfg);
        jobs.push((format!("layer{i:02}.safetensors"), Box::new(move || convert_layer(ck, cfg, i))));
    }
    let total = jobs.len();
    for (n, (file, job)) in jobs.iter().enumerate() {
        let path = dst.join(file);
        if !path.is_file() {
            job()?.write(&path, "drama-rs-v1")?;
        }
        eprint!("\rconverting dense weights: {}/{total}", n + 1);
    }
    eprintln!();

    if !dst.join("tokenizer.json").is_file() {
        std::fs::copy(src.join("tokenizer.json"), dst.join("tokenizer.json")).context("copy tokenizer.json")?;
    }
    if src.join("LICENSE").is_file() && !dst.join("LICENSE").is_file() {
        std::fs::copy(src.join("LICENSE"), dst.join("LICENSE"))?;
    }
    let meta = serde_json::json!({
        "format": "drama-rs-v1",
        "q4_group": Q4_GROUP,
        "q8_group": Q8_GROUP,
        "config": cfg_json,
        "dense_only": true,
    });
    let tmp = dst.join("drama.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&meta)?)?;
    std::fs::rename(tmp, dst.join("drama.json"))?;
    Ok(())
}

// ------------------------------------------------------------------------------------------------- codec

/// `weight` of a (weight-normalised) layer: plain, old-style `weight_g`/`weight_v`, or parametrized
/// `parametrizations.weight.original0/1`; the norm is taken over every dim but the first.
fn codec_weight(p: &Pth, prefix: &str) -> Result<M> {
    if p.has(&format!("{prefix}.weight")) {
        return p.get(&format!("{prefix}.weight"));
    }
    let (g, v) = if p.has(&format!("{prefix}.weight_g")) {
        (p.get(&format!("{prefix}.weight_g"))?, p.get(&format!("{prefix}.weight_v"))?)
    } else {
        (p.get(&format!("{prefix}.parametrizations.weight.original0"))?, p.get(&format!("{prefix}.parametrizations.weight.original1"))?)
    };
    let rows = v.shape[0];
    let per = v.d.len() / rows;
    let mut d = v.d;
    par_rows(&mut d, per, |r, row| {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k = g.d[r] / norm;
        for x in row {
            *x *= k;
        }
    });
    Ok(M { shape: v.shape, d })
}

/// [Cout, Cin, k] -> [Cout, k*Cin] (column j*Cin + ci)
fn conv_w(w: &M) -> M {
    let (co, ci, k) = (w.shape[0], w.shape[1], w.shape[2]);
    let mut d = vec![0f32; w.d.len()];
    for o in 0..co {
        for c in 0..ci {
            for j in 0..k {
                d[o * k * ci + j * ci + c] = w.d[(o * ci + c) * k + j];
            }
        }
    }
    M { shape: vec![co, k * ci], d }
}

/// ConvTranspose1d [Cin, Cout, k] with stride s -> [s*Cout, 2*Cin] (k == 2s; cols x[t] | x[t-1]) or [s*Cout, Cin] (k == s)
fn tconv_w(w: &M, stride: usize) -> Result<M> {
    let (ci, co, k) = (w.shape[0], w.shape[1], w.shape[2]);
    let at = |c: usize, o: usize, j: usize| w.d[(c * co + o) * k + j];
    if k == 2 * stride {
        let mut d = vec![0f32; stride * co * 2 * ci];
        for r in 0..stride {
            for o in 0..co {
                let row = (r * co + o) * 2 * ci;
                for c in 0..ci {
                    d[row + c] = at(c, o, r);
                    d[row + ci + c] = at(c, o, stride + r);
                }
            }
        }
        Ok(M { shape: vec![stride * co, 2 * ci], d })
    } else {
        ensure!(k == stride, "transposed conv with kernel {k} and stride {stride}");
        let mut d = vec![0f32; stride * co * ci];
        for r in 0..stride {
            for o in 0..co {
                for c in 0..ci {
                    d[(r * co + o) * ci + c] = at(c, o, r);
                }
            }
        }
        Ok(M { shape: vec![stride * co, ci], d })
    }
}

fn squeeze(mut w: M, dim: usize) -> M {
    w.shape.remove(dim);
    w
}

/// Writes `codec.safetensors` from the original `codec.pth`.
pub fn codec(pth: &Path, out_path: &Path) -> Result<()> {
    let p = Pth::open(pth)?;
    let mut t = Out::default();
    let g = |name: &str| p.get(name);
    let f = |t: &mut Out, name: &str, w: &M| t.f32(name, &w.shape, &w.d);
    let h = |t: &mut Out, name: &str, w: &M| t.f16(name, &w.shape, &w.d);
    // a pytorch `alpha` [1, C, 1] is stored flat
    let alpha = |t: &mut Out, name: &str, key: &str| -> Result<()> {
        let w = g(key)?;
        t.f32(name, &[w.d.len()], &w.d);
        Ok(())
    };
    let conv = |t: &mut Out, name: &str, prefix: &str, half: bool| -> Result<()> {
        let w = conv_w(&codec_weight(&p, prefix)?);
        if half { h(t, &format!("{name}.w"), &w) } else { f(t, &format!("{name}.w"), &w) }
        let b = g(&format!("{prefix}.bias"))?;
        f(t, &format!("{name}.b"), &b);
        Ok(())
    };
    let tconv = |t: &mut Out, name: &str, prefix: &str, stride: usize| -> Result<()> {
        h(t, &format!("{name}.w"), &tconv_w(&codec_weight(&p, prefix)?, stride)?);
        f(t, &format!("{name}.b"), &g(&format!("{prefix}.bias"))?);
        Ok(())
    };

    // per-code output vectors: out_proj(codebook[c]) + bias, semantic first
    let mut vqs = vec!["quantizer.semantic_quantizer.quantizers.0".to_string()];
    let mut i = 0;
    while p.has(&format!("quantizer.quantizer.quantizers.{i}.codebook.weight")) {
        vqs.push(format!("quantizer.quantizer.quantizers.{i}"));
        i += 1;
    }
    let mut table: Vec<f32> = Vec::new();
    let mut codes = 0;
    for vq in &vqs {
        let cbk = g(&format!("{vq}.codebook.weight"))?; // [n, 8]
        let w = squeeze(codec_weight(&p, &format!("{vq}.out_proj"))?, 2); // [1024, 8]
        let bias = g(&format!("{vq}.out_proj.bias"))?;
        let (n, d, o) = (cbk.shape[0], cbk.shape[1], w.shape[0]);
        for c in 0..n {
            for j in 0..o {
                let mut acc = 0f32;
                for k in 0..d {
                    acc += cbk.d[c * d + k] * w.d[j * d + k];
                }
                table.push(acc + bias.d[j]);
            }
        }
        codes += n;
    }
    t.f32("rvq.table", &[codes, 1024], &table);

    let transformer = |t: &mut Out, prefix: &str, key: &str| -> Result<()> {
        let mut j = 0;
        while p.has(&format!("{key}.layers.{j}.attention.wqkv.weight")) {
            let (l, o) = (format!("{key}.layers.{j}."), format!("{prefix}.l{j}."));
            for (dst, src) in [("attn_norm", "attention_norm.weight"), ("ffn_norm", "ffn_norm.weight"), ("attn_scale", "attention_layer_scale.gamma"), ("ffn_scale", "ffn_layer_scale.gamma")] {
                f(t, &format!("{o}{dst}"), &g(&format!("{l}{src}"))?);
            }
            h(t, &format!("{o}wqkv"), &g(&format!("{l}attention.wqkv.weight"))?);
            h(t, &format!("{o}wo"), &g(&format!("{l}attention.wo.weight"))?);
            h(t, &format!("{o}w13"), &cat_rows(&[&g(&format!("{l}feed_forward.w1.weight"))?, &g(&format!("{l}feed_forward.w3.weight"))?]));
            h(t, &format!("{o}w2"), &g(&format!("{l}feed_forward.w2.weight"))?);
            j += 1;
        }
        ensure!(j > 0, "codec transformer {key} has no layers");
        f(t, &format!("{prefix}.norm"), &g(&format!("{key}.norm.weight"))?);
        Ok(())
    };
    let convnext = |t: &mut Out, prefix: &str, key: &str| -> Result<()> {
        f(t, &format!("{prefix}.dw_w"), &squeeze(g(&format!("{key}.dwconv.conv.weight"))?, 1));
        f(t, &format!("{prefix}.dw_b"), &g(&format!("{key}.dwconv.conv.bias"))?);
        f(t, &format!("{prefix}.ln_w"), &g(&format!("{key}.norm.weight"))?);
        f(t, &format!("{prefix}.ln_b"), &g(&format!("{key}.norm.bias"))?);
        h(t, &format!("{prefix}.pw1_w"), &g(&format!("{key}.pwconv1.weight"))?);
        f(t, &format!("{prefix}.pw1_b"), &g(&format!("{key}.pwconv1.bias"))?);
        h(t, &format!("{prefix}.pw2_w"), &g(&format!("{key}.pwconv2.weight"))?);
        f(t, &format!("{prefix}.pw2_b"), &g(&format!("{key}.pwconv2.bias"))?);
        f(t, &format!("{prefix}.gamma"), &g(&format!("{key}.gamma"))?);
        Ok(())
    };

    transformer(&mut t, "post", "quantizer.post_module")?;
    transformer(&mut t, "pre", "quantizer.pre_module")?;
    let mut i = 0;
    while p.has(&format!("quantizer.upsample.{i}.0.conv.bias")) {
        tconv(&mut t, &format!("up{i}"), &format!("quantizer.upsample.{i}.0.conv"), 2)?;
        convnext(&mut t, &format!("up{i}.cnx"), &format!("quantizer.upsample.{i}.1"))?;
        i += 1;
    }

    conv(&mut t, "dec.in", "decoder.model.0.conv", true)?;
    for (b, stride) in [8usize, 8, 4, 2].into_iter().enumerate() {
        let blk = format!("decoder.model.{}.block", 1 + b);
        let o = format!("dec.b{b}.");
        alpha(&mut t, &format!("{o}alpha"), &format!("{blk}.0.alpha"))?;
        tconv(&mut t, &format!("{o}up"), &format!("{blk}.1.conv"), stride)?;
        for u in 0..3 {
            let ru = format!("{blk}.{}.block", 2 + u);
            alpha(&mut t, &format!("{o}ru{u}.a1"), &format!("{ru}.0.alpha"))?;
            conv(&mut t, &format!("{o}ru{u}.c1"), &format!("{ru}.1.conv"), true)?;
            alpha(&mut t, &format!("{o}ru{u}.a2"), &format!("{ru}.2.alpha"))?;
            conv(&mut t, &format!("{o}ru{u}.c2"), &format!("{ru}.3.conv"), true)?;
        }
    }

    // encoder side (reference audio -> codes)
    conv(&mut t, "enc.in", "encoder.block.0.conv", true)?;
    for b in 0..4 {
        let blk = format!("encoder.block.{}.block", 1 + b);
        let o = format!("enc.b{b}.");
        for u in 0..3 {
            let ru = format!("{blk}.{u}.block");
            alpha(&mut t, &format!("{o}ru{u}.a1"), &format!("{ru}.0.alpha"))?;
            conv(&mut t, &format!("{o}ru{u}.c1"), &format!("{ru}.1.conv"), true)?;
            alpha(&mut t, &format!("{o}ru{u}.a2"), &format!("{ru}.2.alpha"))?;
            conv(&mut t, &format!("{o}ru{u}.c2"), &format!("{ru}.3.conv"), true)?;
        }
        alpha(&mut t, &format!("{o}alpha"), &format!("{blk}.3.alpha"))?;
        conv(&mut t, &format!("{o}down"), &format!("{blk}.4.conv"), true)?;
        if p.has(&format!("{blk}.5.layers.0.attention.wqkv.weight")) {
            transformer(&mut t, &format!("enc.b{b}.tr"), &format!("{blk}.5"))?;
        }
    }
    alpha(&mut t, "enc.out.alpha", "encoder.block.5.alpha")?;
    conv(&mut t, "enc.out", "encoder.block.6.conv", true)?;
    let mut i = 0;
    while p.has(&format!("quantizer.downsample.{i}.0.conv.bias")) {
        conv(&mut t, &format!("down{i}"), &format!("quantizer.downsample.{i}.0.conv"), true)?;
        convnext(&mut t, &format!("down{i}.cnx"), &format!("quantizer.downsample.{i}.1"))?;
        i += 1;
    }
    for (i, vq) in vqs.iter().enumerate() {
        f(&mut t, &format!("vq{i}.in_w"), &squeeze(codec_weight(&p, &format!("{vq}.in_proj"))?, 2));
        f(&mut t, &format!("vq{i}.in_b"), &g(&format!("{vq}.in_proj.bias"))?);
        let mut cb = g(&format!("{vq}.codebook.weight"))?;
        let d = cb.shape[1];
        par_rows(&mut cb.d, d, |_, row| {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in row {
                *x /= n;
            }
        });
        f(&mut t, &format!("vq{i}.codebook_n"), &cb);
    }

    alpha(&mut t, "dec.out.alpha", "decoder.model.5.alpha")?;
    conv(&mut t, "dec.out", "decoder.model.6.conv", false)?;
    t.write(out_path, "drama-rs-codec-v1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_e4m3_values() {
        let t = fp8_lut();
        assert_eq!(t[0x38], 1.0);
        assert_eq!(t[0x7e], 448.0);
        assert_eq!(t[0xb8], -1.0);
        assert_eq!(t[0x01], 2f32.powi(-9));
        assert!(t[0x7f].is_nan());
    }

    #[test]
    fn q8_roundtrip_and_ties() {
        let mut d = vec![0f32; 128];
        d[0] = 127.0;
        d[1] = 0.5; // 0.5 / 1.0 rounds to even (0), not away from zero
        d[2] = 1.5; // rounds to 2
        let mut out = Out::default();
        put_q8(&mut out, "w", &M { shape: vec![1, 128], d }).unwrap();
        let q = &out.items[0].3;
        assert_eq!((q[0] as i8, q[1] as i8, q[2] as i8), (127, 0, 2));
        let s = &out.items[1].3;
        assert_eq!(f16::from_le_bytes([s[0], s[1]]).to_f32(), 1.0);
    }

    #[test]
    fn tconv_layouts() {
        // [Cin=1, Cout=1, k=4], stride 2: row r = [tap r (x[t]) | tap r+2 (x[t-1])]
        let w = M { shape: vec![1, 1, 4], d: vec![1.0, 2.0, 3.0, 4.0] };
        let r = tconv_w(&w, 2).unwrap();
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(r.d, vec![1.0, 3.0, 2.0, 4.0]);
    }
}
