//! Finds the model, or downloads it from Hugging Face into a cache directory.
//!
//! Cache location: `$XDG_CACHE_HOME/drama-rs/drama-q4/`, falling back to `~/.cache/drama-rs/drama-q4/`. Each file
//! is fetched to a `.part` sibling and renamed into place once complete, so a killed download never leaves a file
//! that looks done, and a rerun resumes with the files still missing.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// The Hugging Face repository the model comes from.
pub const HF_REPO: &str = "apiplant/drama-q4";
/// Transformer layers, one `layerNN.safetensors` each.
const LAYERS: usize = 40;
const FILES: &[&str] = &[
    "drama.json",
    "tokenizer.json",
    "globals.safetensors",
    "tower.safetensors",
    "codec.safetensors",
    "LICENSE",
    "NOTICE",
];

fn files() -> Vec<String> {
    FILES.iter().map(|s| s.to_string()).chain((0..LAYERS).map(|i| format!("layer{i:02}.safetensors"))).collect()
}

/// `$XDG_CACHE_HOME/drama-rs`, defaulting to `~/.cache/drama-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("drama-rs"))
}

/// Where [`download`] puts (or would put) the model directory.
pub fn model_dir() -> Option<PathBuf> {
    Some(cache_dir()?.join(HF_REPO.rsplit('/').next().unwrap_or(HF_REPO)))
}

fn is_model_dir(p: &Path) -> bool {
    p.join("drama.json").is_file() && p.join("globals.safetensors").is_file() && p.join("codec.safetensors").is_file()
}

/// The model to use when none is given: `models/drama-3-q4` or `drama-q4` in the working directory, its parent, or
/// next to the binary (up to three levels above it, i.e. the repo root for `target/release/drama`), then the cache
/// directory. `None` if none of them holds a complete model.
pub fn find_model() -> Option<PathBuf> {
    let mut roots = vec![PathBuf::from("."), PathBuf::from("..")];
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        roots.extend(dir.ancestors().take(4).map(Path::to_path_buf));
    }
    let local = roots.iter().flat_map(|r| [r.join("models/drama-3-q4"), r.join("drama-q4")]);
    local.chain(model_dir()).find(|p| is_model_dir(p) && files().iter().all(|f| f.ends_with("LICENSE") || f.ends_with("NOTICE") || p.join(f).is_file()))
}

/// [`find_model`], else download into the cache directory (about 21 GB).
pub fn default_model() -> Result<PathBuf> {
    match find_model() {
        Some(dir) => Ok(dir),
        None => download(),
    }
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string();
    let part = dest.with_file_name(format!("{name}.part"));

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut written, mut last_report) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if written - last_report >= 16 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!("\r  {name} {:.0}% ({} / {} MiB)  ", written as f64 * 100.0 / len as f64, written >> 20, len >> 20),
                None => eprint!("\r  {name} {} MiB  ", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    if written >= 16 << 20 {
        eprintln!("\r  {name} done ({} MiB)          ", written >> 20);
    }
    Ok(())
}

/// Downloads the model into the cache directory unless it is already there, and returns the directory.
pub fn download() -> Result<PathBuf> {
    let dir = model_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    let missing: Vec<String> = files().into_iter().filter(|f| !dir.join(f).is_file()).collect();
    if missing.is_empty() {
        return Ok(dir);
    }
    eprintln!("Downloading {HF_REPO} (about 21 GB) from https://huggingface.co/{HF_REPO} into {}", dir.display());
    for f in missing {
        fetch_to_file(&format!("https://huggingface.co/{HF_REPO}/resolve/main/{f}"), &dir.join(&f))?;
    }
    Ok(dir)
}
