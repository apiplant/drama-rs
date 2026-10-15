use std::path::PathBuf;
use std::process::Command;

/// `DRAMA_CUDA_ARCH` is one arch (`sm_89`) or a comma-separated list (`sm_75,sm_80,sm_86,sm_89,sm_120`, for release
/// builds): machine code for each and no PTX. PTX in a release is JIT-compiled by the driver, and a driver older than
/// the toolkit refuses it: a card the build has no machine code for is unsupported, never a crash on an old driver.
fn gencode(arch: &str) -> Vec<String> {
    arch.split(',')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(|a| {
            let n = a.trim_start_matches("sm_");
            format!("-gencode=arch=compute_{n},code=sm_{n}")
        })
        .collect()
}

/// Writes `$OUT_DIR/ui.rs`: the built web UI (`ui/dist`, from `pnpm build` in `ui/`) as `(path, bytes)` pairs for
/// `drama serve` to embed. Without a build the list is empty and `/` says how to make one.
fn embed_ui(out: &std::path::Path) {
    fn walk(dir: &std::path::Path, root: &std::path::Path, files: &mut Vec<(String, PathBuf)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, files);
            } else {
                files.push((p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), p.canonicalize().unwrap()));
            }
        }
    }
    let root = PathBuf::from("ui/dist");
    println!("cargo:rerun-if-changed=ui/dist");
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    files.sort();
    println!("cargo:rerun-if-env-changed=DRAMA_REQUIRE_UI");
    if files.is_empty() && std::env::var_os("DRAMA_REQUIRE_UI").is_some_and(|v| v != "0") {
        panic!("DRAMA_REQUIRE_UI is set but ui/dist is empty: run `pnpm install && pnpm build` in ui/");
    }
    let mut src = String::from("&[\n");
    for (path, abs) in &files {
        println!("cargo:rerun-if-changed={}", abs.display());
        src += &format!("    ({path:?}, include_bytes!({:?}) as &[u8]),\n", abs.display().to_string());
    }
    src += "]\n";
    std::fs::write(out.join("ui.rs"), src).unwrap();
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let windows = std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".to_string());
    let arch = std::env::var("DRAMA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    embed_ui(&out);
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DRAMA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if windows {
        // MSVC: nvcc packs both kernels into one static library against the dynamic C runtime (/MD), as Rust links
        let lib = out.join("dramakernels.lib");
        let _ = std::fs::remove_file(&lib);
        let status = Command::new(format!("{cuda}/bin/nvcc"))
            .args(["-O3", "-std=c++17", "-lineinfo", "-Xcompiler", "/MD"])
            .args(gencode(&arch))
            .arg("-lib")
            .args(["kernels/drama.cu", "kernels/codec.cu"])
            .arg("-o")
            .arg(&lib)
            .status()
            .expect("failed to run nvcc");
        assert!(status.success(), "nvcc failed");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=dramakernels");
        println!("cargo:rustc-link-search=native={cuda}/lib/x64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        return;
    }

    let mut objs = Vec::new();
    for src in ["drama", "codec"] {
        let obj = out.join(format!("{src}.o"));
        let status = Command::new(format!("{cuda}/bin/nvcc"))
            .args(["-O3", "-std=c++17", "-lineinfo", "-Xcompiler", "-fPIC"])
            .args(gencode(&arch))
            .arg("-c")
            .arg(format!("kernels/{src}.cu"))
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to run nvcc");
        assert!(status.success(), "nvcc failed on {src}.cu");
        objs.push(obj);
    }
    let lib = out.join("libdramakernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=dramakernels");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
