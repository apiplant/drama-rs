use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".to_string());
    let arch = std::env::var("DRAMA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DRAMA_CUDA_ARCH");

    let mut objs = Vec::new();
    for src in ["drama", "codec"] {
        let obj = out.join(format!("{src}.o"));
        let status = Command::new(format!("{cuda}/bin/nvcc"))
            .args(["-O3", "-std=c++17", "-lineinfo", "-Xcompiler", "-fPIC"])
            .arg(format!("-arch={arch}"))
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
