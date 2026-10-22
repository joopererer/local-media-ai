//! 27.5 构建脚本：定位 llama.cpp 运行库（导入库）并链接。
//!
//! 查找顺序（首个存在者）：
//!   1) 环境变量 `FO_LLAMA_LIB_DIR`（打包脚本 / CI 设置的构建目录）
//!   2) `<repo>/app/build/llama`（本仓库内的默认输出目录）
//!
//! 找到时发出 `cfg(llama_linked)` 并链接 llama/mtmd/ggml；找不到时只告警，
//! 使 worker 可编译为"协议骨架"（帧协议可用、嵌入未链接），便于无运行库环境测试。
fn main() {
    println!("cargo:rerun-if-env-changed=FO_LLAMA_LIB_DIR");
    println!("cargo:rustc-check-cfg=cfg(llama_linked)");
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("FO_LLAMA_LIB_DIR") {
        dirs.push(d.into());
    }
    dirs.push(std::path::PathBuf::from("../../app/build/llama"));

    let mut found = None;
    for d in &dirs {
        if d.join("llama.lib").exists() || d.join("libllama.so").exists() {
            found = Some(d.clone());
            break;
        }
    }
    match found {
        Some(d) => {
            println!("cargo:rustc-cfg=llama_linked");
            println!("cargo:rustc-link-search=native={}", d.display());
            println!("cargo:rustc-link-lib=llama");
            println!("cargo:rustc-link-lib=mtmd");
            println!("cargo:rustc-link-lib=ggml");
            println!("cargo:rustc-env=FO_LLAMA_LIB_DIR={}", d.display());
        }
        None => {
            println!(
                "cargo:warning=未找到 llama.cpp 导入库（请设置 FO_LLAMA_LIB_DIR）；\
                 llama-embed-worker 将以骨架模式构建（帧协议可用，嵌入未链接）。"
            );
        }
    }
}
