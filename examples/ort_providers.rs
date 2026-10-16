//! 27.2b：查询当前 ORT 运行库实际可用的执行提供器（EP）。
//! 用法：ort_providers --ort_lib <dll>
//! DirectML 是否可用直接决定 ORT+DirectML 这条路线是否成立。

use std::ffi::CStr;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let lib = args
        .iter()
        .position(|a| a == "--ort_lib")
        .and_then(|i| args.get(i + 1))
        .expect("--ort_lib");
    embed_core::init_ort(&PathBuf::from(lib))?;

    let api = ort::api();
    let mut ptrs: *mut *mut std::os::raw::c_char = std::ptr::null_mut();
    let mut n: i32 = 0;
    let status = unsafe { (api.GetAvailableProviders)(&mut ptrs, &mut n) };
    if !status.0.is_null() {
        return Err("GetAvailableProviders 返回错误状态".into());
    }
    let mut names = Vec::new();
    for i in 0..n {
        let p = unsafe { *ptrs.offset(i as isize) };
        names.push(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
    }
    println!("{}", serde_json::to_string_pretty(&names)?);
    Ok(())
}
