fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
    println!("cargo:rerun-if-changed=src/model/pt_layers.cpp");
    if std::env::var("CARGO_FEATURE_TCH_MODEL").is_err() {
        return;
    }

    let libtorch = std::env::var("LIBTORCH").unwrap_or_else(|_| {
        panic!("LIBTORCH must be set when building relayrl_types with tch-model")
    });
    let libtorch = std::path::PathBuf::from(libtorch);
    let include = libtorch.join("include");
    let api_include = include.join("torch/csrc/api/include");
    let lib_dir = libtorch.join("lib");
    let abi = std::env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_string());

    cc::Build::new()
        .cpp(true)
        .pic(true)
        .warnings(false)
        .include(&include)
        .include(&api_include)
        .flag("-std=c++17")
        .flag(format!("-D_GLIBCXX_USE_CXX11_ABI={abi}"))
        .flag("-DGLOG_USE_GLOG_EXPORT")
        .file("src/model/pt_layers.cpp")
        .compile("relayrl_pt_layers");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10");
}
