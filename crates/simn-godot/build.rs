//! Build-time helper.
//!
//! `libsimn_godot.so` dynamically links against `libsteam_api.so`, which
//! the `steamworks-sys` build script places in its own `OUT_DIR`. At
//! runtime Godot loads `libsimn_godot.so` from `target/<profile>/`, and
//! the dynamic linker looks for `libsteam_api.so` next to it (via the
//! `$ORIGIN` rpath set below) — so we copy the redistributable there.
//!
//! Without this step, loading the extension fails with:
//!   `libsteam_api.so: cannot open shared object file`

use std::path::PathBuf;

fn main() {
    // Search `libsimn_godot.so`'s own directory for `libsteam_api.so`.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    // Find the steamworks-sys build output and copy the redistributable
    // next to our cdylib.
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    // OUT_DIR = target/<profile>/build/simn-godot-<hash>/out
    // Walk up to target/<profile>/build and scan for steamworks-sys-*/out.
    let build_dir = out_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("ancestor of OUT_DIR");
    let target_profile = build_dir
        .parent()
        .expect("target profile dir")
        .to_path_buf();

    let Ok(entries) = std::fs::read_dir(&build_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("steamworks-sys-") {
            continue;
        }
        let candidate = entry.path().join("out").join("libsteam_api.so");
        if candidate.exists() {
            let dest = target_profile.join("libsteam_api.so");
            let _ = std::fs::copy(&candidate, &dest);
            println!("cargo:rerun-if-changed={}", candidate.display());
            break;
        }
    }
}
