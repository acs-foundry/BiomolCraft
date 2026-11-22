#![forbid(unsafe_code)]

//! `xtask` — BiomolCraft Architectural Guardrails (`storytold/craft-anything` discipline).
//!
//! Run `cargo xtask layers` to verify:
//! 1. Strict acyclic `L0 -> L1 -> L2 -> L3 -> L4 -> L5` crate dependency layering.
//! 2. `#![forbid(unsafe_code)]` at the top of every crate entry point (`lib.rs` / `main.rs`).
//! 3. Workspace panic-denial lints (`unwrap_used`, `expect_used`, `panic`, `unimplemented`,
//!    `todo`, `unreachable`) and `[lints] workspace = true` opt-in across every workspace crate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Clone, Copy)]
struct LayerSpec {
    name: &'static str,
    rel_dir: &'static str,
    layer: u8,
    role: &'static str,
}

const WORKSPACE_LAYERS: &[LayerSpec] = &[
    LayerSpec {
        name: "bio-core",
        rel_dir: "crates/bio-core",
        layer: 0,
        role: "L0 Foundation (Biologics Sequence & Antibody Core)",
    },
    LayerSpec {
        name: "bio-adc",
        rel_dir: "crates/bio-adc",
        layer: 0,
        role: "L0 Foundation (ADC Linker/Payload/DAR & TMDD PK)",
    },
    LayerSpec {
        name: "biomol-rna",
        rel_dir: "crates/biomol-rna",
        layer: 0,
        role: "L0 Foundation (Zuker MFE 2D RNA/mRNA/LNP Engine)",
    },
    LayerSpec {
        name: "biomol-enzyme",
        rel_dir: "crates/biomol-enzyme",
        layer: 0,
        role: "L0 Foundation (Catalytic Triads, Tunnels & Disulfides)",
    },
    LayerSpec {
        name: "biomol-model",
        rel_dir: "crates/biomol-model",
        layer: 0,
        role: "L0 Foundation (3D Mutagenesis, swapna, Sculpting & Builder)",
    },
    LayerSpec {
        name: "biomol-physics",
        rel_dir: "crates/biomol-physics",
        layer: 0,
        role: "L0 Foundation (ANM, GB/SA, Langevin MD & OpenMM Bridge)",
    },
    LayerSpec {
        name: "bio-develop",
        rel_dir: "crates/bio-develop",
        layer: 1,
        role: "L1 Analyzer (TAP/SAP Antibody Developability)",
    },
    LayerSpec {
        name: "bio-interface",
        rel_dir: "crates/bio-interface",
        layer: 1,
        role: "L1 Analyzer (Paratope/Epitope & Ala-Scan Interface)",
    },
    LayerSpec {
        name: "bio-engine",
        rel_dir: "crates/bio-engine",
        layer: 2,
        role: "L2 Composite Biologics Orchestrator",
    },
    LayerSpec {
        name: "biomol-modules",
        rel_dir: "crates/biomol-modules",
        layer: 3,
        role: "L3 Pluggable Domain Modules & WASM/JSON Registry",
    },
    LayerSpec {
        name: "biomol-script",
        rel_dir: "crates/biomol-script",
        layer: 4,
        role: "L4 Unified Typed Command Registry, DSL & .biomol.zip Session",
    },
    LayerSpec {
        name: "biomolcraft-cli",
        rel_dir: "apps/biomolcraft-cli",
        layer: 5,
        role: "L5 Headless CLI & Hardened MCP Server",
    },
    LayerSpec {
        name: "biomolcraft",
        rel_dir: "apps/biomolcraft",
        layer: 5,
        role: "L5 Desktop GPU GUI (wgpu 24 + egui)",
    },
];

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map_or(manifest_dir.clone(), Path::to_path_buf)
}

fn parse_dependencies_section(toml_text: &str) -> (Vec<String>, bool) {
    let mut deps = Vec::new();
    let mut in_deps = false;
    let mut in_lints = false;
    let mut has_workspace_lints = false;

    for raw in toml_text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let hdr = line.trim_start_matches('[').trim_end_matches(']').trim();
            in_deps = hdr == "dependencies" || hdr == "build-dependencies";
            in_lints = hdr == "lints";
            continue;
        }
        if in_lints && line.replace(' ', "") == "workspace=true" {
            has_workspace_lints = true;
        }
        if in_deps && let Some((lhs, _)) = line.split_once('=') {
            let key = lhs.trim();
            let dep_name = key.split('.').next().unwrap_or(key).trim();
            if !dep_name.is_empty() {
                deps.push(dep_name.to_string());
            }
        }
    }
    (deps, has_workspace_lints)
}

fn check_layers() -> Result<(), String> {
    let root = workspace_root();
    let root_toml = std::fs::read_to_string(root.join("Cargo.toml"))
        .map_err(|e| format!("failed to read root Cargo.toml: {e}"))?;

    let mut violations = Vec::new();
    for required in [
        "unsafe_code = \"forbid\"",
        "unwrap_used = \"deny\"",
        "expect_used = \"deny\"",
        "panic = \"deny\"",
        "unimplemented = \"deny\"",
        "todo = \"deny\"",
        "unreachable = \"deny\"",
    ] {
        if !root_toml.contains(required) {
            violations.push(format!("root Cargo.toml missing lint policy: `{required}`"));
        }
    }

    let layer_map: BTreeMap<&str, u8> =
        WORKSPACE_LAYERS.iter().map(|s| (s.name, s.layer)).collect();

    println!("# BiomolCraft Architecture Layer Verification (`cargo xtask layers`)\n");
    println!(
        "| Layer | Crate | Role | Internal Deps | `#![forbid(unsafe_code)]` | Workspace Lints | Status |"
    );
    println!("|---|---|---|---|---:|---:|---|");

    for spec in WORKSPACE_LAYERS {
        let crate_dir = root.join(spec.rel_dir);
        let cargo_toml_path = crate_dir.join("Cargo.toml");
        let toml_text = std::fs::read_to_string(&cargo_toml_path)
            .map_err(|e| format!("failed to read {}: {e}", cargo_toml_path.display()))?;
        let (deps, has_ws_lints) = parse_dependencies_section(&toml_text);

        let mut internal_deps = Vec::new();
        let mut crate_ok = true;

        for dep in &deps {
            if let Some(&dep_layer) = layer_map.get(dep.as_str()) {
                internal_deps.push(format!("{dep}(L{dep_layer})"));
                if dep_layer >= spec.layer {
                    crate_ok = false;
                    violations.push(format!(
                        "LAYER VIOLATION: '{}' (L{}) depends on '{}' (L{}) — dependencies must point strictly downward!",
                        spec.name, spec.layer, dep, dep_layer
                    ));
                }
            }
        }

        if !has_ws_lints {
            crate_ok = false;
            violations.push(format!(
                "LINT VIOLATION: '{}' missing `[lints] workspace = true` in {}",
                spec.name,
                cargo_toml_path.display()
            ));
        }

        let entry_path = if crate_dir.join("src/lib.rs").is_file() {
            crate_dir.join("src/lib.rs")
        } else {
            crate_dir.join("src/main.rs")
        };
        let entry_src = std::fs::read_to_string(&entry_path)
            .map_err(|e| format!("failed to read {}: {e}", entry_path.display()))?;
        let has_forbid_unsafe = entry_src
            .lines()
            .take(5)
            .any(|l| l.trim() == "#![forbid(unsafe_code)]");
        if !has_forbid_unsafe {
            crate_ok = false;
            violations.push(format!(
                "SAFETY VIOLATION: '{}' missing `#![forbid(unsafe_code)]` in {}",
                spec.name,
                entry_path.display()
            ));
        }

        let deps_str = if internal_deps.is_empty() {
            "—".to_string()
        } else {
            internal_deps.join(", ")
        };
        println!(
            "| L{} | `{}` | {} | {} | {} | {} | {} |",
            spec.layer,
            spec.name,
            spec.role,
            deps_str,
            if has_forbid_unsafe { "PASS" } else { "FAIL" },
            if has_ws_lints { "PASS" } else { "FAIL" },
            if crate_ok { "PASS" } else { "FAIL" }
        );
    }

    println!();
    if violations.is_empty() {
        println!(
            "PASS: {} crates verified across L0..L5 (0 layer violations, 0 unsafe_code, 100% panic-denial workspace lints).",
            WORKSPACE_LAYERS.len()
        );
        Ok(())
    } else {
        for v in &violations {
            eprintln!("ERROR: {v}");
        }
        Err(format!(
            "{} architectural guardrail violation(s) detected",
            violations.len()
        ))
    }
}

fn main() -> ExitCode {
    let arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "layers".to_string());
    let res = match arg.as_str() {
        "layers" | "check" => check_layers(),
        "help" | "--help" | "-h" => {
            println!(
                "BiomolCraft xtask commands:\n  cargo xtask layers   Verify L0->L5 acyclic crate layering, zero-unsafe & panic-denial lints"
            );
            Ok(())
        }
        other => Err(format!(
            "unknown xtask subcommand '{other}' (try `cargo xtask layers`)"
        )),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask failed: {e}");
            ExitCode::FAILURE
        }
    }
}
