#![forbid(unsafe_code)]

//! CraftBio Ecosystem IPC Bridges:
//! Integrates BioCraft seamlessly with sibling binaries (`admetcraft-cli`, `chemcraft-cli`,
//! `dockcraft-cli`) and external 3D visualizers (`UCSF ChimeraX`).

use std::process::Command;

use serde::{Deserialize, Serialize};

/// Detected installed sibling CraftBio tools and 3D viewer binaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemStatus {
    pub chemcraft_cli: bool,
    pub admetcraft_cli: bool,
    pub dockcraft_cli: bool,
    pub chimerax: bool,
}

fn has_binary(bin: &str) -> bool {
    Command::new("which")
        .arg(bin)
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Detect available sibling CraftBio binaries and UCSF ChimeraX on PATH.
#[must_use]
pub fn detect_ecosystem() -> EcosystemStatus {
    EcosystemStatus {
        chemcraft_cli: has_binary("chemcraft-cli"),
        admetcraft_cli: has_binary("admetcraft-cli"),
        dockcraft_cli: has_binary("dockcraft-cli"),
        chimerax: has_binary("chimerax"),
    }
}

/// Invoke `admetcraft-cli eval --tpp oncology <SMILES>` to run a live ADMET/Tox/permeability
/// audit on an ADC cytotoxic payload (returns `None` gracefully if binary is absent or fails).
#[must_use]
pub fn audit_payload_with_admetcraft(smiles: &str) -> Option<String> {
    let out = Command::new("admetcraft-cli")
        .args(["eval", "--tpp", "oncology", smiles])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let val: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let formula = val
        .pointer("/physicochemical/formula")
        .and_then(serde_json::Value::as_str);
    let mw = val
        .pointer("/physicochemical/molecular_weight")
        .and_then(serde_json::Value::as_f64);
    let logp = val
        .pointer("/lipophilicity/consensus_log_p")
        .and_then(serde_json::Value::as_f64);
    let tpsa = val
        .pointer("/physicochemical/tpsa")
        .and_then(serde_json::Value::as_f64);
    let gi = val
        .pointer("/pharmacokinetics/gi_absorption")
        .and_then(serde_json::Value::as_str);
    let pgp = val
        .pointer("/pharmacokinetics/pgp_substrate")
        .and_then(serde_json::Value::as_bool);
    let qed = val
        .pointer("/druglikeness/qed")
        .and_then(serde_json::Value::as_f64);
    let mut parts = Vec::new();
    if let Some(f) = formula {
        parts.push(f.to_string());
    }
    if let Some(v) = mw {
        parts.push(format!("MW={v:.2}"));
    }
    if let Some(v) = logp {
        parts.push(format!("cLogP={v:.2}"));
    }
    if let Some(v) = tpsa {
        parts.push(format!("TPSA={v:.1}"));
    }
    if let Some(g) = gi {
        parts.push(format!("GI={g}"));
    }
    if let Some(p) = pgp {
        parts.push(format!("Pgp={p}"));
    }
    if let Some(q) = qed {
        parts.push(format!("QED={q:.2}"));
    }
    if parts.is_empty() {
        Some("AdmetCraft payload eval succeeded.".to_string())
    } else {
        Some(format!("AdmetCraft v1.2 live audit: {}", parts.join(" | ")))
    }
}

/// Spawn a sibling GUI/desktop application (`chemcraft`, `admetcraft`, `dockcraft`, `biomolcraft`, `chimerax`)
/// detached from BioCraft GUI without blocking the UI thread.
pub fn spawn_companion_app(bin: &str, args: &[&str]) -> Result<(), String> {
    Command::new(bin)
        .args(args)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("failed to launch '{bin}': {e}"))
}

/// Push a biologic PDB/mmCIF structure live into a running `BiomolCraft` studio over **Craft Link v1** (`craft-ipc`).
pub fn send_to_biomolcraft(pdb_path: &str, module_hint: Option<&str>) -> Result<String, String> {
    let reply = craftbio_scene::push_pdb_to_live_biomolcraft("biocraft", pdb_path, module_hint)
        .map_err(|e| e.to_string())?;
    Ok(format!("Craft Link v1 → BiomolCraft: {reply}"))
}
