#![forbid(unsafe_code)]

//! Hardened MCP Server for `BiomolCraft` built directly on `storytold/craft-libs::craft-mcp-server`
//! and kernel-confined `craftbio-sandbox::PathSandbox` (including live base64 PNG 3D viewport rendering).

use std::path::Path;

use biomol_rna::{analyze_ncrna_mrna, find_rna_preset};
use biomol_script::BiomolSession;
use craft_mcp_server::{
    Effect, Info, Options, Server, Tools, image_result, object, string, text_result, tool,
};
use craftbio_report::write_biomol_html_dossier;
use craftbio_sandbox::PathSandbox;
use craftbio_struct::{write_mmcif, write_pdb};
use craftbio_viewport::render_scene_png_bytes;
use serde_json::{Value, json};

pub struct BiomolTools {
    sandbox: PathSandbox,
    session: BiomolSession,
}

impl BiomolTools {
    #[must_use]
    pub fn new(sandbox: PathSandbox) -> Self {
        Self {
            sandbox,
            session: BiomolSession::default(),
        }
    }
}

impl Tools for BiomolTools {
    fn instructions(&self) -> String {
        "BiomolCraft MCP Server (CraftBio Suite, aligned with storytold/craft-mcp-server): load PDB/mmCIF biomolecular structures (proteins, antibodies/VHH, ncRNA/mRNA, enzymes), run PyMOL/ChimeraX-compatible DSL commands, perform Dunbrack 3D mutagenesis & local MM minimization, activate modular biomolecular analyzers, fold ncRNA/mRNA with Zuker MFE, and render PNG views."
            .to_string()
    }

    fn list(&mut self) -> Vec<Value> {
        vec![
            tool(
                "load_biomolecule",
                "Load 3D Biomolecule",
                "Load a PDB or mmCIF biomolecular structure (protein, antibody, enzyme, ncRNA/mRNA complex) from inside the sandbox read root.",
                object(
                    json!({ "path": string("Relative or sandbox-confined path to .pdb or .cif file") }),
                    &["path"],
                ),
                Effect::Replaces,
            ),
            tool(
                "run_dsl",
                "Run BiomolCraft DSL",
                "Execute one or more semicolon-separated PyMOL/ChimeraX-compatible commands (select, style, color, surface, align, mutate, minimize, module, undo, redo).",
                object(
                    json!({ "command": string("DSL command string, e.g. 'style cartoon; color esp; module antibodies'") }),
                    &["command"],
                ),
                Effect::Edits,
            ),
            tool(
                "mutate_residue_3d",
                "3D Rotamer Mutagenesis",
                "Mutate a protein residue in 3D using Dunbrack rotamers with clash & H-bond evaluation.",
                object(
                    json!({
                        "chain": string("Single-character chain ID, e.g. 'B'"),
                        "res_seq": { "type": "integer", "description": "Residue sequence number" },
                        "target_aa": string("Target 1-letter amino acid code, e.g. 'A' or 'W'"),
                        "rotamer": { "type": "integer", "description": "Optional rotamer index (omit for automatic clash-minimum repacking)" }
                    }),
                    &["chain", "res_seq", "target_aa"],
                ),
                Effect::Edits,
            ),
            tool(
                "minimize_local",
                "Local MM Minimization",
                "Relax steric clashes in the active selection (or whole model) with pure-Rust ISOLDE-lite local MM minimization.",
                object(
                    json!({ "steps": { "type": "integer", "description": "Steepest-descent steps (default 80, max 300)" } }),
                    &[],
                ),
                Effect::Edits,
            ),
            tool(
                "run_biomol_module",
                "Run Biomolecular Module",
                "Execute a BiomolCraft design module on the active 3D structure and apply its 3D visual overlays (antibodies, ncrna, enzymes, adc).",
                object(
                    json!({ "module": string("Module key: antibodies | ncrna | enzymes | adc") }),
                    &["module"],
                ),
                Effect::ReadOnly,
            ),
            tool(
                "fold_ncrna_mrna",
                "Fold & Audit ncRNA/mRNA",
                "Run deterministic Zuker/Turner-2004 MFE 2D RNA folding, G-quadruplex & TLR7/8/m1Psi audit, and LNP formulation calculator.",
                object(
                    json!({
                        "name": string("Optional RNA construct name"),
                        "preset": string("Optional preset ID (guanine_riboswitch, crispr_sgrna, bnt162b2_5utr, patisiran_sirna, g4_terra)"),
                        "sequence": string("Raw RNA/DNA sequence (A/U/G/C/T)")
                    }),
                    &[],
                ),
                Effect::ReadOnly,
            ),
            tool(
                "render_view_png",
                "Render 3D Viewport PNG",
                "Render the current 3D biomolecular scene headlessly and return a PNG image directly to the MCP client.",
                object(
                    json!({
                        "width": { "type": "integer", "description": "Width in pixels (default 800)" },
                        "height": { "type": "integer", "description": "Height in pixels (default 600)" },
                        "transparent": { "type": "boolean", "description": "Transparent PNG background" }
                    }),
                    &[],
                ),
                Effect::ReadOnly,
            ),
            tool(
                "save_structure_or_scene",
                "Save PDB or CraftScene",
                "Write the current (possibly mutated/minimized) structure as .pdb, .cif, or .craftscene.json inside the sandbox write root.",
                object(
                    json!({ "out_path": string("Output filename inside sandbox write root (.pdb, .cif, or .craftscene.json)") }),
                    &["out_path"],
                ),
                Effect::Files,
            ),
            tool(
                "run_anm_modes",
                "ANM Normal Mode Analysis",
                "Compute the 6 lowest collective Anisotropic Network Model (ANM) normal modes, hinge residues, and generate animated 3D trajectory frames.",
                object(
                    json!({
                        "mode": { "type": "integer", "description": "Mode index 1..6 (default 1)" },
                        "amplitude_a": { "type": "number", "description": "Max displacement amplitude in Angstroms (default 2.2)" }
                    }),
                    &[],
                ),
                Effect::Edits,
            ),
            tool(
                "run_gbsa_physics",
                "GB/SA Solvation & Dipole",
                "Compute pure-Rust Still GB/SA implicit solvation energy, 3D electrostatic dipole moment (Debye), and single-trajectory interface ΔG_bind.",
                object(json!({}), &[]),
                Effect::ReadOnly,
            ),
            tool(
                "run_enzyme_engineering",
                "Enzyme Triads, Tunnels & Disulfides",
                "Detect catalytic triads/dyads, metal/cofactor coordination spheres, 3D substrate access tunnels, and thermostabilizing disulfide bridge proposals.",
                object(json!({}), &[]),
                Effect::ReadOnly,
            ),
            tool(
                "simulate_cryoem_map",
                "Cryo-EM MolMap & FitMap",
                "Synthesize a resolution-filtered 3D Cryo-EM density map (ChimeraX molmap), display volumetric contour beads at sigma_level, and report fitmap correlation.",
                object(
                    json!({
                        "resolution_a": { "type": "number", "description": "Map resolution in Angstroms (default 4.5)" },
                        "sigma": { "type": "number", "description": "Contour level in sigma (default 1.5)" }
                    }),
                    &[],
                ),
                Effect::Edits,
            ),
            tool(
                "stage_denovo_binder",
                "De Novo & Co-Fold Epitope Staging",
                "Stage a zero-weight RFdiffusion + ProteinMPNN + ESM2 + Boltz-2/Chai-1 de novo binder campaign targeting 3D epitope hotspots on target_chain.",
                object(
                    json!({
                        "chain": string("Target antigen chain ID, e.g. 'C'"),
                        "hotspots": string("Optional space-separated hotspot residues, e.g. 'C558 C560 C573'")
                    }),
                    &["chain"],
                ),
                Effect::ReadOnly,
            ),
            tool(
                "export_offline_html_dossier",
                "Export Offline Mol* 3D HTML Dossier",
                "Write a self-contained single-file interactive 3D HTML biomolecular dossier inside the sandbox write root.",
                object(
                    json!({ "out_path": string("Output HTML filename inside sandbox write root") }),
                    &["out_path"],
                ),
                Effect::Files,
            ),
        ]
    }

    fn call(&mut self, name: &str, args: &Value) -> Value {
        match name {
            "load_biomolecule" => {
                let p = args.get("path").and_then(Value::as_str).unwrap_or("");
                match self.sandbox.read_string(Path::new(p)) {
                    Ok(raw) => {
                        let id = Path::new(p)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("biomol");
                        match self
                            .session
                            .load_structure_text(id, &raw, Some(p.to_string()))
                        {
                            Ok(msg) => text_result(msg, false),
                            Err(e) => text_result(e.to_string(), true),
                        }
                    }
                    Err(e) => text_result(format!("sandbox read rejected: {e}"), true),
                }
            }
            "run_dsl" => {
                let cmd = args.get("command").and_then(Value::as_str).unwrap_or("");
                // Reject raw file-system open/save/png inside generic DSL to preserve sandbox confinement
                let lower = cmd.to_ascii_lowercase();
                if lower.contains("open ")
                    || lower.contains("load ")
                    || lower.contains("save ")
                    || lower.contains("png ")
                {
                    return text_result(
                        "use dedicated sandboxed tools `load_biomolecule`, `save_structure_or_scene`, or `render_view_png` for file I/O",
                        true,
                    );
                }
                match self.session.execute_dsl(cmd) {
                    Ok(out) => text_result(out, false),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "mutate_residue_3d" => {
                let ch = args.get("chain").and_then(Value::as_str).unwrap_or("A");
                let rseq = args.get("res_seq").and_then(Value::as_i64).unwrap_or(1);
                let aa = args.get("target_aa").and_then(Value::as_str).unwrap_or("A");
                let cmd = if let Some(rot) = args.get("rotamer").and_then(Value::as_u64) {
                    format!("mutate {ch}:{rseq} {aa} {rot}")
                } else {
                    format!("mutate {ch}:{rseq} {aa}")
                };
                match self.session.execute_dsl(&cmd) {
                    Ok(msg) => text_result(msg, false),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "minimize_local" => {
                let steps = args.get("steps").and_then(Value::as_u64).unwrap_or(80);
                match self.session.execute_dsl(&format!("minimize {steps}")) {
                    Ok(msg) => text_result(msg, false),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "run_biomol_module" => {
                let m = args
                    .get("module")
                    .and_then(Value::as_str)
                    .unwrap_or("antibodies");
                match self.session.execute_dsl(&format!("module {m}")) {
                    Ok(msg) => text_result(
                        format!("{msg}\n\n{}", self.session.last_module_markdown),
                        false,
                    ),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "fold_ncrna_mrna" => {
                let (label, seq) = if let Some(pid) = args.get("preset").and_then(Value::as_str) {
                    match find_rna_preset(pid) {
                        Some(p) => (p.name.to_string(), p.sequence.to_string()),
                        None => return text_result(format!("unknown RNA preset '{pid}'"), true),
                    }
                } else if let Some(s) = args.get("sequence").and_then(Value::as_str) {
                    (
                        args.get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("Custom_RNA")
                            .to_string(),
                        s.to_string(),
                    )
                } else {
                    return text_result("provide either 'preset' or 'sequence'", true);
                };
                match analyze_ncrna_mrna(&label, "ncRNA/mRNA", &seq, None) {
                    Ok(rep) => {
                        let json_txt =
                            serde_json::to_string_pretty(&rep).unwrap_or_else(|_| "{}".to_string());
                        text_result(json_txt, false)
                    }
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "render_view_png" => {
                let w = args
                    .get("width")
                    .and_then(Value::as_u64)
                    .unwrap_or(800)
                    .clamp(64, 2048) as u32;
                let h = args
                    .get("height")
                    .and_then(Value::as_u64)
                    .unwrap_or(600)
                    .clamp(64, 2048) as u32;
                let transp = args
                    .get("transparent")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let scene = self.session.build_viewport_scene();
                match render_scene_png_bytes(&scene, &self.session.camera, w, h, transp) {
                    Ok(png_bytes) => image_result(
                        &png_bytes,
                        "image/png",
                        format!(
                            "Rendered {w}x{h} BiomolCraft 3D scene ({} bytes).",
                            png_bytes.len()
                        ),
                    ),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "save_structure_or_scene" => {
                let out = args.get("out_path").and_then(Value::as_str).unwrap_or("");
                let content = if out.ends_with(".json") {
                    match serde_json::to_string_pretty(&self.session.to_craftscene_document()) {
                        Ok(s) => s,
                        Err(e) => return text_result(e.to_string(), true),
                    }
                } else if let Some(ref mol) = self.session.molecule {
                    if out.ends_with(".cif") || out.ends_with(".mmcif") {
                        write_mmcif(mol)
                    } else {
                        write_pdb(mol)
                    }
                } else {
                    return text_result("no biomolecule loaded to save", true);
                };
                match self.sandbox.write_string(Path::new(out), &content) {
                    Ok(path) => text_result(format!("Saved to {}", path.display()), false),
                    Err(e) => text_result(format!("sandbox write rejected: {e}"), true),
                }
            }
            "run_anm_modes" => {
                let m = args.get("mode").and_then(Value::as_u64).unwrap_or(1);
                let amp = args
                    .get("amplitude_a")
                    .and_then(Value::as_f64)
                    .unwrap_or(2.2);
                match self.session.execute_dsl(&format!("anm {m} {amp:.2}")) {
                    Ok(msg) => text_result(
                        format!("{msg}\n\n{}", self.session.last_module_markdown),
                        false,
                    ),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "run_gbsa_physics" => match self.session.execute_dsl("gbsa") {
                Ok(msg) => text_result(
                    format!("{msg}\n\n{}", self.session.last_module_markdown),
                    false,
                ),
                Err(e) => text_result(e.to_string(), true),
            },
            "run_enzyme_engineering" => match self.session.execute_dsl("tunnels") {
                Ok(msg) => text_result(
                    format!("{msg}\n\n{}", self.session.last_module_markdown),
                    false,
                ),
                Err(e) => text_result(e.to_string(), true),
            },
            "simulate_cryoem_map" => {
                let res = args
                    .get("resolution_a")
                    .and_then(Value::as_f64)
                    .unwrap_or(4.5);
                let sig = args.get("sigma").and_then(Value::as_f64).unwrap_or(1.5);
                match self
                    .session
                    .execute_dsl(&format!("molmap {res:.1} {sig:.1}"))
                {
                    Ok(msg) => text_result(
                        format!("{msg}\n\n{}", self.session.last_module_markdown),
                        false,
                    ),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "stage_denovo_binder" => {
                let ch = args.get("chain").and_then(Value::as_str).unwrap_or("C");
                let hs = args.get("hotspots").and_then(Value::as_str).unwrap_or("");
                let cmd = if hs.trim().is_empty() {
                    format!("denovo {ch}")
                } else {
                    format!("denovo {ch} {hs}")
                };
                match self.session.execute_dsl(&cmd) {
                    Ok(msg) => text_result(
                        format!("{msg}\n\n{}", self.session.last_module_markdown),
                        false,
                    ),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "export_offline_html_dossier" => {
                let out = args.get("out_path").and_then(Value::as_str).unwrap_or("");
                let Some(ref mol) = self.session.molecule else {
                    return text_result("no biomolecule loaded", true);
                };
                // Enforce PathSandbox write confinement by creating placeholder inside sandbox first
                match self.sandbox.write_string(Path::new(out), "") {
                    Ok(safe_path) => match write_biomol_html_dossier(
                        &mol.title,
                        &write_pdb(mol),
                        &self.session.last_module_markdown,
                        &[],
                        true,
                        &safe_path,
                    ) {
                        Ok(sum) => text_result(
                            format!(
                                "Wrote offline 3D HTML dossier to {} ({} bytes, molstar_inline_offline={}).",
                                sum.out_path, sum.bytes, sum.molstar_inline_offline
                            ),
                            false,
                        ),
                        Err(e) => text_result(e, true),
                    },
                    Err(e) => text_result(format!("sandbox write rejected: {e}"), true),
                }
            }
            other => text_result(format!("unknown tool '{other}'"), true),
        }
    }
}

/// Start the BiomolCraft MCP server over stdio using `storytold/craft-libs::craft-mcp-server`.
pub fn run_mcp_stdio(sandbox: PathSandbox) -> Result<(), String> {
    let info = Info {
        name: "biomolcraft-mcp",
        title: "BiomolCraft MCP Server",
        version: env!("CARGO_PKG_VERSION"),
        max_line: 8 * 1024 * 1024,
    };
    let mut srv = Server::new(info, BiomolTools::new(sandbox)).with_options(Options {
        tool_errors_as_results: true,
        strict_request_ids: true,
        ..Options::default()
    });
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    srv.serve(stdin.lock(), stdout.lock())
        .map_err(|e| format!("MCP stdio error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use craftbio_sandbox::DEFAULT_MAX_FILE_BYTES;

    #[test]
    fn craft_mcp_server_lists_8_tools_folds_rna_and_blocks_traversal()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join(format!("biomol_mcp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base)?;
        let sbx = PathSandbox::new(
            Some(base.clone()),
            Some(base.clone()),
            DEFAULT_MAX_FILE_BYTES,
        )?;
        let info = Info {
            name: "biomolcraft-mcp",
            title: "BiomolCraft MCP",
            version: "0.1.0",
            max_line: 1024 * 1024,
        };
        let mut srv = Server::new(info, BiomolTools::new(sbx)).with_options(Options {
            tool_errors_as_results: true,
            ..Options::default()
        });

        let init = srv
            .handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#)
            .ok_or("no init reply")?;
        assert!(init.contains("biomolcraft-mcp"));

        let list = srv
            .handle_line(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#)
            .ok_or("no tools/list reply")?;
        let v: Value = serde_json::from_str(&list)?;
        let tools = v
            .pointer("/result/tools")
            .and_then(Value::as_array)
            .ok_or("no tools")?;
        assert_eq!(tools.len(), 14);

        let rna = srv
            .handle_line(r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fold_ncrna_mrna","arguments":{"preset":"guanine_riboswitch"}}}"#)
            .ok_or("no rna reply")?;
        assert!(rna.contains("dot_bracket"));

        let esc = srv
            .handle_line(r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"load_biomolecule","arguments":{"path":"../../etc/passwd"}}}"#)
            .ok_or("no escape reply")?;
        assert!(esc.contains("\"isError\":true"));

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}
