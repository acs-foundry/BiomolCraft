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
            tool(
                "load_dockcraft_bundle",
                "Load DockCraft ViewDockX Bundle",
                "Load a DockCraft ViewDockX JSON bundle (receptor, poses SDF, HYDE coronas, torsion lights, pharmacophores, ΔMIF, GridBox) from the sandbox read root.",
                object(
                    json!({
                        "path": string("Relative or sandbox-confined path to .dockbundle.json"),
                        "pose": { "type": "integer", "description": "Optional 0-based pose index to activate (default 0)" }
                    }),
                    &["path"],
                ),
                Effect::Replaces,
            ),
            tool(
                "export_mesh_3d",
                "Export 3D Mesh (.glb / .obj)",
                "Export the active triangulated Cartoon/SES/Cryo-EM scene to binary glTF 2.0 (.glb) or Wavefront (.obj) inside the sandbox write root.",
                object(
                    json!({ "out_path": string("Output filename (.glb or .obj) inside sandbox write root") }),
                    &["out_path"],
                ),
                Effect::Files,
            ),
            tool(
                "manage_scene",
                "Manage BiomolCraft Scenes (F1-F12)",
                "Save, recall, list, or delete named / F1-F12 multi-model camera & styling scenes.",
                object(
                    json!({
                        "action": string("save | recall | list | delete"),
                        "name": string("Optional scene name, e.g. 'f1' or 'active_site'")
                    }),
                    &["action"],
                ),
                Effect::Edits,
            ),
            tool(
                "analyze_interactions",
                "Chemical H-Bonds, Clashes, Contacts & Putty",
                "Detect geometric hydrogen bonds, VDW steric clashes, and inter-chain contacts on the active structure, optionally enabling B-factor Putty, Color Key, and Scale Bar.",
                object(
                    json!({
                        "max_dist_a": { "type": "number", "description": "H-bond cutoff in Angstroms (default 3.5)" },
                        "clash_overlap_a": { "type": "number", "description": "Steric clash VDW overlap cutoff in Angstroms (default 0.60)" },
                        "putty": { "type": "boolean", "description": "Switch active model to B-factor/pLDDT Putty mode" }
                    }),
                    &[],
                ),
                Effect::Edits,
            ),
            tool(
                "cealign_models",
                "Sequence-Independent 3D CEAlign",
                "Load an optional second PDB/mmCIF from the sandbox and perform sequence-independent Combinatorial Extension 3D superposition (works at 0% sequence identity).",
                object(
                    json!({
                        "mobile_path": string("Optional second PDB/mmCIF path inside sandbox read root to add as mobile model"),
                        "mobile_chain": string("Mobile chain ID (default 'A')"),
                        "target_chain": string("Target chain ID on #1 (default 'A')")
                    }),
                    &[],
                ),
                Effect::Edits,
            ),
            tool(
                "morph_structures",
                "Conformational Morph Trajectory",
                "Generate a smooth SHAKE-relaxed conformational morph trajectory between active #1 and a second conformation PDB/mmCIF inside the sandbox read root.",
                object(
                    json!({
                        "target_path": string("Second conformation PDB/mmCIF inside sandbox read root"),
                        "frames": { "type": "integer", "description": "Number of morph frames (default 16, range 4..60)" }
                    }),
                    &["target_path"],
                ),
                Effect::Edits,
            ),
            tool(
                "render_movie",
                "Render Animated Biomolecular Movie (.gif / .mp4)",
                "Render a smooth turntable, presentation rock, scene fly-through, or morph trajectory movie into the sandbox write root (pure-Rust GIF89a or H.264 MP4).",
                object(
                    json!({
                        "mode": string("turn | rock | fly | morph"),
                        "out_path": string("Output filename (.gif or .mp4) inside sandbox write root"),
                        "frames": { "type": "integer", "description": "Frame count (default 16, max 60)" },
                        "width": { "type": "integer", "description": "Width in px (default 480, max 1280)" },
                        "height": { "type": "integer", "description": "Height in px (default 360, max 960)" },
                        "fps": { "type": "integer", "description": "Frames per second (default 12)" }
                    }),
                    &["mode", "out_path"],
                ),
                Effect::Files,
            ),
            tool(
                "swapna_nucleotide",
                "3D RNA/DNA Nucleotide Mutagenesis (swapna)",
                "Mutate an RNA or DNA nucleotide in 3D preserving ribose-phosphate backbone and planar base orientation, reporting clashes and 3D base pairs.",
                object(
                    json!({
                        "chain": string("Nucleic acid chain ID, e.g. 'X'"),
                        "res_seq": { "type": "integer", "description": "Nucleotide residue sequence number" },
                        "target_base": string("Target base: A | U | G | C | T")
                    }),
                    &["chain", "res_seq", "target_base"],
                ),
                Effect::Edits,
            ),
            tool(
                "build_structure_3d",
                "De Novo 3D Peptide / RNA Duplex Builder",
                "Build a de novo 3D alpha-helix, beta-sheet, extended peptide, or A-form RNA duplex and add it to the active multi-model scene.",
                object(
                    json!({
                        "kind": string("helix | sheet | extended | rna"),
                        "sequence": string("1-letter amino acid or RNA sequence"),
                        "chain": string("Optional chain ID (default 'A')")
                    }),
                    &["kind", "sequence"],
                ),
                Effect::Edits,
            ),
            tool(
                "symmetry_and_glycans",
                "Symmetry (BIOMT/CRYST1) & 3D SNFG Glycans",
                "Generate Biological Assemblies (sym), crystallographic symmetry mates (symexp), toggle Unit Cell wireframe, or detect & display 3D SNFG Glycans.",
                object(
                    json!({
                        "action": string("sym | symexp | crystalcontacts | unitcell | snfg"),
                        "cutoff_a": { "type": "number", "description": "Optional distance cutoff for symexp/crystalcontacts (default 5.0 A)" }
                    }),
                    &["action"],
                ),
                Effect::Edits,
            ),
            tool(
                "volume_operations_vop",
                "Cryo-EM Volumetric Operations (vop & Orthoplanes)",
                "Apply vop gaussian, sharpen, zone (carve around site), hide_dust, or toggle XY/XZ/YZ Orthoplanes on the active volumetric map.",
                object(
                    json!({
                        "op": string("gaussian | sharpen | zone | hide_dust | orthoplanes"),
                        "param": { "type": "number", "description": "Sigma/amount/radius/min_voxels parameter (default 1.5)" }
                    }),
                    &["op"],
                ),
                Effect::Edits,
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
                } else if let Some(mol) = self.session.active_molecule() {
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
                let Some(mol) = self.session.active_molecule() else {
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
            "load_dockcraft_bundle" => {
                let p = args.get("path").and_then(Value::as_str).unwrap_or("");
                match self.sandbox.read_string(Path::new(p)) {
                    Ok(raw) => {
                        match serde_json::from_str::<craftbio_scene::DockCraftSceneBundle>(&raw) {
                            Ok(bundle) => match self.session.load_dockcraft_bundle(bundle) {
                                Ok(mut msg) => {
                                    if let Some(pose_idx) = args.get("pose").and_then(Value::as_u64)
                                        && pose_idx > 0
                                        && let Ok(pm) =
                                            self.session.select_dock_pose(pose_idx as usize)
                                    {
                                        msg.push('\n');
                                        msg.push_str(&pm);
                                    }
                                    text_result(msg, false)
                                }
                                Err(e) => text_result(e.to_string(), true),
                            },
                            Err(e) => {
                                text_result(format!("invalid DockCraftSceneBundle JSON: {e}"), true)
                            }
                        }
                    }
                    Err(e) => text_result(format!("sandbox read rejected: {e}"), true),
                }
            }
            "export_mesh_3d" => {
                let out = args.get("out_path").and_then(Value::as_str).unwrap_or("");
                let scene = self.session.build_viewport_scene();
                let title = self
                    .session
                    .active_molecule()
                    .map_or_else(|| "BiomolCraft".to_string(), |m| m.title.clone());
                let res = if out.to_ascii_lowercase().ends_with(".obj") {
                    let obj = scene.export_obj(&title);
                    self.sandbox
                        .write_string(Path::new(out), &obj)
                        .map(|p| (p, obj.len()))
                } else {
                    let glb = scene.export_glb(&title);
                    self.sandbox
                        .write_bytes(Path::new(out), &glb)
                        .map(|p| (p, glb.len()))
                };
                match res {
                    Ok((p, bytes)) => text_result(
                        format!("Exported 3D mesh to {} ({bytes} bytes).", p.display()),
                        false,
                    ),
                    Err(e) => text_result(format!("sandbox write rejected: {e}"), true),
                }
            }
            "manage_scene" => {
                let act = args.get("action").and_then(Value::as_str).unwrap_or("list");
                let nm = args.get("name").and_then(Value::as_str).unwrap_or("f1");
                let cmd = if act.eq_ignore_ascii_case("list") {
                    "scene list".to_string()
                } else {
                    format!("scene {act} {nm}")
                };
                match self.session.execute_dsl(&cmd) {
                    Ok(msg) => text_result(msg, false),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "analyze_interactions" => {
                let max_d = args
                    .get("max_dist_a")
                    .and_then(Value::as_f64)
                    .unwrap_or(3.5);
                let ov = args
                    .get("clash_overlap_a")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.60);
                let putty = args.get("putty").and_then(Value::as_bool).unwrap_or(false);
                let mut cmd = format!("hbonds {max_d:.2}; clashes {ov:.2}; contacts 4.0");
                if putty {
                    cmd.push_str("; putty on; key plddt; scalebar 20");
                }
                match self.session.execute_dsl(&cmd) {
                    Ok(msg) => text_result(
                        format!("{msg}\n\n{}", self.session.last_module_markdown),
                        false,
                    ),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "cealign_models" => {
                let mc = args
                    .get("mobile_chain")
                    .and_then(Value::as_str)
                    .unwrap_or("A");
                let tc = args
                    .get("target_chain")
                    .and_then(Value::as_str)
                    .unwrap_or("A");
                if let Some(p) = args.get("mobile_path").and_then(Value::as_str)
                    && !p.is_empty()
                {
                    match self.sandbox.read_string(Path::new(p)) {
                        Ok(raw) => {
                            let id = Path::new(p)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("mobile");
                            if let Err(e) =
                                self.session
                                    .add_structure_text(id, &raw, Some(p.to_string()))
                            {
                                return text_result(e.to_string(), true);
                            }
                        }
                        Err(e) => return text_result(format!("sandbox read rejected: {e}"), true),
                    }
                }
                let cmd = if self.session.models.len() >= 2 {
                    let mob_id = self.session.models.last().map_or(2, |m| m.model_id);
                    let ref_id = self.session.models[0].model_id;
                    format!("cealign #{mob_id}:{mc} #{ref_id}:{tc}")
                } else {
                    format!("cealign {mc} {tc}")
                };
                match self.session.execute_dsl(&cmd) {
                    Ok(msg) => text_result(msg, false),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "morph_structures" => {
                let p = args
                    .get("target_path")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let frames = args
                    .get("frames")
                    .and_then(Value::as_u64)
                    .unwrap_or(16)
                    .clamp(4, 60);
                match self.sandbox.read_string(Path::new(p)) {
                    Ok(raw) => {
                        let id = Path::new(p)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("target");
                        if let Err(e) =
                            self.session
                                .add_structure_text(id, &raw, Some(p.to_string()))
                        {
                            return text_result(e.to_string(), true);
                        }
                        let a_id = self.session.models[0].model_id;
                        let b_id = self.session.models.last().map_or(2, |m| m.model_id);
                        match self
                            .session
                            .execute_dsl(&format!("morph #{a_id} #{b_id} {frames}"))
                        {
                            Ok(msg) => text_result(msg, false),
                            Err(e) => text_result(e.to_string(), true),
                        }
                    }
                    Err(e) => text_result(format!("sandbox read rejected: {e}"), true),
                }
            }
            "render_movie" => {
                let mode = args.get("mode").and_then(Value::as_str).unwrap_or("turn");
                let out = args.get("out_path").and_then(Value::as_str).unwrap_or("");
                let frames = args
                    .get("frames")
                    .and_then(Value::as_u64)
                    .unwrap_or(16)
                    .clamp(4, 60);
                let w = args
                    .get("width")
                    .and_then(Value::as_u64)
                    .unwrap_or(480)
                    .clamp(64, 1280);
                let h = args
                    .get("height")
                    .and_then(Value::as_u64)
                    .unwrap_or(360)
                    .clamp(64, 960);
                let fps = args
                    .get("fps")
                    .and_then(Value::as_u64)
                    .unwrap_or(12)
                    .clamp(2, 30);
                // Enforce sandbox write confinement via zero-byte preflight creation
                match self.sandbox.write_bytes(Path::new(out), &[]) {
                    Ok(safe_path) => {
                        let cmd = format!(
                            "movie {mode} {} {frames} {w} {h} {fps}",
                            safe_path.display()
                        );
                        match self.session.execute_dsl(&cmd) {
                            Ok(msg) => text_result(msg, false),
                            Err(e) => text_result(e.to_string(), true),
                        }
                    }
                    Err(e) => text_result(format!("sandbox write rejected: {e}"), true),
                }
            }
            "swapna_nucleotide" => {
                let ch = args.get("chain").and_then(Value::as_str).unwrap_or("A");
                let rseq = args.get("res_seq").and_then(Value::as_i64).unwrap_or(1);
                let base = args
                    .get("target_base")
                    .and_then(Value::as_str)
                    .unwrap_or("A");
                match self
                    .session
                    .execute_dsl(&format!("swapna {ch}:{rseq} {base}"))
                {
                    Ok(msg) => text_result(msg, false),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "build_structure_3d" => {
                let kind = args.get("kind").and_then(Value::as_str).unwrap_or("helix");
                let seq = args.get("sequence").and_then(Value::as_str).unwrap_or("");
                let ch = args.get("chain").and_then(Value::as_str).unwrap_or("A");
                match self
                    .session
                    .execute_dsl(&format!("build {kind} {seq} {ch}"))
                {
                    Ok(msg) => text_result(msg, false),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "symmetry_and_glycans" => {
                let act = args.get("action").and_then(Value::as_str).unwrap_or("snfg");
                let cut = args.get("cutoff_a").and_then(Value::as_f64).unwrap_or(5.0);
                let cmd = match act.to_ascii_lowercase().as_str() {
                    "sym" => "sym".to_string(),
                    "symexp" => format!("symexp {cut:.1}"),
                    "crystalcontacts" => format!("crystalcontacts {cut:.1}"),
                    "unitcell" => "unitcell on".to_string(),
                    _ => "snfg on".to_string(),
                };
                match self.session.execute_dsl(&cmd) {
                    Ok(msg) => text_result(
                        format!("{msg}\n\n{}", self.session.last_module_markdown),
                        false,
                    ),
                    Err(e) => text_result(e.to_string(), true),
                }
            }
            "volume_operations_vop" => {
                let op = args.get("op").and_then(Value::as_str).unwrap_or("sharpen");
                let param = args.get("param").and_then(Value::as_f64).unwrap_or(1.5);
                let cmd = if op.eq_ignore_ascii_case("orthoplanes") {
                    "vop orthoplanes on".to_string()
                } else {
                    format!("vop {op} {param:.2}")
                };
                match self.session.execute_dsl(&cmd) {
                    Ok(msg) => text_result(msg, false),
                    Err(e) => text_result(e.to_string(), true),
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
        assert_eq!(tools.len(), 25);

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
