#![forbid(unsafe_code)]

//! `biomol-script` — PyMOL / ChimeraX-Compatible Biomolecular Command Console DSL,
//! Transactional Undo/Redo Session State, Provenance Evidence Log, and Macro Runner.

use std::collections::HashMap;
use std::path::Path;

use biomol_model::{
    audit_structure_prep, minimize_local_mm, mutate_and_repack_best_rotamer, mutate_residue_3d,
};
use biomol_modules::{
    run_adc_module, run_antibody_module, run_cryoem_molmap_module, run_cryoem_volume_module,
    run_denovo_cofold_module, run_enzyme_module, run_ncrna_module, run_physics_anm_module,
    run_physics_gbsa_module,
};
use biomol_physics::run_local_langevin_md;
use biomol_rna::{analyze_ncrna_mrna, find_rna_preset, render_rna_markdown};
use craftbio_report::{export_artcraft_bundle, write_biomol_html_dossier};
use craftbio_scene::{
    ColorScheme, CraftSceneDocument, ProvenanceEntry, ProvenanceKind, RepresentationKind,
};
use craftbio_struct::{
    Biomolecule, VolumeGrid3D, evaluate_selection, load_dcd_frames, matchmaker_align, parse_dx_map,
    parse_mrc_map, parse_structure_auto, write_mmcif, write_multi_model_pdb, write_pdb,
};
use craftbio_viewport::{
    Camera3D, RenderOptions, RenderSceneData, build_render_scene, render_scene_png_bytes,
    write_scene_png,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("command error: {0}")]
    Command(String),
}

/// Complete stateful `BiomolCraft` interactive biomolecular session (shared by GUI, CLI & MCP).
#[derive(Debug, Clone)]
pub struct BiomolSession {
    pub molecule: Option<Biomolecule>,
    pub active_volume: Option<VolumeGrid3D>,
    pub source_path: Option<String>,
    pub camera: Camera3D,
    pub render_opts: RenderOptions,
    pub named_selections: HashMap<String, Vec<bool>>,
    pub undo_stack: Vec<Biomolecule>,
    pub redo_stack: Vec<Biomolecule>,
    pub provenance: Vec<ProvenanceEntry>,
    pub command_history: Vec<String>,
    pub last_module_markdown: String,
    pub dirty_geometry: bool,
}

impl Default for BiomolSession {
    fn default() -> Self {
        Self {
            molecule: None,
            active_volume: None,
            source_path: None,
            camera: Camera3D::default(),
            render_opts: RenderOptions::default(),
            named_selections: HashMap::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            provenance: Vec::new(),
            command_history: Vec::new(),
            last_module_markdown: String::new(),
            dirty_geometry: true,
        }
    }
}

impl BiomolSession {
    pub fn load_structure_text(
        &mut self,
        id: &str,
        raw_text: &str,
        source_label: Option<String>,
    ) -> Result<String, ScriptError> {
        let mol =
            parse_structure_auto(id, raw_text).map_err(|e| ScriptError::Command(e.to_string()))?;
        let audit = audit_structure_prep(&mol);
        let msg = format!(
            "Loaded '{}' ({} atoms, {} chains: prot={:?}, nuc={:?}, lig={}, bp={}).",
            mol.title,
            audit.total_atoms,
            mol.chains.len(),
            audit.protein_chains,
            audit.nucleic_chains,
            audit.ligand_residues,
            audit.rna_base_pair_count
        );
        self.camera.focus_on_biomolecule(&mol);
        self.render_opts.residue_highlights.clear();
        self.render_opts.sphere_overlays.clear();
        self.render_opts.line_overlays.clear();
        self.render_opts.selected_atoms.clear();
        // Auto-select NucleicLadder representation when structure is predominantly RNA/DNA
        if !audit.nucleic_chains.is_empty() && audit.protein_chains.is_empty() {
            self.render_opts.representation = RepresentationKind::NucleicLadder;
            self.render_opts.color_scheme = ColorScheme::Nucleotide;
        } else {
            self.render_opts.representation = RepresentationKind::Cartoon;
            self.render_opts.color_scheme = ColorScheme::Chain;
        }
        self.named_selections.clear();
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.source_path = source_label;
        self.provenance.push(ProvenanceEntry {
            kind: ProvenanceKind::ExperimentalStructure,
            source_app: "biomolcraft".to_string(),
            summary: msg.clone(),
        });
        self.molecule = Some(mol);
        self.dirty_geometry = true;
        Ok(msg)
    }

    pub fn load_structure_file(&mut self, path: &Path) -> Result<String, ScriptError> {
        let raw = std::fs::read_to_string(path)?;
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("structure");
        self.load_structure_text(id, &raw, Some(path.display().to_string()))
    }

    #[must_use]
    pub fn build_viewport_scene(&self) -> RenderSceneData {
        match &self.molecule {
            Some(mol) => build_render_scene(mol, &self.render_opts),
            None => RenderSceneData {
                background: self.render_opts.background,
                ..Default::default()
            },
        }
    }

    #[must_use]
    pub fn to_craftscene_document(&self) -> CraftSceneDocument {
        CraftSceneDocument {
            schema_version: "craftscene/v1".to_string(),
            title: self
                .molecule
                .as_ref()
                .map_or_else(|| "Empty Scene".to_string(), |m| m.title.clone()),
            source_app: "biomolcraft".to_string(),
            pdb_path: self.source_path.clone(),
            inline_pdb: self.molecule.as_ref().map(write_pdb),
            representation: self.render_opts.representation,
            color_scheme: self.render_opts.color_scheme,
            named_selections: self
                .named_selections
                .keys()
                .map(|k| (k.clone(), format!("saved:{k}")))
                .collect(),
            residue_highlights: self
                .render_opts
                .residue_highlights
                .iter()
                .map(|(&(c, r), &rgb)| (c, r, rgb, "highlight".to_string()))
                .collect(),
            sphere_overlays: self.render_opts.sphere_overlays.clone(),
            line_overlays: self.render_opts.line_overlays.clone(),
            active_modules: vec![
                "antibodies".to_string(),
                "ncrna".to_string(),
                "enzymes".to_string(),
            ],
            provenance: self.provenance.clone(),
        }
    }

    fn push_undo_snapshot(&mut self) {
        if let Some(ref mol) = self.molecule {
            self.undo_stack.push(mol.clone());
            if self.undo_stack.len() > 25 {
                self.undo_stack.remove(0);
            }
            self.redo_stack.clear();
        }
    }

    /// Execute one or more semicolon-separated PyMOL/ChimeraX-compatible DSL commands.
    pub fn execute_dsl(&mut self, script: &str) -> Result<String, ScriptError> {
        let mut responses = Vec::new();
        for raw_stmt in script.split(';') {
            let stmt = raw_stmt.trim();
            if stmt.is_empty() || stmt.starts_with('#') {
                continue;
            }
            self.command_history.push(stmt.to_string());
            responses.push(self.execute_single_cmd(stmt)?);
        }
        Ok(responses.join("\n"))
    }

    fn execute_single_cmd(&mut self, stmt: &str) -> Result<String, ScriptError> {
        let mut parts = stmt.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
        let rest = parts.next().unwrap_or("").trim();

        match cmd.as_str() {
            "help" => Ok(
                "Commands: open <path> | select <name>, <expr> | style <cartoon|nucleic|sticks|spheres|surface|lines> | color <chain|ss|cpk|nucleotide|plddt|sap|esp> | surface <on|off> | silhouettes <on|off> | align <mob_ch>, <ref_ch> | mutate <C:RES> <AA> [rot|auto] | minimize [steps] | module <antibodies|ncrna|enzymes|adc> | rnafold <seq|preset> | png <path> [W] [H] [transparent] | save <path> | undo | redo | reset"
                    .to_string(),
            ),
            "open" | "load" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command("usage: open <path.pdb|cif>".to_string()));
                }
                self.load_structure_file(Path::new(rest.trim_matches('"')))
            }
            "style" | "show" => {
                let r = match rest.to_ascii_lowercase().as_str() {
                    "cartoon" | "ribbon" => RepresentationKind::Cartoon,
                    "nucleic" | "rna" | "ladder" => RepresentationKind::NucleicLadder,
                    "sticks" | "stick" => RepresentationKind::Sticks,
                    "spheres" | "sphere" | "vdw" => RepresentationKind::Spheres,
                    "surface" | "surf" => RepresentationKind::Surface,
                    "lines" | "line" => RepresentationKind::Lines,
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown style '{other}' (use cartoon|nucleic|sticks|spheres|surface|lines)"
                        )));
                    }
                };
                self.render_opts.representation = r;
                self.dirty_geometry = true;
                Ok(format!("Representation set to {}.", r.label()))
            }
            "color" => {
                let c = match rest.to_ascii_lowercase().as_str() {
                    "chain" => ColorScheme::Chain,
                    "ss" | "dssp" | "secondary" => ColorScheme::SecondaryStructure,
                    "cpk" | "elem" | "element" => ColorScheme::ElementCpk,
                    "nucleotide" | "rna" | "base" => ColorScheme::Nucleotide,
                    "plddt" | "bfactor" | "b" => ColorScheme::BfactorPlddt,
                    "sap" => ColorScheme::Sap3D,
                    "esp" | "electrostatic" => ColorScheme::ElectrostaticEsp,
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown color scheme '{other}' (use chain|ss|cpk|nucleotide|plddt|sap|esp)"
                        )));
                    }
                };
                self.render_opts.color_scheme = c;
                self.dirty_geometry = true;
                Ok(format!("Color scheme set to {}.", c.label()))
            }
            "surface" => {
                let on = matches!(rest.to_ascii_lowercase().as_str(), "on" | "true" | "1" | "");
                self.render_opts.show_surface_shell = on;
                self.dirty_geometry = true;
                Ok(format!("Translucent surface shell: {on}."))
            }
            "silhouettes" => {
                let on = matches!(rest.to_ascii_lowercase().as_str(), "on" | "true" | "1" | "");
                self.render_opts.show_silhouettes = on;
                Ok(format!("Silhouettes: {on}."))
            }
            "waters" => {
                let on = matches!(rest.to_ascii_lowercase().as_str(), "on" | "true" | "1" | "");
                self.render_opts.show_waters = on;
                self.dirty_geometry = true;
                Ok(format!("Show waters: {on}."))
            }
            "select" | "sele" => {
                let mol = self
                    .molecule
                    .as_ref()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let (name, expr) = if let Some((a, b)) = rest.split_once(',') {
                    (a.trim().to_string(), b.trim())
                } else {
                    ("sele".to_string(), rest)
                };
                let mask = evaluate_selection(mol, expr, &self.named_selections)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let count = mask.iter().filter(|&&b| b).count();
                self.render_opts.selected_atoms = mask.clone();
                self.named_selections.insert(name.clone(), mask);
                self.dirty_geometry = true;
                Ok(format!("Selection '{name}' matched {count} atoms."))
            }
            "mutate" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: mutate <CHAIN:RESSEQ> <AA1> [rotamer_idx|auto]".to_string(),
                    ));
                }
                let (c_s, r_s) = toks[0].split_once(':').ok_or_else(|| {
                    ScriptError::Command("expected CHAIN:RESSEQ (e.g. B:105)".to_string())
                })?;
                let ch = c_s
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing chain id".to_string()))?;
                let rseq: i32 = r_s
                    .parse()
                    .map_err(|_| ScriptError::Command("invalid residue sequence number".to_string()))?;
                let aa = toks[1]
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing mutant amino acid".to_string()))?;

                self.push_undo_snapshot();
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = if let Some(&rot_s) = toks.get(2)
                    && let Ok(rot_idx) = rot_s.parse::<usize>()
                {
                    mutate_residue_3d(mol, ch, rseq, aa, rot_idx)
                        .map_err(|e| ScriptError::Command(e.to_string()))?
                } else {
                    mutate_and_repack_best_rotamer(mol, ch, rseq, aa)
                        .map_err(|e| ScriptError::Command(e.to_string()))?
                };
                self.render_opts
                    .residue_highlights
                    .insert((ch, rseq), [0.18, 0.92, 0.55]);
                self.render_opts.sphere_overlays.extend(rep.clash_overlays.clone());
                self.dirty_geometry = true;
                let msg = format!(
                    "Mutated {ch}:{}{rseq}{} (rotamer {}/{}, chi1={:.0}°): {} clashes, {} new H-bonds.",
                    rep.wild_type,
                    rep.mutant,
                    rep.rotamer_index + 1,
                    rep.total_rotamers,
                    rep.chi1_deg,
                    rep.clash_count,
                    rep.new_hbond_count
                );
                self.provenance.push(ProvenanceEntry {
                    kind: ProvenanceKind::InteractiveEdit,
                    source_app: "biomolcraft::biomol-model".to_string(),
                    summary: msg.clone(),
                });
                Ok(msg)
            }
            "minimize" => {
                let steps: usize = rest.parse().unwrap_or(80);
                self.push_undo_snapshot();
                let mask = if !self.render_opts.selected_atoms.is_empty() {
                    self.render_opts.selected_atoms.clone()
                } else if let Some(ref m) = self.molecule {
                    vec![true; m.atoms.len()]
                } else {
                    return Err(ScriptError::Command("no biomolecule loaded".to_string()));
                };
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = minimize_local_mm(mol, &mask, steps);
                self.dirty_geometry = true;
                let msg = format!(
                    "Minimized {} atoms ({} steps): E {:.2} -> {:.2} kcal/mol (max disp {:.3} Å).",
                    rep.atoms_relaxed,
                    rep.steps,
                    rep.energy_before_kcal,
                    rep.energy_after_kcal,
                    rep.max_displacement_a
                );
                self.provenance.push(ProvenanceEntry {
                    kind: ProvenanceKind::InteractiveEdit,
                    source_app: "biomolcraft::mm-local".to_string(),
                    summary: msg.clone(),
                });
                Ok(msg)
            }
            "align" => {
                let toks: Vec<&str> = rest
                    .split([',', ' '])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: align <mobile_chain>, <target_chain>".to_string(),
                    ));
                }
                let mc = toks[0].chars().next().ok_or_else(|| ScriptError::Command("bad mobile chain".to_string()))?;
                let tc = toks[1].chars().next().ok_or_else(|| ScriptError::Command("bad target chain".to_string()))?;
                self.push_undo_snapshot();
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let target_clone = mol.clone();
                let tf = matchmaker_align(mol, mc, &target_clone, tc)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                Ok(format!(
                    "Aligned chain {mc} onto {tc}: RMSD = {:.3} Å ({} pairs).",
                    tf.rmsd_a, tf.aligned_pairs
                ))
            }
            "module" => {
                let mol = self
                    .molecule
                    .as_ref()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let key = rest.to_ascii_lowercase();
                let visual = match key.as_str() {
                    "antibodies" | "ab" | "vhh" => run_antibody_module(mol, None, None)
                        .map_err(ScriptError::Command)?
                        .visual,
                    "ncrna" | "rna" | "mrna" => run_ncrna_module(mol)
                        .map_err(ScriptError::Command)?
                        .0,
                    "enzymes" | "enzyme" => run_enzyme_module(mol),
                    "adc" => run_adc_module(mol, "ggfg", "dxd", 8.0, false)
                        .map_err(ScriptError::Command)?
                        .0,
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown module '{other}' (use antibodies|ncrna|enzymes|adc)"
                        )));
                    }
                };
                if let Some(r) = visual.recommended_repr {
                    self.render_opts.representation = r;
                }
                if let Some(c) = visual.recommended_color {
                    self.render_opts.color_scheme = c;
                }
                self.render_opts.residue_highlights = visual.residue_highlights;
                self.render_opts.sphere_overlays = visual.sphere_overlays;
                self.render_opts.line_overlays = visual.line_overlays;
                self.last_module_markdown = visual.summary_markdown;
                self.provenance.extend(visual.provenance);
                self.dirty_geometry = true;
                Ok(format!("Activated module '{}'.", visual.module_name))
            }
            "rnafold" => {
                let (name, seq) = if let Some(p) = find_rna_preset(rest) {
                    (p.name.to_string(), p.sequence.to_string())
                } else {
                    ("Custom_RNA".to_string(), rest.to_string())
                };
                let rep = analyze_ncrna_mrna(&name, "ncRNA/mRNA", &seq, None)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.last_module_markdown = render_rna_markdown(&rep);
                Ok(format!(
                    "Folded '{}' ({} nt): MFE = {:.2} kcal/mol, {} pairs, {} hairpins.",
                    rep.name,
                    rep.fold.folded_len,
                    rep.fold.mfe_kcal_mol,
                    rep.fold.pairs.len(),
                    rep.fold.hairpin_count
                ))
            }
            "png" | "ray" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: png <out.png> [width] [height] [transparent]".to_string(),
                    ));
                }
                let out = Path::new(toks[0]);
                let w: u32 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(1600);
                let h: u32 = toks.get(2).and_then(|s| s.parse().ok()).unwrap_or(1200);
                let transp = toks.iter().any(|&t| t.eq_ignore_ascii_case("transparent"));
                let scene = self.build_viewport_scene();
                let bytes = write_scene_png(&scene, &self.camera, w, h, transp, out)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                Ok(format!("Rendered {}x{} PNG to {} ({bytes} bytes).", w, h, out.display()))
            }
            "save" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command("usage: save <out.pdb|out.craftscene.json>".to_string()));
                }
                let out = Path::new(rest);
                if rest.ends_with(".json") {
                    let doc = self.to_craftscene_document();
                    let text = serde_json::to_string_pretty(&doc)
                        .map_err(|e| ScriptError::Command(e.to_string()))?;
                    std::fs::write(out, text)?;
                    Ok(format!("Saved CraftScene JSON to {}.", out.display()))
                } else if rest.ends_with(".cif") || rest.ends_with(".mmcif") {
                    let mol = self
                        .molecule
                        .as_ref()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    std::fs::write(out, write_mmcif(mol))?;
                    Ok(format!("Saved mmCIF to {}.", out.display()))
                } else {
                    let mol = self
                        .molecule
                        .as_ref()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    let txt = if mol.frames.len() > 1 {
                        write_multi_model_pdb(mol)
                    } else {
                        write_pdb(mol)
                    };
                    std::fs::write(out, txt)?;
                    Ok(format!("Saved PDB to {}.", out.display()))
                }
            }
            "anm" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let mode: usize = toks.first().and_then(|s| s.parse().ok()).unwrap_or(1);
                let amp: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(2.2);
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let (vis, rep) = run_physics_anm_module(mol, mode, amp, 20)
                    .map_err(ScriptError::Command)?;
                self.render_opts.residue_highlights = vis.residue_highlights;
                self.render_opts.line_overlays = vis.line_overlays;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "ANM mode {} animated across {} frames ({} nodes, {} hinges).",
                    mode,
                    mol.frames.len(),
                    rep.guide_nodes,
                    rep.hinge_residues.len()
                ))
            }
            "gbsa" | "dipole" => {
                let mol = self
                    .molecule
                    .as_ref()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let (vis, rep) = run_physics_gbsa_module(mol);
                self.render_opts.color_scheme = ColorScheme::ElectrostaticEsp;
                self.render_opts.line_overlays = vis.line_overlays;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "GB/SA total {:.2} kcal/mol | Dipole {:.1} D | dG_bind={:?}.",
                    rep.complex_energy.total_kcal,
                    rep.dipole.dipole_magnitude_debye,
                    rep.delta_g_bind_kcal
                ))
            }
            "md" => {
                let steps: usize = rest.parse().unwrap_or(120);
                self.push_undo_snapshot();
                let mask = self.render_opts.selected_atoms.clone();
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = run_local_langevin_md(mol, &mask, steps, 6, 300.0, 1.5, 2026)
                    .map_err(ScriptError::Command)?;
                self.dirty_geometry = true;
                let msg = format!(
                    "Pure-Rust Langevin MD ({} mobile atoms, {} steps -> {} frames): final RMSD {:.3} Å, mean E {:.1} kcal/mol.",
                    rep.mobile_atoms,
                    rep.total_steps,
                    rep.frames_recorded,
                    rep.final_rmsd_a,
                    rep.mean_potential_kcal
                );
                self.provenance.push(ProvenanceEntry {
                    kind: ProvenanceKind::MolecularDynamics,
                    source_app: "biomolcraft::biomol-physics (Langevin-Rust)".to_string(),
                    summary: msg.clone(),
                });
                Ok(msg)
            }
            "frame" => {
                let idx: usize = rest
                    .parse()
                    .map_err(|_| ScriptError::Command("usage: frame <0-based index>".to_string()))?;
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                if mol.set_active_frame(idx) {
                    self.dirty_geometry = true;
                    Ok(format!("Switched to trajectory frame {}/{}.", idx + 1, mol.frames.len()))
                } else {
                    Err(ScriptError::Command(format!(
                        "frame {idx} out of range (loaded frames: {})",
                        mol.frames.len()
                    )))
                }
            }
            "load_dcd" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command("usage: load_dcd <trajectory.dcd>".to_string()));
                }
                let bytes = std::fs::read(rest)?;
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let loaded = load_dcd_frames(mol, &bytes)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                let msg = format!("Loaded {loaded} binary DCD trajectory frames from {rest}.");
                self.provenance.push(ProvenanceEntry {
                    kind: ProvenanceKind::MolecularDynamics,
                    source_app: "biomolcraft::dcd-loader".to_string(),
                    summary: msg.clone(),
                });
                Ok(msg)
            }
            "tunnels" => self.execute_single_cmd("module enzymes"),
            "molmap" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let res: f64 = toks.first().and_then(|s| s.parse().ok()).unwrap_or(4.5);
                let sigma: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.5);
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let (vis, grid, fit) =
                    run_cryoem_molmap_module(mol, res, sigma).map_err(ScriptError::Command)?;
                self.active_volume = Some(grid);
                self.render_opts.sphere_overlays = vis.sphere_overlays;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "Synthesized {:.1} Å Cryo-EM molmap ({:.1}σ): CC={:.3}, {:.1}% atoms >= 1σ.",
                    res, sigma, fit.correlation_after, fit.atoms_inside_1sigma_pct
                ))
            }
            "open_map" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let path_s = toks
                    .first()
                    .copied()
                    .ok_or_else(|| ScriptError::Command("usage: open_map <map.mrc|map.dx> [sigma]".to_string()))?;
                let sigma: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.5);
                let grid = if path_s.to_ascii_lowercase().ends_with(".dx") {
                    let txt = std::fs::read_to_string(path_s)?;
                    parse_dx_map(path_s, &txt).map_err(|e| ScriptError::Command(e.to_string()))?
                } else {
                    let bytes = std::fs::read(path_s)?;
                    parse_mrc_map(path_s, &bytes).map_err(|e| ScriptError::Command(e.to_string()))?
                };
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let (vis, fit) = run_cryoem_volume_module(mol, &grid, sigma, false);
                self.active_volume = Some(grid);
                self.render_opts.sphere_overlays = vis.sphere_overlays;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "Loaded volumetric map '{path_s}' ({sigma:.1}σ): CC={:.3}.",
                    fit.correlation_after
                ))
            }
            "fitmap" => {
                let grid = self
                    .active_volume
                    .clone()
                    .ok_or_else(|| ScriptError::Command("no active volume map (run `molmap` or `open_map` first)".to_string()))?;
                let mol = self
                    .molecule
                    .as_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let (vis, fit) = run_cryoem_volume_module(mol, &grid, 1.5, true);
                self.render_opts.sphere_overlays = vis.sphere_overlays;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "fitmap complete: CC {:.3} -> {:.3}, shift {:?} Å.",
                    fit.correlation_before, fit.correlation_after, fit.translation_applied_a
                ))
            }
            "denovo" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let ch = toks
                    .first()
                    .and_then(|s| s.chars().next())
                    .unwrap_or('C');
                let hotspots: Vec<String> = toks.iter().skip(1).map(|s| (*s).to_string()).collect();
                let mol = self
                    .molecule
                    .as_ref()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let (vis, plan) = run_denovo_cofold_module(mol, ch, &hotspots, None)
                    .map_err(ScriptError::Command)?;
                self.render_opts.residue_highlights = vis.residue_highlights;
                self.render_opts.sphere_overlays = vis.sphere_overlays;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "Staged De Novo & Co-Fold campaign '{}' (contig {}).",
                    plan.campaign_name, plan.rfdiffusion_contig
                ))
            }
            "export_html" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command("usage: export_html <out.html>".to_string()));
                }
                let mol = self
                    .molecule
                    .as_ref()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let sum = write_biomol_html_dossier(
                    &mol.title,
                    &write_pdb(mol),
                    &self.last_module_markdown,
                    &[],
                    true,
                    Path::new(rest),
                )
                .map_err(ScriptError::Command)?;
                Ok(format!(
                    "Exported interactive 3D HTML dossier to {} ({} bytes, offline_molstar={}).",
                    sum.out_path, sum.bytes, sum.molstar_inline_offline
                ))
            }
            "export_artcraft" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: export_artcraft <out_dir>".to_string(),
                    ));
                }
                let title = self
                    .molecule
                    .as_ref()
                    .map_or_else(|| "BiomolCraft Scene".to_string(), |m| m.title.clone());
                let atom_cnt = self.molecule.as_ref().map_or(0, |m| m.atoms.len());
                let scene = self.build_viewport_scene();
                let png = render_scene_png_bytes(&scene, &self.camera, 1600, 1200, false)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let rows = vec![
                    ("Structure", title.clone()),
                    ("Atoms", atom_cnt.to_string()),
                    ("Representation", self.render_opts.representation.label().to_string()),
                    ("Color Scheme", self.render_opts.color_scheme.label().to_string()),
                ];
                let sum = export_artcraft_bundle(
                    &title,
                    &png,
                    &rows,
                    &self.last_module_markdown,
                    Path::new(rest),
                )
                .map_err(ScriptError::Command)?;
                Ok(format!(
                    "Exported ArtCraft bundle (PNG + GridCraft TSV + WordCraft/DeckCraft MD) to {}.",
                    sum.out_dir
                ))
            }
            "undo" => {
                if let Some(prev) = self.undo_stack.pop() {
                    if let Some(cur) = self.molecule.replace(prev) {
                        self.redo_stack.push(cur);
                    }
                    self.dirty_geometry = true;
                    Ok("Undid last 3D structural edit.".to_string())
                } else {
                    Ok("Nothing to undo.".to_string())
                }
            }
            "redo" => {
                if let Some(next) = self.redo_stack.pop() {
                    if let Some(cur) = self.molecule.replace(next) {
                        self.undo_stack.push(cur);
                    }
                    self.dirty_geometry = true;
                    Ok("Redid 3D structural edit.".to_string())
                } else {
                    Ok("Nothing to redo.".to_string())
                }
            }
            "reset" | "center" => {
                if let Some(ref mol) = self.molecule {
                    self.camera.focus_on_biomolecule(mol);
                }
                self.render_opts.selected_atoms.clear();
                self.dirty_geometry = true;
                Ok("Centered camera and cleared selection.".to_string())
            }
            other => Err(ScriptError::Command(format!(
                "unknown DSL command '{other}' (type 'help' for command list)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsl_executes_selection_mutation_undo_and_rna_fold() -> Result<(), Box<dyn std::error::Error>>
    {
        let pdb = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 90.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 90.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.210  1.00 90.00           C
END
";
        let mut sess = BiomolSession::default();
        sess.load_structure_text("mini", pdb, None)?;
        let out = sess.execute_dsl(
            "select active, chain A and resi 1; mutate A:1 W; undo; rnafold guanine_riboswitch",
        )?;
        assert!(out.contains("matched 5 atoms"));
        assert!(out.contains("Mutated A:A1W"));
        assert!(out.contains("Undid last 3D structural edit"));
        assert!(out.contains("Folded"));
        Ok(())
    }
}
