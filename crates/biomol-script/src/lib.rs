#![forbid(unsafe_code)]

//! `biomol-script` — Multi-Model (`#1, #2, #3...`) PyMOL/ChimeraX-Compatible Biomolecular Command
//! Console DSL, Scenes (`F1–F12`), Cached Online `fetch`, Native **`ViewDockX`** Docking Session,
//! Transactional Undo/Redo, and Provenance Evidence Log (shared by GUI, CLI & MCP).

pub mod fetch;

use std::collections::{BTreeMap, HashMap};
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
    ColorScheme, CraftSceneDocument, DockCraftSceneBundle, GridBox3D, LightingPreset,
    ProvenanceEntry, ProvenanceKind, RepresentationKind, SelectionBoxSnapshot,
};
use craftbio_struct::{
    Biomolecule, ResidueClass, VolumeGrid3D, evaluate_selection, load_dcd_frames, matchmaker_align,
    parse_dx_map, parse_mrc_map, parse_sdf_poses, parse_structure_auto, write_mmcif,
    write_multi_model_pdb, write_pdb,
};
use craftbio_viewport::{
    Camera3D, CapsuleInstance, RenderOptions, RenderSceneData, SphereInstance, build_render_scene,
    extract_isosurface_mesh, push_grid_box_wireframe, render_scene_png_bytes, write_scene_png,
};
pub use fetch::{FetchedAsset, fetch_structure_spec};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("command error: {0}")]
    Command(String),
}

/// One loaded 3D biomolecular model (`#1`, `#2`, `#3`...) with independent style, color & visibility.
#[derive(Debug, Clone)]
pub struct SceneModel {
    pub model_id: u32,
    pub name: String,
    pub role: String,
    pub molecule: Biomolecule,
    pub visible: bool,
    pub render_opts: RenderOptions,
    pub source_path: Option<String>,
    pub undo_stack: Vec<Biomolecule>,
    pub redo_stack: Vec<Biomolecule>,
}

/// Restorable ChimeraX / PyMOL Scene snapshot (`F1–F12` or named scene).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedScene {
    pub name: String,
    pub camera: Camera3D,
    pub lighting: LightingPreset,
    pub background: [f32; 4],
    pub ssao_strength: f32,
    pub show_silhouettes: bool,
    pub model_states: Vec<(u32, bool, RepresentationKind, ColorScheme, bool)>,
    pub active_dock_pose: Option<usize>,
}

/// Triage status tag for poses inside `ViewDockX` (ChimeraX ViewDock + medicinal chemistry workflow).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoseStatusTag {
    Unreviewed,
    Favorite,
    Review,
    Discarded,
}

impl PoseStatusTag {
    #[must_use]
    pub fn short_label(self) -> &'static str {
        match self {
            Self::Unreviewed => "—",
            Self::Favorite => "★ Fav",
            Self::Review => "? Rev",
            Self::Discarded => "✕ Disc",
        }
    }

    #[must_use]
    pub fn cycle(self) -> Self {
        match self {
            Self::Unreviewed => Self::Favorite,
            Self::Favorite => Self::Review,
            Self::Review => Self::Discarded,
            Self::Discarded => Self::Unreviewed,
        }
    }
}

/// Active `ViewDockX` multi-pose docking & SBDD inspection state synced with `DockCraft`.
#[derive(Debug, Clone)]
pub struct ViewDockSession {
    pub bundle: DockCraftSceneBundle,
    pub active_pose_idx: usize,
    pub pose_models: Vec<Option<Biomolecule>>,
    pub pose_tags: Vec<PoseStatusTag>,
    pub show_hyde_coronas: bool,
    pub show_torsion_lights: bool,
    pub show_interactions: bool,
    pub show_pharmacophores: bool,
    pub show_dmif_hotspots: bool,
    pub show_bridging_waters: bool,
    pub show_grid_box: bool,
    pub editable_grid_box: Option<GridBox3D>,
}

/// Complete stateful multi-model `BiomolCraft` interactive session (shared by GUI, CLI & MCP).
#[derive(Debug, Clone)]
pub struct BiomolSession {
    pub models: Vec<SceneModel>,
    pub active_model_idx: usize,
    next_model_id: u32,
    pub scenes: BTreeMap<String, SavedScene>,
    pub viewdock: Option<ViewDockSession>,
    pub active_volume: Option<VolumeGrid3D>,
    pub active_volume_sigma: f64,
    pub volume_wireframe: bool,
    pub camera: Camera3D,
    /// Global viewport rendering defaults & active model mirror (`representation`, `color_scheme`, etc.).
    pub render_opts: RenderOptions,
    pub named_selections: HashMap<String, Vec<bool>>,
    pub provenance: Vec<ProvenanceEntry>,
    pub command_history: Vec<String>,
    pub last_module_markdown: String,
    pub dirty_geometry: bool,
}

impl Default for BiomolSession {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            active_model_idx: 0,
            next_model_id: 1,
            scenes: BTreeMap::new(),
            viewdock: None,
            active_volume: None,
            active_volume_sigma: 1.5,
            volume_wireframe: false,
            camera: Camera3D::default(),
            render_opts: RenderOptions::default(),
            named_selections: HashMap::new(),
            provenance: Vec::new(),
            command_history: Vec::new(),
            last_module_markdown: String::new(),
            dirty_geometry: true,
        }
    }
}

impl BiomolSession {
    #[must_use]
    pub fn active_model(&self) -> Option<&SceneModel> {
        self.models.get(self.active_model_idx)
    }

    pub fn active_model_mut(&mut self) -> Option<&mut SceneModel> {
        self.models.get_mut(self.active_model_idx)
    }

    #[must_use]
    pub fn active_molecule(&self) -> Option<&Biomolecule> {
        self.active_model().map(|m| &m.molecule)
    }

    pub fn active_molecule_mut(&mut self) -> Option<&mut Biomolecule> {
        self.active_model_mut().map(|m| &mut m.molecule)
    }

    /// Sync global `self.render_opts` onto the currently active `SceneModel` so GUI/CLI mutations stay coherent.
    pub fn sync_active_render_opts(&mut self) {
        let g = self.render_opts.clone();
        if let Some(m) = self.models.get_mut(self.active_model_idx) {
            m.render_opts.representation = g.representation;
            m.render_opts.color_scheme = g.color_scheme;
            m.render_opts.show_surface_shell = g.show_surface_shell;
            m.render_opts.surface_alpha = g.surface_alpha;
            m.render_opts.surface_wireframe = g.surface_wireframe;
            m.render_opts.show_ligands_sticks = g.show_ligands_sticks;
            m.render_opts.show_waters = g.show_waters;
            m.render_opts.show_rna_base_pairs = g.show_rna_base_pairs;
            m.render_opts.show_silhouettes = g.show_silhouettes;
            m.render_opts.lighting = g.lighting;
            m.render_opts.ssao_strength = g.ssao_strength;
            m.render_opts.background = g.background;
            m.render_opts.residue_highlights = g.residue_highlights;
            m.render_opts.sphere_overlays = g.sphere_overlays;
            m.render_opts.line_overlays = g.line_overlays;
            m.render_opts.selected_atoms = g.selected_atoms;
            m.render_opts.grid_box = g.grid_box;
        }
    }

    fn pull_active_render_opts(&mut self) {
        if let Some(m) = self.models.get(self.active_model_idx) {
            let bg = self.render_opts.background;
            let light = self.render_opts.lighting;
            let ssao = self.render_opts.ssao_strength;
            let sil = self.render_opts.show_silhouettes;
            self.render_opts = m.render_opts.clone();
            self.render_opts.background = bg;
            self.render_opts.lighting = light;
            self.render_opts.ssao_strength = ssao;
            self.render_opts.show_silhouettes = sil;
        }
    }

    fn insert_model(
        &mut self,
        mol: Biomolecule,
        source_label: Option<String>,
        role: &str,
        replace_workspace: bool,
    ) -> String {
        let audit = audit_structure_prep(&mol);
        let mut m_opts = RenderOptions {
            background: self.render_opts.background,
            lighting: self.render_opts.lighting,
            ssao_strength: self.render_opts.ssao_strength,
            show_silhouettes: self.render_opts.show_silhouettes,
            ..RenderOptions::default()
        };
        if !audit.nucleic_chains.is_empty() && audit.protein_chains.is_empty() {
            m_opts.representation = RepresentationKind::NucleicLadder;
            m_opts.color_scheme = ColorScheme::Nucleotide;
        } else if audit.protein_chains.is_empty() && audit.nucleic_chains.is_empty() {
            m_opts.representation = RepresentationKind::Sticks;
            m_opts.color_scheme = ColorScheme::ElementCpk;
        } else {
            m_opts.representation = RepresentationKind::Cartoon;
            m_opts.color_scheme = ColorScheme::Chain;
        }

        if replace_workspace {
            self.models.clear();
            self.next_model_id = 1;
            self.named_selections.clear();
            self.viewdock = None;
        }
        let id_num = self.next_model_id;
        self.next_model_id += 1;
        let name = if mol.id.is_empty() {
            format!("model_{id_num}")
        } else {
            mol.id.clone()
        };
        let msg = format!(
            "Loaded #{id_num} '{}' ({} atoms, {} chains: prot={:?}, nuc={:?}, lig={}, bp={}).",
            mol.title,
            audit.total_atoms,
            mol.chains.len(),
            audit.protein_chains,
            audit.nucleic_chains,
            audit.ligand_residues,
            audit.rna_base_pair_count
        );
        self.models.push(SceneModel {
            model_id: id_num,
            name,
            role: role.to_string(),
            molecule: mol,
            visible: true,
            render_opts: m_opts,
            source_path: source_label,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        });
        self.active_model_idx = self.models.len().saturating_sub(1);
        self.pull_active_render_opts();
        self.focus_camera_all_visible();
        self.provenance.push(ProvenanceEntry {
            kind: ProvenanceKind::ExperimentalStructure,
            source_app: "biomolcraft".to_string(),
            summary: msg.clone(),
        });
        self.dirty_geometry = true;
        msg
    }

    /// Replace workspace with a single parsed biomolecule (default for single-file CLI workflows).
    pub fn load_structure_text(
        &mut self,
        id: &str,
        raw_text: &str,
        source_label: Option<String>,
    ) -> Result<String, ScriptError> {
        let mol =
            parse_structure_auto(id, raw_text).map_err(|e| ScriptError::Command(e.to_string()))?;
        Ok(self.insert_model(mol, source_label, "structure", true))
    }

    /// Append a new model (`#2, #3...`) without closing already loaded models!
    pub fn add_structure_text(
        &mut self,
        id: &str,
        raw_text: &str,
        source_label: Option<String>,
    ) -> Result<String, ScriptError> {
        let mol =
            parse_structure_auto(id, raw_text).map_err(|e| ScriptError::Command(e.to_string()))?;
        Ok(self.insert_model(mol, source_label, "structure", false))
    }

    pub fn load_structure_file(&mut self, path: &Path) -> Result<String, ScriptError> {
        let raw = std::fs::read_to_string(path)?;
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("structure");
        self.load_structure_text(id, &raw, Some(path.display().to_string()))
    }

    pub fn add_structure_file(&mut self, path: &Path) -> Result<String, ScriptError> {
        let raw = std::fs::read_to_string(path)?;
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("structure");
        self.add_structure_text(id, &raw, Some(path.display().to_string()))
    }

    /// Center camera cleanly across all currently visible models (`#1..#N`).
    pub fn focus_camera_all_visible(&mut self) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut any = false;
        for m in &self.models {
            if !m.visible || m.molecule.atoms.is_empty() {
                continue;
            }
            let (mlo, mhi, _, _) = m.molecule.bounds();
            for k in 0..3 {
                lo[k] = lo[k].min(mlo[k]);
                hi[k] = hi[k].max(mhi[k]);
            }
            any = true;
        }
        if any {
            let center = [
                0.5 * (lo[0] + hi[0]),
                0.5 * (lo[1] + hi[1]),
                0.5 * (lo[2] + hi[2]),
            ];
            let dx = hi[0] - lo[0];
            let dy = hi[1] - lo[1];
            let dz = hi[2] - lo[2];
            let rad = ((dx * dx + dy * dy + dz * dz).sqrt() * 0.5).max(6.0);
            self.camera.focus_on_bounds(center, rad);
        }
    }

    /// Load a complete rich `DockCraftSceneBundle` into `#1` receptor + native **`ViewDockX`** session.
    pub fn load_dockcraft_bundle(
        &mut self,
        bundle: DockCraftSceneBundle,
    ) -> Result<String, ScriptError> {
        let rec_raw = if let Some(ref txt) = bundle.receptor_pdb_inline
            && !txt.trim().is_empty()
        {
            txt.clone()
        } else if let Some(ref p) = bundle.receptor_pdb_path {
            std::fs::read_to_string(p)?
        } else {
            return Err(ScriptError::Command(
                "DockCraftSceneBundle missing both receptor_pdb_inline and receptor_pdb_path"
                    .to_string(),
            ));
        };
        let mut rec = parse_structure_auto(&bundle.receptor_name, &rec_raw)
            .map_err(|e| ScriptError::Command(e.to_string()))?;

        // Apply optional per-residue MD RMSF or MM/GBSA metrics into B-factor channel
        if !bundle.residue_metric_overrides.is_empty() {
            let mut metric_map: HashMap<(char, i32), f64> = HashMap::new();
            for &(ch, rseq, val, _) in &bundle.residue_metric_overrides {
                metric_map.insert((ch, rseq), val);
            }
            for a in &mut rec.atoms {
                if let Some(&v) = metric_map.get(&(a.chain, a.res_seq)) {
                    a.b_factor = v;
                }
            }
        }

        // Optional binary DCD trajectory bundled from DockCraft MD validation
        if let Some(ref dcd_p) = bundle.trajectory_dcd_path
            && let Ok(bytes) = std::fs::read(dcd_p)
        {
            let _ = load_dcd_frames(&mut rec, &bytes);
        }

        let _ = self.insert_model(rec, bundle.receptor_pdb_path.clone(), "receptor", true);

        // Parse every docked pose SDF block
        let mut pose_models = Vec::with_capacity(bundle.poses.len());
        for p in &bundle.poses {
            let parsed = if !p.sdf_block.trim().is_empty() {
                parse_sdf_poses(&p.sdf_block)
                    .ok()
                    .and_then(|mut v| v.drain(..).next().map(|r| r.molecule))
            } else {
                None
            };
            pose_models.push(parsed);
        }
        let n_poses = bundle.poses.len();
        let editable_grid_box = bundle.grid_box;
        let title = bundle.title.clone();
        self.provenance.extend(bundle.provenance.clone());
        self.provenance.push(ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: bundle.source_app.clone(),
            summary: format!(
                "Loaded ViewDockX bundle '{}' ({} poses, {} pharm, {} ΔMIF hotspots).",
                bundle.title,
                n_poses,
                bundle.pharmacophore_spheres.len(),
                bundle.selectivity_dmif_hotspots.len()
            ),
        });

        self.viewdock = Some(ViewDockSession {
            bundle,
            active_pose_idx: 0,
            pose_models,
            pose_tags: vec![PoseStatusTag::Unreviewed; n_poses],
            show_hyde_coronas: true,
            show_torsion_lights: true,
            show_interactions: true,
            show_pharmacophores: true,
            show_dmif_hotspots: true,
            show_bridging_waters: true,
            show_grid_box: true,
            editable_grid_box,
        });

        // Focus smoothly on the active binding pocket / GridBox!
        if let Some(gb) = editable_grid_box {
            let r = (gb.size[0].max(gb.size[1]).max(gb.size[2]) * 0.65).max(9.0);
            self.camera.focus_on_bounds(gb.center, r);
        }
        self.dirty_geometry = true;
        Ok(format!(
            "ViewDockX loaded '{title}' ({n_poses} poses ready — use ↑/↓ or `pose <idx>`)."
        ))
    }

    /// Switch the active docked pose inside `ViewDockX` (0-based index).
    pub fn select_dock_pose(&mut self, idx: usize) -> Result<String, ScriptError> {
        let Some(ref mut vd) = self.viewdock else {
            return Err(ScriptError::Command(
                "no ViewDockX bundle loaded (use `load_dock <bundle.json>` or push from DockCraft)"
                    .to_string(),
            ));
        };
        if idx >= vd.bundle.poses.len() {
            return Err(ScriptError::Command(format!(
                "pose index {idx} out of range (loaded: {})",
                vd.bundle.poses.len()
            )));
        }
        vd.active_pose_idx = idx;
        let p = &vd.bundle.poses[idx];
        let msg = format!(
            "ViewDockX pose #{}/{}: {} | Vina {:.2} | HYDE {} kcal/mol.",
            idx + 1,
            vd.bundle.poses.len(),
            p.name,
            p.vina_kcal,
            p.hyde_kcal
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.2}"))
        );
        self.dirty_geometry = true;
        Ok(msg)
    }

    /// Compute the live `SelectionBoxSnapshot` served synchronously over Craft Link v1
    /// (`biomol.get_selection_box`) to `DockCraft`! Priority:
    /// 1. Explicit editable `GridBox3D` in `ViewDockX` or `render_opts.grid_box`
    /// 2. Currently selected atoms (`selected_atoms`) in active model + `padding_a`
    /// 3. Bound ligand atoms (`ResidueClass::Ligand`) in active model + `padding_a`
    #[must_use]
    pub fn current_selection_box_snapshot(&self, padding_a: f64) -> Option<SelectionBoxSnapshot> {
        let pad = padding_a.clamp(0.0, 20.0);
        if let Some(ref vd) = self.viewdock
            && let Some(gb) = vd.editable_grid_box
        {
            return Some(SelectionBoxSnapshot {
                grid_box: gb,
                padding_a: pad,
                source_model: vd.bundle.receptor_name.clone(),
                selected_residues: vec!["ViewDockX GridBox".to_string()],
                heavy_atoms: vd
                    .pose_models
                    .get(vd.active_pose_idx)
                    .and_then(Option::as_ref)
                    .map_or(0, |m| m.atoms.len()),
            });
        }
        if let Some(gb) = self.render_opts.grid_box {
            return Some(SelectionBoxSnapshot {
                grid_box: gb,
                padding_a: pad,
                source_model: self
                    .active_model()
                    .map_or_else(|| "scene".to_string(), |m| m.name.clone()),
                selected_residues: vec!["Custom 3D GridBox".to_string()],
                heavy_atoms: 0,
            });
        }
        let m = self.active_model()?;
        let mut indices: Vec<usize> = m
            .render_opts
            .selected_atoms
            .iter()
            .enumerate()
            .filter(|&(_, &sel)| sel)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            indices = m
                .molecule
                .atoms
                .iter()
                .enumerate()
                .filter(|(_, a)| a.res_class == ResidueClass::Ligand && a.element != "H")
                .map(|(i, _)| i)
                .collect();
        }
        if indices.is_empty() {
            return None;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut residues = Vec::new();
        let mut heavy = 0usize;
        for &ai in &indices {
            let Some(a) = m.molecule.atoms.get(ai) else {
                continue;
            };
            if a.element == "H" {
                continue;
            }
            heavy += 1;
            for k in 0..3 {
                lo[k] = lo[k].min(a.coords[k]);
                hi[k] = hi[k].max(a.coords[k]);
            }
            let tag = format!("{}:{} {}", a.chain, a.res_seq, a.res_name);
            if !residues.contains(&tag) {
                residues.push(tag);
            }
        }
        if heavy == 0 {
            return None;
        }
        let center = [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ];
        let size = [
            ((hi[0] - lo[0]) + 2.0 * pad).clamp(16.0, 56.0),
            ((hi[1] - lo[1]) + 2.0 * pad).clamp(16.0, 56.0),
            ((hi[2] - lo[2]) + 2.0 * pad).clamp(16.0, 56.0),
        ];
        Some(SelectionBoxSnapshot {
            grid_box: GridBox3D {
                center,
                size,
                spacing: 0.375,
            },
            padding_a: pad,
            source_model: m.name.clone(),
            selected_residues: residues,
            heavy_atoms: heavy,
        })
    }

    /// Build the unified multi-model + Cryo-EM Volume + ViewDockX 3D viewport scene.
    #[must_use]
    pub fn build_viewport_scene(&self) -> RenderSceneData {
        let mut scene = RenderSceneData {
            background: self.render_opts.background,
            lighting: self.render_opts.lighting,
            ssao_strength: self.render_opts.ssao_strength,
            show_silhouettes: self.render_opts.show_silhouettes,
            ..Default::default()
        };

        for m in &self.models {
            if !m.visible {
                continue;
            }
            let mut opts = m.render_opts.clone();
            opts.background = self.render_opts.background;
            opts.lighting = self.render_opts.lighting;
            opts.ssao_strength = self.render_opts.ssao_strength;
            opts.show_silhouettes = self.render_opts.show_silhouettes;
            scene.merge_from(build_render_scene(&m.molecule, &opts));
        }

        // Volumetric Cryo-EM / APBS Marching-Tetrahedra isosurface mesh
        if let Some(ref vol) = self.active_volume {
            let iso = (vol.mean + self.active_volume_sigma * vol.std_dev) as f32;
            let vol_mesh = extract_isosurface_mesh(
                vol,
                iso,
                |_| [0.24, 0.78, 0.96],
                if self.volume_wireframe { 0.85 } else { 0.36 },
                self.volume_wireframe,
                1,
            );
            if !vol_mesh.indices.is_empty() {
                scene.transparent_meshes.push(vol_mesh);
            }
        }

        // ViewDockX pose & rich SBDD overlays (HYDE coronas, torsion traffic-lights, interactions,
        // pharmacophores, ΔMIF selectivity hotspots, bridging waters, editable 3D GridBox)
        if let Some(ref vd) = self.viewdock {
            if let Some(Some(pose_mol)) = vd.pose_models.get(vd.active_pose_idx) {
                let pose_opts = RenderOptions {
                    representation: RepresentationKind::Sticks,
                    color_scheme: ColorScheme::ElementCpk,
                    show_surface_shell: false,
                    show_ligands_sticks: true,
                    background: self.render_opts.background,
                    lighting: self.render_opts.lighting,
                    ssao_strength: self.render_opts.ssao_strength,
                    show_silhouettes: self.render_opts.show_silhouettes,
                    ..RenderOptions::default()
                };
                scene.merge_from(build_render_scene(pose_mol, &pose_opts));
            }
            if let Some(pose_entry) = vd.bundle.poses.get(vd.active_pose_idx) {
                if vd.show_hyde_coronas {
                    for c in &pose_entry.hyde_coronas {
                        scene.spheres.push(SphereInstance {
                            center: [c.coords[0] as f32, c.coords[1] as f32, c.coords[2] as f32],
                            radius: c.corona_radius_a.clamp(0.45, 2.2),
                            color: [
                                c.corona_rgba[0],
                                c.corona_rgba[1],
                                c.corona_rgba[2],
                                c.corona_rgba[3].clamp(0.22, 0.65),
                            ],
                        });
                    }
                }
                if vd.show_torsion_lights {
                    for tl in &pose_entry.torsion_lights {
                        scene.capsules.push(CapsuleInstance {
                            a: [tl.from[0] as f32, tl.from[1] as f32, tl.from[2] as f32],
                            radius: 0.27,
                            b: [tl.to[0] as f32, tl.to[1] as f32, tl.to[2] as f32],
                            dashed: 0.0,
                            color: tl.rgba,
                        });
                    }
                }
                if vd.show_interactions {
                    for ln in &pose_entry.interaction_lines {
                        scene.capsules.push(CapsuleInstance {
                            a: [ln.from[0] as f32, ln.from[1] as f32, ln.from[2] as f32],
                            radius: 0.11,
                            b: [ln.to[0] as f32, ln.to[1] as f32, ln.to[2] as f32],
                            dashed: if ln.dashed { 1.0 } else { 0.0 },
                            color: ln.rgba,
                        });
                    }
                }
            }
            if vd.show_pharmacophores {
                for sp in &vd.bundle.pharmacophore_spheres {
                    scene.spheres.push(SphereInstance {
                        center: [
                            sp.coords[0] as f32,
                            sp.coords[1] as f32,
                            sp.coords[2] as f32,
                        ],
                        radius: sp.radius_a as f32,
                        color: sp.rgba,
                    });
                }
                for ln in &vd.bundle.pharmacophore_vectors {
                    scene.capsules.push(CapsuleInstance {
                        a: [ln.from[0] as f32, ln.from[1] as f32, ln.from[2] as f32],
                        radius: 0.14,
                        b: [ln.to[0] as f32, ln.to[1] as f32, ln.to[2] as f32],
                        dashed: 0.0,
                        color: ln.rgba,
                    });
                }
            }
            if vd.show_dmif_hotspots {
                for sp in &vd.bundle.selectivity_dmif_hotspots {
                    scene.spheres.push(SphereInstance {
                        center: [
                            sp.coords[0] as f32,
                            sp.coords[1] as f32,
                            sp.coords[2] as f32,
                        ],
                        radius: sp.radius_a as f32,
                        color: sp.rgba,
                    });
                }
            }
            if vd.show_bridging_waters {
                for sp in &vd.bundle.bridging_waters {
                    scene.spheres.push(SphereInstance {
                        center: [
                            sp.coords[0] as f32,
                            sp.coords[1] as f32,
                            sp.coords[2] as f32,
                        ],
                        radius: sp.radius_a as f32,
                        color: sp.rgba,
                    });
                }
            }
            if vd.show_grid_box
                && let Some(gb) = vd.editable_grid_box
            {
                push_grid_box_wireframe(&mut scene, &gb, [0.98, 0.82, 0.22, 0.95]);
            }
        }

        scene
    }

    /// Save the current multi-model camera/style/clip/ViewDock state into a named or `f1..f12` scene.
    pub fn save_scene(&mut self, name: &str) -> String {
        let key = name.trim().to_ascii_lowercase();
        let snap = SavedScene {
            name: key.clone(),
            camera: self.camera.clone(),
            lighting: self.render_opts.lighting,
            background: self.render_opts.background,
            ssao_strength: self.render_opts.ssao_strength,
            show_silhouettes: self.render_opts.show_silhouettes,
            model_states: self
                .models
                .iter()
                .map(|m| {
                    (
                        m.model_id,
                        m.visible,
                        m.render_opts.representation,
                        m.render_opts.color_scheme,
                        m.render_opts.show_surface_shell,
                    )
                })
                .collect(),
            active_dock_pose: self.viewdock.as_ref().map(|v| v.active_pose_idx),
        };
        self.scenes.insert(key.clone(), snap);
        format!("Saved scene '{key}' ({} models).", self.models.len())
    }

    /// Recall a previously saved scene (`F1..F12` or named scene).
    pub fn recall_scene(&mut self, name: &str) -> Result<String, ScriptError> {
        let key = name.trim().to_ascii_lowercase();
        let Some(snap) = self.scenes.get(&key).cloned() else {
            return Err(ScriptError::Command(format!(
                "scene '{key}' not found (saved: {:?})",
                self.scenes.keys().collect::<Vec<_>>()
            )));
        };
        self.camera = snap.camera;
        self.render_opts.lighting = snap.lighting;
        self.render_opts.background = snap.background;
        self.render_opts.ssao_strength = snap.ssao_strength;
        self.render_opts.show_silhouettes = snap.show_silhouettes;
        for (mid, vis, repr, col, surf) in snap.model_states {
            if let Some(m) = self.models.iter_mut().find(|m| m.model_id == mid) {
                m.visible = vis;
                m.render_opts.representation = repr;
                m.render_opts.color_scheme = col;
                m.render_opts.show_surface_shell = surf;
            }
        }
        if let Some(pidx) = snap.active_dock_pose
            && let Some(ref mut vd) = self.viewdock
            && pidx < vd.bundle.poses.len()
        {
            vd.active_pose_idx = pidx;
        }
        self.pull_active_render_opts();
        self.dirty_geometry = true;
        Ok(format!("Recalled scene '{key}'."))
    }

    #[must_use]
    pub fn to_craftscene_document(&self) -> CraftSceneDocument {
        let active = self.active_model();
        CraftSceneDocument {
            schema_version: "craftscene/v1".to_string(),
            title: active.map_or_else(|| "Empty Scene".to_string(), |m| m.molecule.title.clone()),
            source_app: "biomolcraft".to_string(),
            pdb_path: active.and_then(|m| m.source_path.clone()),
            inline_pdb: active.map(|m| write_pdb(&m.molecule)),
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
                "viewdockx".to_string(),
            ],
            provenance: self.provenance.clone(),
        }
    }

    fn push_undo_snapshot(&mut self) {
        if let Some(m) = self.active_model_mut() {
            m.undo_stack.push(m.molecule.clone());
            if m.undo_stack.len() > 25 {
                m.undo_stack.remove(0);
            }
            m.redo_stack.clear();
        }
    }

    fn parse_model_id_arg(s: &str) -> Option<u32> {
        s.trim().trim_start_matches('#').parse::<u32>().ok()
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
                "Commands: open <path> [add|replace] | fetch <1N8Z|af:UNIPROT|emdb:ID|pubchem:CID> | models | active <#id> | show <#id> | hide <#id> | close <#id|all> | scene <save|recall|list|delete> <name> | clip <near_a> <far_a> | clip reset | lighting <soft|flat|full|studio|pub_white> | preset <pub_white|pub_dark|chimerax_soft|pymol_ray> | ssao <0..1> | bg <dark|white> | select <name>, <expr> | style <cartoon|nucleic|sticks|spheres|surface|lines> | color <chain|ss|cpk|nucleotide|plddt|sap|esp|mlp> | surface <on|off|wireframe> | silhouettes <on|off> | export_mesh <out.glb|out.obj> | load_dock <bundle.json> | pose <idx> | gridbox <from_selection [pad] | cx cy cz sx sy sz> | align <mob_ch>, <ref_ch> | mutate <C:RES> <AA> | minimize [steps] | module <antibodies|ncrna|enzymes|adc> | anm | gbsa | md | molmap | open_map | fitmap | png <path> [W] [H] [transparent] | save <path> | undo | redo | reset"
                    .to_string(),
            ),
            "open" | "load" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: open <path.pdb|cif|sdf> [add|replace]".to_string(),
                    ));
                }
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let path_s = toks[0].trim_matches('"');
                let mode = toks.get(1).copied().unwrap_or("");
                let add_mode = mode.eq_ignore_ascii_case("add")
                    || (mode.is_empty() && !self.models.is_empty());
                if mode.eq_ignore_ascii_case("replace") || !add_mode {
                    self.load_structure_file(Path::new(path_s))
                } else {
                    self.add_structure_file(Path::new(path_s))
                }
            }
            "fetch" => {
                let asset = fetch_structure_spec(rest).map_err(ScriptError::Command)?;
                let tag = if asset.cached_hit { "cache hit" } else { "downloaded" };
                if asset.is_volume_map {
                    let bytes = std::fs::read(&asset.local_path)?;
                    let grid = parse_mrc_map(&asset.canonical_id, &bytes)
                        .map_err(|e| ScriptError::Command(e.to_string()))?;
                    self.active_volume = Some(grid);
                    self.dirty_geometry = true;
                    Ok(format!(
                        "Fetched {} {} ({tag}) -> {} (Cryo-EM volume active).",
                        asset.provider,
                        asset.canonical_id,
                        asset.local_path.display()
                    ))
                } else {
                    let msg = self.add_structure_file(&asset.local_path)?;
                    Ok(format!(
                        "Fetched {} {} ({tag}) -> {msg}",
                        asset.provider, asset.canonical_id
                    ))
                }
            }
            "models" => {
                if self.models.is_empty() {
                    return Ok("No models loaded.".to_string());
                }
                let lines: Vec<String> = self
                    .models
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        format!(
                            "{} #{} '{}' [{}] ({} atoms, vis={}, style={}, color={})",
                            if i == self.active_model_idx { "*" } else { " " },
                            m.model_id,
                            m.name,
                            m.role,
                            m.molecule.atoms.len(),
                            m.visible,
                            m.render_opts.representation.label(),
                            m.render_opts.color_scheme.label()
                        )
                    })
                    .collect();
                Ok(lines.join("\n"))
            }
            "active" => {
                let id = Self::parse_model_id_arg(rest)
                    .ok_or_else(|| ScriptError::Command("usage: active <#id>".to_string()))?;
                if let Some(pos) = self.models.iter().position(|m| m.model_id == id) {
                    self.active_model_idx = pos;
                    self.pull_active_render_opts();
                    Ok(format!("Active model switched to #{id}."))
                } else {
                    Err(ScriptError::Command(format!("model #{id} not found")))
                }
            }
            "hide" => {
                let id = Self::parse_model_id_arg(rest)
                    .ok_or_else(|| ScriptError::Command("usage: hide <#id>".to_string()))?;
                if let Some(m) = self.models.iter_mut().find(|m| m.model_id == id) {
                    m.visible = false;
                    self.dirty_geometry = true;
                    Ok(format!("Hidden model #{id}."))
                } else {
                    Err(ScriptError::Command(format!("model #{id} not found")))
                }
            }
            "close" => {
                if rest.eq_ignore_ascii_case("all") {
                    self.models.clear();
                    self.viewdock = None;
                    self.active_volume = None;
                    self.active_model_idx = 0;
                    self.next_model_id = 1;
                    self.dirty_geometry = true;
                    return Ok("Closed all models.".to_string());
                }
                let id = Self::parse_model_id_arg(rest)
                    .ok_or_else(|| ScriptError::Command("usage: close <#id|all>".to_string()))?;
                if let Some(pos) = self.models.iter().position(|m| m.model_id == id) {
                    self.models.remove(pos);
                    if self.active_model_idx >= self.models.len() {
                        self.active_model_idx = self.models.len().saturating_sub(1);
                    }
                    self.pull_active_render_opts();
                    self.dirty_geometry = true;
                    Ok(format!("Closed model #{id}."))
                } else {
                    Err(ScriptError::Command(format!("model #{id} not found")))
                }
            }
            "scene" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                match toks.first().copied().unwrap_or("").to_ascii_lowercase().as_str() {
                    "save" => {
                        let name = toks.get(1).copied().unwrap_or("f1");
                        Ok(self.save_scene(name))
                    }
                    "recall" | "load" | "restore" => {
                        let name = toks.get(1).copied().unwrap_or("f1");
                        self.recall_scene(name)
                    }
                    "delete" | "rm" => {
                        let name = toks.get(1).copied().unwrap_or("").to_ascii_lowercase();
                        if self.scenes.remove(&name).is_some() {
                            Ok(format!("Deleted scene '{name}'."))
                        } else {
                            Err(ScriptError::Command(format!("scene '{name}' not found")))
                        }
                    }
                    "list" | "" => {
                        if self.scenes.is_empty() {
                            Ok("No saved scenes (use `scene save f1`).".to_string())
                        } else {
                            let names: Vec<&str> = self.scenes.keys().map(String::as_str).collect();
                            Ok(format!("Saved scenes: {}", names.join(", ")))
                        }
                    }
                    other => Err(ScriptError::Command(format!(
                        "unknown scene subcommand '{other}' (use save|recall|list|delete)"
                    ))),
                }
            }
            "clip" => {
                if rest.eq_ignore_ascii_case("reset") || rest.eq_ignore_ascii_case("off") {
                    self.camera.clip_slab_enabled = false;
                    return Ok("Clipping slab disabled (full view restored).".to_string());
                }
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let near_a: f32 = toks
                    .first()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(12.0)
                    .clamp(1.0, 200.0);
                let far_a: f32 = toks
                    .get(1)
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(near_a)
                    .clamp(1.0, 200.0);
                self.camera.clip_slab_enabled = true;
                self.camera.clip_near_a = near_a;
                self.camera.clip_far_a = far_a;
                Ok(format!(
                    "Clipping slab active: front -{near_a:.1} Å / back +{far_a:.1} Å around camera target."
                ))
            }
            "lighting" => {
                let l = match rest.to_ascii_lowercase().as_str() {
                    "soft" => LightingPreset::Soft,
                    "flat" => LightingPreset::Flat,
                    "full" => LightingPreset::Full,
                    "studio" => LightingPreset::Studio,
                    "pub_white" | "white" | "publication" => LightingPreset::PublicationWhite,
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown lighting '{other}' (use soft|flat|full|studio|pub_white)"
                        )));
                    }
                };
                self.render_opts.lighting = l;
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Lighting set to {}.", l.label()))
            }
            "ssao" => {
                let v: f32 = rest
                    .parse()
                    .map_err(|_| ScriptError::Command("usage: ssao <0.0..1.0>".to_string()))?;
                self.render_opts.ssao_strength = v.clamp(0.0, 1.0);
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("SSAO strength set to {:.2}.", self.render_opts.ssao_strength))
            }
            "bg" | "background" => {
                match rest.to_ascii_lowercase().as_str() {
                    "white" | "light" => {
                        self.render_opts.background = [1.0, 1.0, 1.0, 1.0];
                    }
                    "black" => {
                        self.render_opts.background = [0.0, 0.0, 0.0, 1.0];
                    }
                    _ => {
                        self.render_opts.background = [0.06, 0.08, 0.12, 1.0];
                    }
                }
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Background updated ({rest})."))
            }
            "preset" => {
                match rest.to_ascii_lowercase().as_str() {
                    "pub_white" | "pymol_ray" | "publication" => {
                        self.render_opts.background = [1.0, 1.0, 1.0, 1.0];
                        self.render_opts.lighting = LightingPreset::PublicationWhite;
                        self.render_opts.ssao_strength = 0.65;
                        self.render_opts.show_silhouettes = true;
                    }
                    "chimerax_soft" | "soft" => {
                        self.render_opts.background = [0.97, 0.98, 0.99, 1.0];
                        self.render_opts.lighting = LightingPreset::Soft;
                        self.render_opts.ssao_strength = 0.72;
                        self.render_opts.show_silhouettes = true;
                    }
                    "pub_dark" | "studio" => {
                        self.render_opts.background = [0.05, 0.07, 0.11, 1.0];
                        self.render_opts.lighting = LightingPreset::Studio;
                        self.render_opts.ssao_strength = 0.60;
                        self.render_opts.show_silhouettes = true;
                    }
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown preset '{other}' (use pub_white|pub_dark|chimerax_soft|pymol_ray)"
                        )));
                    }
                }
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Applied publication preset '{rest}'."))
            }
            "style" | "show" => {
                if cmd == "show"
                    && let Some(id) = Self::parse_model_id_arg(rest)
                {
                    if let Some(m) = self.models.iter_mut().find(|m| m.model_id == id) {
                        m.visible = true;
                        self.dirty_geometry = true;
                        return Ok(format!("Shown model #{id}."));
                    }
                    return Err(ScriptError::Command(format!("model #{id} not found")));
                }
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
                self.sync_active_render_opts();
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
                    "mlp" | "lipophilicity" => ColorScheme::LipophilicityMlp,
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown color scheme '{other}' (use chain|ss|cpk|nucleotide|plddt|sap|esp|mlp)"
                        )));
                    }
                };
                self.render_opts.color_scheme = c;
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Color scheme set to {}.", c.label()))
            }
            "surface" => {
                let low = rest.to_ascii_lowercase();
                if low == "wireframe" || low == "mesh" || low == "isomesh" {
                    self.render_opts.show_surface_shell = true;
                    self.render_opts.surface_wireframe = true;
                    self.sync_active_render_opts();
                    self.dirty_geometry = true;
                    return Ok("Triangulated SES wireframe mesh enabled.".to_string());
                }
                let on = matches!(low.as_str(), "on" | "true" | "1" | "");
                self.render_opts.show_surface_shell = on;
                self.render_opts.surface_wireframe = false;
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Smooth SES triangulated surface: {on}."))
            }
            "silhouettes" => {
                let on = matches!(rest.to_ascii_lowercase().as_str(), "on" | "true" | "1" | "");
                self.render_opts.show_silhouettes = on;
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Silhouettes: {on}."))
            }
            "waters" => {
                let on = matches!(rest.to_ascii_lowercase().as_str(), "on" | "true" | "1" | "");
                self.render_opts.show_waters = on;
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Show waters: {on}."))
            }
            "export_mesh" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: export_mesh <out.glb|out.obj>".to_string(),
                    ));
                }
                let out = Path::new(rest);
                if let Some(p) = out.parent()
                    && !p.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(p)?;
                }
                let scene = self.build_viewport_scene();
                let title = self
                    .active_molecule()
                    .map_or_else(|| "BiomolCraft".to_string(), |m| m.title.clone());
                if rest.to_ascii_lowercase().ends_with(".obj") {
                    let obj = scene.export_obj(&title);
                    std::fs::write(out, &obj)?;
                    Ok(format!(
                        "Exported Wavefront OBJ mesh to {} ({} bytes).",
                        out.display(),
                        obj.len()
                    ))
                } else {
                    let glb = scene.export_glb(&title);
                    std::fs::write(out, &glb)?;
                    Ok(format!(
                        "Exported binary glTF 2.0 (.glb) mesh to {} ({} bytes).",
                        out.display(),
                        glb.len()
                    ))
                }
            }
            "load_dock" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: load_dock <dockbundle.json>".to_string(),
                    ));
                }
                let raw = std::fs::read_to_string(rest.trim_matches('"'))?;
                let bundle: DockCraftSceneBundle = serde_json::from_str(&raw)
                    .map_err(|e| ScriptError::Command(format!("invalid dock bundle JSON: {e}")))?;
                self.load_dockcraft_bundle(bundle)
            }
            "pose" => {
                let idx: usize = rest
                    .parse()
                    .map_err(|_| ScriptError::Command("usage: pose <0-based index>".to_string()))?;
                self.select_dock_pose(idx)
            }
            "gridbox" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.first().copied() == Some("off") {
                    self.render_opts.grid_box = None;
                    if let Some(ref mut vd) = self.viewdock {
                        vd.show_grid_box = false;
                    }
                    self.sync_active_render_opts();
                    self.dirty_geometry = true;
                    return Ok("GridBox hidden.".to_string());
                }
                if toks.first().copied() == Some("from_selection") {
                    let pad: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(5.0);
                    let snap = self.current_selection_box_snapshot(pad).ok_or_else(|| {
                        ScriptError::Command(
                            "select residues or load a ligand before `gridbox from_selection`"
                                .to_string(),
                        )
                    })?;
                    self.render_opts.grid_box = Some(snap.grid_box);
                    if let Some(ref mut vd) = self.viewdock {
                        vd.editable_grid_box = Some(snap.grid_box);
                        vd.show_grid_box = true;
                    }
                    self.sync_active_render_opts();
                    self.dirty_geometry = true;
                    return Ok(format!(
                        "GridBox set from selection: center=[{:.2},{:.2},{:.2}] size=[{:.1},{:.1},{:.1}] Å.",
                        snap.grid_box.center[0],
                        snap.grid_box.center[1],
                        snap.grid_box.center[2],
                        snap.grid_box.size[0],
                        snap.grid_box.size[1],
                        snap.grid_box.size[2]
                    ));
                }
                if toks.len() < 6 {
                    return Err(ScriptError::Command(
                        "usage: gridbox <cx cy cz sx sy sz | from_selection [pad] | off>"
                            .to_string(),
                    ));
                }
                let nums: Vec<f64> = toks
                    .iter()
                    .take(6)
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if nums.len() < 6 {
                    return Err(ScriptError::Command("invalid numeric GridBox coords".to_string()));
                }
                let gb = GridBox3D {
                    center: [nums[0], nums[1], nums[2]],
                    size: [
                        nums[3].clamp(8.0, 64.0),
                        nums[4].clamp(8.0, 64.0),
                        nums[5].clamp(8.0, 64.0),
                    ],
                    spacing: 0.375,
                };
                self.render_opts.grid_box = Some(gb);
                if let Some(ref mut vd) = self.viewdock {
                    vd.editable_grid_box = Some(gb);
                    vd.show_grid_box = true;
                }
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!(
                    "GridBox set to center={:?}, size={:?} Å.",
                    gb.center, gb.size
                ))
            }
            "select" | "sele" => {
                let mol = self
                    .active_molecule()
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
                self.sync_active_render_opts();
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
                    .active_molecule_mut()
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
                self.render_opts
                    .sphere_overlays
                    .extend(rep.clash_overlays.clone());
                self.sync_active_render_opts();
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
                } else if let Some(m) = self.active_molecule() {
                    vec![true; m.atoms.len()]
                } else {
                    return Err(ScriptError::Command("no biomolecule loaded".to_string()));
                };
                let mol = self
                    .active_molecule_mut()
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
                        "usage: align <mobile_chain|#2:B>, <target_chain|#1:B>".to_string(),
                    ));
                }
                // Cross-model alignment support: `align #2:B #1:B`
                if toks[0].starts_with('#') && toks[1].starts_with('#') {
                    let parse_spec = |s: &str| -> Result<(u32, char), ScriptError> {
                        let (m_s, c_s) = s.split_once(':').ok_or_else(|| {
                            ScriptError::Command("expected #MODEL:CHAIN (e.g. #2:B)".to_string())
                        })?;
                        let mid = Self::parse_model_id_arg(m_s)
                            .ok_or_else(|| ScriptError::Command("bad model id".to_string()))?;
                        let ch = c_s
                            .chars()
                            .next()
                            .ok_or_else(|| ScriptError::Command("bad chain id".to_string()))?;
                        Ok((mid, ch))
                    };
                    let (m_id, mc) = parse_spec(toks[0])?;
                    let (t_id, tc) = parse_spec(toks[1])?;
                    let target_mol = self
                        .models
                        .iter()
                        .find(|m| m.model_id == t_id)
                        .map(|m| m.molecule.clone())
                        .ok_or_else(|| ScriptError::Command(format!("model #{t_id} not found")))?;
                    let mob = self
                        .models
                        .iter_mut()
                        .find(|m| m.model_id == m_id)
                        .ok_or_else(|| ScriptError::Command(format!("model #{m_id} not found")))?;
                    let tf = matchmaker_align(&mut mob.molecule, mc, &target_mol, tc)
                        .map_err(|e| ScriptError::Command(e.to_string()))?;
                    self.focus_camera_all_visible();
                    self.dirty_geometry = true;
                    return Ok(format!(
                        "Aligned #{m_id}:{mc} onto #{t_id}:{tc}: RMSD = {:.4} Å ({} pairs).",
                        tf.rmsd_a, tf.aligned_pairs
                    ));
                }
                let mc = toks[0]
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("bad mobile chain".to_string()))?;
                let tc = toks[1]
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("bad target chain".to_string()))?;
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
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
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let key = rest.to_ascii_lowercase();
                let visual = match key.as_str() {
                    "antibodies" | "ab" | "vhh" => run_antibody_module(&mol, None, None)
                        .map_err(ScriptError::Command)?
                        .visual,
                    "ncrna" | "rna" | "mrna" => run_ncrna_module(&mol)
                        .map_err(ScriptError::Command)?
                        .0,
                    "enzymes" | "enzyme" => run_enzyme_module(&mol),
                    "adc" => run_adc_module(&mol, "ggfg", "dxd", 8.0, false)
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
                self.sync_active_render_opts();
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
                Ok(format!(
                    "Rendered {}x{} PNG to {} ({bytes} bytes).",
                    w,
                    h,
                    out.display()
                ))
            }
            "save" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: save <out.pdb|out.cif|out.craftscene.json>".to_string(),
                    ));
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
                        .active_molecule()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    std::fs::write(out, write_mmcif(mol))?;
                    Ok(format!("Saved mmCIF to {}.", out.display()))
                } else {
                    let mol = self
                        .active_molecule()
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
                let (vis, rep, frames_len) = {
                    let mol = self
                        .active_molecule_mut()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    let (vis, rep) =
                        run_physics_anm_module(mol, mode, amp, 20).map_err(ScriptError::Command)?;
                    let fl = mol.frames.len();
                    (vis, rep, fl)
                };
                self.render_opts.residue_highlights = vis.residue_highlights;
                self.render_opts.line_overlays = vis.line_overlays;
                self.sync_active_render_opts();
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "ANM mode {mode} animated across {frames_len} frames ({} nodes, {} hinges).",
                    rep.guide_nodes,
                    rep.hinge_residues.len()
                ))
            }
            "gbsa" | "dipole" => {
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let (vis, rep) = run_physics_gbsa_module(&mol);
                self.render_opts.color_scheme = ColorScheme::ElectrostaticEsp;
                self.render_opts.line_overlays = vis.line_overlays;
                self.sync_active_render_opts();
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
                    .active_molecule_mut()
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
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                if mol.set_active_frame(idx) {
                    let tot = mol.frames.len();
                    self.dirty_geometry = true;
                    Ok(format!("Switched to trajectory frame {}/{tot}.", idx + 1))
                } else {
                    Err(ScriptError::Command(format!(
                        "frame {idx} out of range (loaded frames: {})",
                        mol.frames.len()
                    )))
                }
            }
            "load_dcd" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: load_dcd <trajectory.dcd>".to_string(),
                    ));
                }
                let bytes = std::fs::read(rest)?;
                let mol = self
                    .active_molecule_mut()
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
                let (vis, grid, fit) = {
                    let mol = self
                        .active_molecule_mut()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    run_cryoem_molmap_module(mol, res, sigma).map_err(ScriptError::Command)?
                };
                self.active_volume = Some(grid);
                self.active_volume_sigma = sigma;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "Synthesized {:.1} Å Cryo-EM molmap ({:.1}σ Marching-Tetrahedra mesh): CC={:.3}, {:.1}% atoms >= 1σ.",
                    res, sigma, fit.correlation_after, fit.atoms_inside_1sigma_pct
                ))
            }
            "open_map" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let path_s = toks.first().copied().ok_or_else(|| {
                    ScriptError::Command("usage: open_map <map.mrc|map.dx> [sigma]".to_string())
                })?;
                let sigma: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.5);
                let grid = if path_s.to_ascii_lowercase().ends_with(".dx") {
                    let txt = std::fs::read_to_string(path_s)?;
                    parse_dx_map(path_s, &txt).map_err(|e| ScriptError::Command(e.to_string()))?
                } else {
                    let bytes = std::fs::read(path_s)?;
                    parse_mrc_map(path_s, &bytes)
                        .map_err(|e| ScriptError::Command(e.to_string()))?
                };
                let (vis, fit) = {
                    let mol = self
                        .active_molecule_mut()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    run_cryoem_volume_module(mol, &grid, sigma, false)
                };
                self.active_volume = Some(grid);
                self.active_volume_sigma = sigma;
                self.last_module_markdown = vis.summary_markdown;
                self.provenance.extend(vis.provenance);
                self.dirty_geometry = true;
                Ok(format!(
                    "Loaded volumetric map '{path_s}' ({sigma:.1}σ Marching-Tetrahedra mesh): CC={:.3}.",
                    fit.correlation_after
                ))
            }
            "fitmap" => {
                let grid = self.active_volume.clone().ok_or_else(|| {
                    ScriptError::Command(
                        "no active volume map (run `molmap` or `open_map` first)".to_string(),
                    )
                })?;
                let sig = self.active_volume_sigma;
                let (vis, fit) = {
                    let mol = self
                        .active_molecule_mut()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    run_cryoem_volume_module(mol, &grid, sig, true)
                };
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
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let (vis, plan) = run_denovo_cofold_module(&mol, ch, &hotspots, None)
                    .map_err(ScriptError::Command)?;
                self.render_opts.residue_highlights = vis.residue_highlights;
                self.render_opts.sphere_overlays = vis.sphere_overlays;
                self.sync_active_render_opts();
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
                    return Err(ScriptError::Command(
                        "usage: export_html <out.html>".to_string(),
                    ));
                }
                let mol = self
                    .active_molecule()
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
                    .active_molecule()
                    .map_or_else(|| "BiomolCraft Scene".to_string(), |m| m.title.clone());
                let atom_cnt = self.active_molecule().map_or(0, |m| m.atoms.len());
                let scene = self.build_viewport_scene();
                let png = render_scene_png_bytes(&scene, &self.camera, 1600, 1200, false)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let rows = vec![
                    ("Structure", title.clone()),
                    ("Atoms", atom_cnt.to_string()),
                    (
                        "Representation",
                        self.render_opts.representation.label().to_string(),
                    ),
                    (
                        "Color Scheme",
                        self.render_opts.color_scheme.label().to_string(),
                    ),
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
                if let Some(m) = self.active_model_mut()
                    && let Some(prev) = m.undo_stack.pop()
                {
                    let cur = std::mem::replace(&mut m.molecule, prev);
                    m.redo_stack.push(cur);
                    self.dirty_geometry = true;
                    Ok("Undid last 3D structural edit.".to_string())
                } else {
                    Ok("Nothing to undo.".to_string())
                }
            }
            "redo" => {
                if let Some(m) = self.active_model_mut()
                    && let Some(next) = m.redo_stack.pop()
                {
                    let cur = std::mem::replace(&mut m.molecule, next);
                    m.undo_stack.push(cur);
                    self.dirty_geometry = true;
                    Ok("Redid 3D structural edit.".to_string())
                } else {
                    Ok("Nothing to redo.".to_string())
                }
            }
            "reset" | "center" => {
                self.camera.clip_slab_enabled = false;
                self.focus_camera_all_visible();
                self.render_opts.selected_atoms.clear();
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok("Centered camera across all models and cleared selection.".to_string())
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
    use craftbio_scene::DockPoseEntry;

    #[test]
    fn multi_model_scenes_clip_glb_and_viewdockx_workflow() -> Result<(), Box<dyn std::error::Error>>
    {
        let pdb_a = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A   1       2.050   1.380   0.000  1.00 90.00           C
ATOM      4  O   ALA A   1       1.350   2.380   0.000  1.00 90.00           O
ATOM      5  CA  TYR A   2       4.100   2.680   0.200  1.00 88.00           C
END
";
        let pdb_b = "\
ATOM      1  N   VAL B   1       6.000   0.000   0.000  1.00 85.00           N
ATOM      2  CA  VAL B   1       7.450   0.000   0.000  1.00 85.00           C
END
";
        let mut sess = BiomolSession::default();
        sess.load_structure_text("model_a", pdb_a, None)?;
        sess.add_structure_text("model_b", pdb_b, None)?;
        assert_eq!(sess.models.len(), 2);

        let _ = sess.execute_dsl("preset pub_white; color mlp; scene save f1")?;
        let _ = sess.execute_dsl("hide #1; clip 5 5; color esp")?;
        assert!(!sess.models[0].visible);
        assert!(sess.camera.clip_slab_enabled);
        let _ = sess.execute_dsl("scene recall f1")?;
        assert!(sess.models[0].visible);
        assert!(!sess.camera.clip_slab_enabled);

        // ViewDockX bundle + synchronous SelectionBoxSnapshot test
        let sdf = "\
LigPose1
  DockCraft

  2  1  0  0  0  0  0  0  0  0999 V2000
    2.0000    1.0000    0.5000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.4000    1.0000    0.5000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
M  END
$$$$
";
        let bundle = DockCraftSceneBundle {
            title: "UnitTest ViewDockX".to_string(),
            receptor_name: "model_a".to_string(),
            receptor_pdb_inline: Some(pdb_a.to_string()),
            grid_box: Some(GridBox3D {
                center: [2.5, 1.0, 0.5],
                size: [18.0, 18.0, 18.0],
                spacing: 0.375,
            }),
            poses: vec![DockPoseEntry {
                pose_id: 1,
                name: "LigPose1".to_string(),
                canonical_smiles: None,
                sdf_block: sdf.to_string(),
                vina_kcal: -8.9,
                hyde_kcal: Some(-9.4),
                consensus_kcal: Some(-9.1),
                cnn_score: None,
                estimated_ki_nm: Some(120.0),
                ligand_efficiency: Some(0.41),
                mpo_score: Some(5.0),
                posebusters_valid: Some(true),
                posebusters_summary: Some("5/5 PASS".to_string()),
                torsion_counts: Some([2, 0, 0]),
                hyde_coronas: Vec::new(),
                torsion_lights: Vec::new(),
                interaction_lines: Vec::new(),
            }],
            ..Default::default()
        };
        sess.load_dockcraft_bundle(bundle)?;
        let snap = sess
            .current_selection_box_snapshot(5.0)
            .ok_or("expected GridBox snapshot")?;
        assert!((snap.grid_box.center[0] - 2.5).abs() < 1e-6);
        let sc = sess.build_viewport_scene();
        assert!(!sc.opaque_meshes.is_empty());
        assert!(!sc.capsules.is_empty());
        Ok(())
    }
}
