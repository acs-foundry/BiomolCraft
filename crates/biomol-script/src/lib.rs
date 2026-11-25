#![forbid(unsafe_code)]

//! `biomol-script` — Multi-Model (`#1, #2, #3...`) PyMOL/ChimeraX-Compatible Biomolecular Command
//! Console DSL, Scenes (`F1–F12`), Cached Online `fetch`, Native **`ViewDockX`** Docking Session,
//! Transactional Undo/Redo, and Provenance Evidence Log (shared by GUI, CLI & MCP).

pub mod fetch;
pub mod registry;
pub mod session_zip;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use biomol_model::{
    Aggregation3DReport, RamachandranReport, RotamerTableReport, add_polar_hydrogens_ph,
    aggregation_visual_palette, analyze_aggregation_3d, analyze_ramachandran, audit_structure_prep,
    build_de_novo_peptide_3d, build_de_novo_rna_helix_3d, combine_biomolecules,
    delete_atoms_by_mask, enumerate_rotamers_3d, inspect_structure_info, minimize_local_mm,
    mutate_and_repack_best_rotamer, mutate_residue_3d, parse_aa_code, rename_chain,
    render_aggregation_markdown, render_info_markdown, render_ramachandran_markdown,
    render_rotamers_markdown, renumber_chain, rotate_bond_torsion, simulate_solubility_rescue,
    swapna_nucleotide_3d, tug_atom_with_sculpting,
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
    ArtCraftPushPayload, ArtCraftPushResult, ArtCraftTarget, ColorKeyLegend, ColorScheme,
    CraftSceneDocument, DockCraftSceneBundle, GridBox3D, Label2D, Label3D, LightingPreset,
    LineOverlay, ProvenanceEntry, ProvenanceKind, RepresentationKind, ScaleBarSpec,
    SelectionBoxSnapshot, SphereOverlay, push_to_artcraft_via_craft_link,
};
use craftbio_struct::{
    BindingPocket3D, Biomolecule, GeometryAxis, GeometryPlane, MsaAlignment, PaeMatrix,
    ResidueClass, VolumeGrid3D, axis_angle_deg, build_synchronized_msa, cealign_superpose,
    compute_best_fit_plane, compute_geometry_axis, compute_selection_sasa, detect_binding_pockets,
    detect_clashes, detect_clashes_between, detect_contacts, detect_contacts_between,
    detect_hbonds, detect_hbonds_between, detect_snfg_glycans, estimate_pae_proxy_from_structure,
    evaluate_selection, generate_biological_assembly, generate_crystal_mates, load_dcd_frames,
    load_xtc_frames, matchmaker_align, measure_angle_deg, measure_dihedral_deg, measure_distance,
    morph_biomolecules, parse_dx_map, parse_mrc_map, parse_pae_json, parse_sdf_poses,
    parse_structure_auto, write_mmcif, write_multi_model_pdb, write_pdb,
};
use craftbio_viewport::{
    Camera3D, CapsuleInstance, RenderOptions, RenderSceneData, SphereInstance,
    build_orthoplanes_mesh, build_pockets_mesh, build_render_scene, extract_isosurface_mesh,
    push_grid_box_wireframe, render_fly_frames, render_rock_frames, render_scene_png_bytes,
    render_scene_rgba, render_turntable_frames, write_movie_file, write_scene_png,
};
pub use fetch::{FetchedAsset, fetch_structure_spec};
pub use registry::{
    COMMAND_REGISTRY, COMPUTE_VERBS, CommandSpec, CompletionItem, FILESYSTEM_VERBS,
    autocomplete_dsl, filter_command_palette, render_registry_help, verb_is_sandbox_safe,
    verb_touches_filesystem,
};
use serde::{Deserialize, Serialize};
pub use session_zip::{
    SessionZipSummary, decode_session_zip_bytes, encode_session_zip_bytes, inspect_session_zip,
    load_session_zip, save_session_zip,
};
use thiserror::Error;

/// Split a DSL script into executable statements, exactly as [`BiomolSession::execute_dsl`] does.
///
/// Callers that need to vet a script before running it (the MCP server) must use this
/// function rather than their own splitter: any divergence between the two is a hole.
#[must_use]
pub fn split_dsl_statements(script: &str) -> Vec<&str> {
    script
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .collect()
}

/// Extract the lowercase verb of a single DSL statement, exactly as the interpreter does.
///
/// The verb ends at the first whitespace character of any kind — space, tab, newline,
/// vertical tab — which is why substring checks such as `contains("save ")` do not confine it.
#[must_use]
pub fn dsl_statement_verb(stmt: &str) -> String {
    stmt.trim()
        .split(char::is_whitespace)
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Deepest parenthesis nesting, and longest chain of prefix operators, accepted from an
/// untrusted caller.
///
/// The selection parser in `craftbio-struct` recurses once per nesting level, so a deep
/// enough expression overflows the stack and aborts the process — which no `catch_unwind`
/// can intercept. Hand-written selections nest a handful of levels; 64 is far beyond any
/// real query and far below the thousands needed to exhaust the stack.
pub const MAX_EXPRESSION_NESTING: usize = 64;

/// Verdict of [`vet_dsl_for_sandbox`]: why a script may not run under a sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DslRejection {
    /// The verb reads or writes files (or the network) with a caller-supplied path.
    TouchesFilesystem(String),
    /// The verb is not classified as sandbox-safe (unknown, or newly added and unclassified).
    NotWhitelisted(String),
    /// The expression nests deeply enough to threaten the parser's stack.
    TooDeeplyNested { depth: usize },
}

/// Measure how deeply a statement nests: parenthesis depth, and the length of any run of
/// prefix operators (`not`, `!`), which recurse just as parentheses do but carry no brackets.
#[must_use]
fn expression_nesting(stmt: &str) -> usize {
    let mut depth = 0usize;
    let mut max_depth = 0usize;
    for c in stmt.chars() {
        match c {
            '(' | '[' => {
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            ')' | ']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }

    let mut run = 0usize;
    let mut max_run = 0usize;
    for tok in stmt.split_whitespace() {
        let t = tok.trim_start_matches(['(', '[']).to_ascii_lowercase();
        if t == "not" || t == "!" {
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 0;
        }
    }

    max_depth.max(max_run)
}

impl std::fmt::Display for DslRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TouchesFilesystem(v) => write!(
                f,
                "DSL command '{v}' performs its own file or network I/O and is not available over MCP; \
                 use the dedicated sandboxed tools (`load_biomolecule`, `save_structure_or_scene`, \
                 `render_view_png`, `export_mesh_3d`, `save_biomol_session`, …) instead"
            ),
            Self::NotWhitelisted(v) => write!(
                f,
                "DSL command '{v}' is not on the sandbox-safe command whitelist"
            ),
            Self::TooDeeplyNested { depth } => write!(
                f,
                "expression nests {depth} levels deep (limit {MAX_EXPRESSION_NESTING}); \
                 such an expression would overflow the parser's stack"
            ),
        }
    }
}

/// Vet a whole DSL script for execution on behalf of a sandbox-confined caller.
///
/// Every statement must carry a verb that only touches in-memory session state. Splitting
/// uses the interpreter's own lexer, so a statement hidden behind `;`, a tab after the verb,
/// or odd casing is seen exactly as the interpreter will see it.
///
/// # Errors
/// Returns the first statement whose verb is not sandbox-safe.
pub fn vet_dsl_for_sandbox(script: &str) -> Result<(), DslRejection> {
    for stmt in split_dsl_statements(script) {
        let verb = dsl_statement_verb(stmt);
        if verb_touches_filesystem(&verb) {
            return Err(DslRejection::TouchesFilesystem(verb));
        }
        if !verb_is_sandbox_safe(&verb) {
            return Err(DslRejection::NotWhitelisted(verb));
        }
        let depth = expression_nesting(stmt);
        if depth > MAX_EXPRESSION_NESTING {
            return Err(DslRejection::TooDeeplyNested { depth });
        }
    }
    Ok(())
}

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

fn default_scene_shadow() -> f32 {
    0.55
}

/// Restorable ChimeraX / PyMOL Scene snapshot (`F1–F12` or named scene).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedScene {
    pub name: String,
    pub camera: Camera3D,
    pub lighting: LightingPreset,
    pub background: [f32; 4],
    pub ssao_strength: f32,
    #[serde(default = "default_scene_shadow")]
    pub shadow_strength: f32,
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

#[derive(Debug, Clone)]
struct InteractionQuery {
    cutoff: f64,
    model_a: Option<u32>,
    expr_a: Option<String>,
    model_b: Option<u32>,
    expr_b: Option<String>,
    inter_model: bool,
    intra_model: bool,
}

/// Complete stateful multi-model `BiomolCraft` interactive session (shared by GUI, CLI & MCP).
#[derive(Debug, Clone)]
pub struct BiomolSession {
    pub models: Vec<SceneModel>,
    pub active_model_idx: usize,
    pub(crate) next_model_id: u32,
    pub scenes: BTreeMap<String, SavedScene>,
    pub viewdock: Option<ViewDockSession>,
    pub active_volume: Option<VolumeGrid3D>,
    pub active_volume_sigma: f64,
    pub volume_wireframe: bool,
    pub show_orthoplanes: bool,
    pub orthoplane_frac: [f32; 3],
    pub camera: Camera3D,
    /// Global viewport rendering defaults & active model mirror (`representation`, `color_scheme`, etc.).
    pub render_opts: RenderOptions,
    pub named_selections: HashMap<String, Vec<bool>>,
    pub provenance: Vec<ProvenanceEntry>,
    pub command_history: Vec<String>,
    pub last_module_markdown: String,
    pub measurements_log: Vec<String>,
    pub axes: Vec<GeometryAxis>,
    pub planes: Vec<GeometryPlane>,
    pub msa: Option<MsaAlignment>,
    pub pae: Option<PaeMatrix>,
    pub ramachandran: Option<RamachandranReport>,
    pub aggregation: Option<Aggregation3DReport>,
    pub rotamer_table: Option<RotamerTableReport>,
    pub labels_3d: Vec<Label3D>,
    pub labels_2d: Vec<Label2D>,
    pub color_key: Option<ColorKeyLegend>,
    pub scale_bar: Option<ScaleBarSpec>,
    pub binding_pockets: Vec<BindingPocket3D>,
    pub selected_pocket_id: Option<usize>,
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
            show_orthoplanes: false,
            orthoplane_frac: [0.5, 0.5, 0.5],
            camera: Camera3D::default(),
            render_opts: RenderOptions::default(),
            named_selections: HashMap::new(),
            provenance: Vec::new(),
            command_history: Vec::new(),
            last_module_markdown: String::new(),
            measurements_log: Vec::new(),
            axes: Vec::new(),
            planes: Vec::new(),
            msa: None,
            pae: None,
            ramachandran: None,
            aggregation: None,
            rotamer_table: None,
            labels_3d: Vec::new(),
            labels_2d: Vec::new(),
            color_key: None,
            scale_bar: None,
            binding_pockets: Vec::new(),
            selected_pocket_id: None,
            dirty_geometry: true,
        }
    }
}

impl BiomolSession {
    pub(crate) fn next_model_id_counter(&self) -> u32 {
        self.next_model_id
    }

    /// Export a publication PNG + GridCraft TSV + WordCraft/DeckCraft Markdown bundle and
    /// live-push it over Craft Link v1 to `WordCraft`, `DeckCraft`, or `GridCraft`.
    pub fn push_artcraft_target(
        &mut self,
        target: ArtCraftTarget,
        out_dir: &Path,
    ) -> Result<ArtCraftPushResult, ScriptError> {
        let title = self
            .active_molecule()
            .map_or_else(|| "BiomolCraft Scene".to_string(), |m| m.title.clone());
        let atom_cnt = self.active_molecule().map_or(0, |m| m.atoms.len());
        let scene = self.build_viewport_scene();
        let png = render_scene_png_bytes(&scene, &self.camera, 1600, 1200, false)
            .map_err(|e| ScriptError::Command(e.to_string()))?;
        let mut rows = vec![
            ("Structure", title.clone()),
            ("Models", self.models.len().to_string()),
            ("Active Atoms", atom_cnt.to_string()),
            (
                "Representation",
                self.render_opts.representation.label().to_string(),
            ),
            (
                "Color Scheme",
                self.render_opts.color_scheme.label().to_string(),
            ),
            ("Saved Scenes", self.scenes.len().to_string()),
        ];
        if let Some(ref vd) = self.viewdock {
            rows.push(("ViewDockX Poses", vd.bundle.poses.len().to_string()));
        }
        let sum = export_artcraft_bundle(&title, &png, &rows, &self.last_module_markdown, out_dir)
            .map_err(ScriptError::Command)?;

        let markdown_path = match target {
            ArtCraftTarget::DeckCraft => sum.deckcraft_slides_md.clone(),
            ArtCraftTarget::WordCraft | ArtCraftTarget::GridCraft => sum.wordcraft_md.clone(),
        };
        let native_doc_path = match target {
            ArtCraftTarget::WordCraft => sum.wordcraft_docx.clone(),
            ArtCraftTarget::DeckCraft => sum.deckcraft_pptx.clone(),
            ArtCraftTarget::GridCraft => sum.gridcraft_xlsx.clone(),
        };
        let payload = ArtCraftPushPayload {
            target,
            title,
            source_app: "biomolcraft".to_string(),
            bundle_dir: sum.out_dir,
            figure_png_path: sum.figure_png,
            markdown_path,
            table_tsv_path: sum.gridcraft_tsv,
            native_doc_path,
            summary_markdown: self.last_module_markdown.clone(),
        };
        let res = push_to_artcraft_via_craft_link(None, &payload)
            .map_err(|e| ScriptError::Command(e.to_string()))?;
        self.provenance.push(ProvenanceEntry {
            kind: ProvenanceKind::InteractiveEdit,
            source_app: format!("biomolcraft::push->{}", target.app_id()),
            summary: res.message.clone(),
        });
        Ok(res)
    }

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
            m.render_opts.shadow_strength = g.shadow_strength;
            m.render_opts.background = g.background;
            m.render_opts.residue_highlights = g.residue_highlights;
            m.render_opts.residue_colors = g.residue_colors;
            m.render_opts.sphere_overlays = g.sphere_overlays;
            m.render_opts.line_overlays = g.line_overlays;
            m.render_opts.selected_atoms = g.selected_atoms;
            m.render_opts.grid_box = g.grid_box;
            m.render_opts.show_snfg_glycans = g.show_snfg_glycans;
            m.render_opts.show_unit_cell = g.show_unit_cell;
        }
    }

    fn pull_active_render_opts(&mut self) {
        if let Some(m) = self.models.get(self.active_model_idx) {
            let bg = self.render_opts.background;
            let light = self.render_opts.lighting;
            let ssao = self.render_opts.ssao_strength;
            let shadow = self.render_opts.shadow_strength;
            let sil = self.render_opts.show_silhouettes;
            self.render_opts = m.render_opts.clone();
            self.render_opts.background = bg;
            self.render_opts.lighting = light;
            self.render_opts.ssao_strength = ssao;
            self.render_opts.shadow_strength = shadow;
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
            shadow_strength: self.render_opts.shadow_strength,
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
            self.binding_pockets.clear();
            self.selected_pocket_id = None;
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
    ///
    /// Reads any file the bundle names with the caller's own permissions. Correct for the
    /// desktop app, where the user picked the bundle; callers that handle bundles from an
    /// untrusted source must use [`BiomolSession::load_dockcraft_bundle_with`] instead.
    pub fn load_dockcraft_bundle(
        &mut self,
        bundle: DockCraftSceneBundle,
    ) -> Result<String, ScriptError> {
        self.load_dockcraft_bundle_with(bundle, &|p| std::fs::read(p).map_err(|e| e.to_string()))
    }

    /// Load a `DockCraftSceneBundle`, reading the files it names through `read_file`.
    ///
    /// A bundle is data, not a trusted instruction: its `receptor_pdb_path` and
    /// `trajectory_dcd_path` fields are chosen by whoever wrote the JSON. Passing a
    /// sandbox-confined reader keeps those fields from naming arbitrary files on disk.
    pub fn load_dockcraft_bundle_with(
        &mut self,
        bundle: DockCraftSceneBundle,
        read_file: &dyn Fn(&Path) -> Result<Vec<u8>, String>,
    ) -> Result<String, ScriptError> {
        let rec_raw = if let Some(ref txt) = bundle.receptor_pdb_inline
            && !txt.trim().is_empty()
        {
            txt.clone()
        } else if let Some(ref p) = bundle.receptor_pdb_path {
            let bytes = read_file(Path::new(p)).map_err(ScriptError::Command)?;
            String::from_utf8(bytes)
                .map_err(|_| ScriptError::Command(format!("receptor file '{p}' is not UTF-8")))?
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

        // Optional binary DCD trajectory bundled from DockCraft MD validation.
        // A path the reader refuses is an error, not something to skip quietly: the caller
        // asked for a trajectory and must learn it was not loaded.
        if let Some(ref dcd_p) = bundle.trajectory_dcd_path {
            let bytes = read_file(Path::new(dcd_p)).map_err(ScriptError::Command)?;
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
            shadow_strength: self.render_opts.shadow_strength,
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
            opts.shadow_strength = self.render_opts.shadow_strength;
            opts.show_silhouettes = self.render_opts.show_silhouettes;
            scene.merge_from(build_render_scene(&m.molecule, &opts));
        }

        // Detected 3D Druggable Binding Pocket cavity meshes (Marching Tetrahedra)
        if !self.binding_pockets.is_empty() {
            let pocket_mesh = build_pockets_mesh(&self.binding_pockets, self.selected_pocket_id);
            if !pocket_mesh.indices.is_empty() {
                scene.transparent_meshes.push(pocket_mesh);
            }
        }

        // Volumetric Cryo-EM / APBS Marching-Tetrahedra isosurface mesh + optional Orthoplanes
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
            if self.show_orthoplanes {
                let ortho = build_orthoplanes_mesh(vol, self.orthoplane_frac, 0.56);
                if !ortho.indices.is_empty() {
                    scene.transparent_meshes.push(ortho);
                }
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

        // Geometric PCA Axes (`define axis`) & Best-Fit Planes (`define plane`)
        for ax in &self.axes {
            let sa = [ax.start[0] as f32, ax.start[1] as f32, ax.start[2] as f32];
            let sb = [ax.end[0] as f32, ax.end[1] as f32, ax.end[2] as f32];
            scene.capsules.push(CapsuleInstance {
                a: sa,
                radius: 0.34,
                b: sb,
                dashed: 0.0,
                color: [0.98, 0.78, 0.16, 1.0],
            });
            scene.spheres.push(SphereInstance {
                center: sa,
                radius: 0.48,
                color: [0.98, 0.78, 0.16, 1.0],
            });
            scene.spheres.push(SphereInstance {
                center: sb,
                radius: 0.48,
                color: [0.98, 0.78, 0.16, 1.0],
            });
        }
        for pl in &self.planes {
            let c = [
                pl.centroid[0] as f32,
                pl.centroid[1] as f32,
                pl.centroid[2] as f32,
            ];
            let n = [
                pl.normal[0] as f32,
                pl.normal[1] as f32,
                pl.normal[2] as f32,
            ];
            // Normal arrow + 12-segment perimeter ring representing best-fit plane disc
            let tip = [c[0] + n[0] * 4.5, c[1] + n[1] * 4.5, c[2] + n[2] * 4.5];
            scene.capsules.push(CapsuleInstance {
                a: c,
                radius: 0.16,
                b: tip,
                dashed: 0.0,
                color: [0.22, 0.88, 0.96, 0.95],
            });
            let alt = if n[1].abs() < 0.8 {
                [0.0_f32, 1.0, 0.0]
            } else {
                [1.0_f32, 0.0, 0.0]
            };
            let u = craftbio_viewport::normalize3(craftbio_viewport::cross3(n, alt));
            let v = craftbio_viewport::normalize3(craftbio_viewport::cross3(n, u));
            let rad = (pl.radius_a as f32).clamp(2.0, 24.0);
            let segs = 16usize;
            for s in 0..segs {
                let a0 = (s as f32) * std::f32::consts::TAU / (segs as f32);
                let a1 = ((s + 1) as f32) * std::f32::consts::TAU / (segs as f32);
                let p0 = [
                    c[0] + (u[0] * a0.cos() + v[0] * a0.sin()) * rad,
                    c[1] + (u[1] * a0.cos() + v[1] * a0.sin()) * rad,
                    c[2] + (u[2] * a0.cos() + v[2] * a0.sin()) * rad,
                ];
                let p1 = [
                    c[0] + (u[0] * a1.cos() + v[0] * a1.sin()) * rad,
                    c[1] + (u[1] * a1.cos() + v[1] * a1.sin()) * rad,
                    c[2] + (u[2] * a1.cos() + v[2] * a1.sin()) * rad,
                ];
                scene.capsules.push(CapsuleInstance {
                    a: p0,
                    radius: 0.18,
                    b: p1,
                    dashed: 0.0,
                    color: [0.22, 0.88, 0.96, 0.90],
                });
            }
        }

        // Propagate 3D/2D labels, publication Color Key legend, and calibrated Scale Bar
        scene.labels_3d = self.labels_3d.clone();
        scene.labels_2d = self.labels_2d.clone();
        scene.color_key = self.color_key.clone();
        scene.scale_bar = self.scale_bar.clone();

        scene
    }

    /// Build a standard publication `ColorKeyLegend` matching `kind` (`esp|mlp|sap|plddt|bfactor|rainbow|agg|auto`).
    #[must_use]
    pub fn build_color_key(&self, kind: &str) -> ColorKeyLegend {
        let mode = if kind.is_empty()
            || kind.eq_ignore_ascii_case("auto")
            || kind.eq_ignore_ascii_case("on")
        {
            match self.render_opts.color_scheme {
                ColorScheme::ElectrostaticEsp => "esp",
                ColorScheme::LipophilicityMlp => "mlp",
                ColorScheme::Sap3D => "sap",
                ColorScheme::CrystallographicBfactor => "bfactor",
                ColorScheme::Rainbow => "rainbow",
                _ => "plddt",
            }
        } else {
            kind
        };
        match mode.to_ascii_lowercase().as_str() {
            "esp" | "electrostatic" => ColorKeyLegend {
                title: "Electrostatic Potential (kT/e)".to_string(),
                min_label: "-5.0".to_string(),
                max_label: "+5.0".to_string(),
                stops: vec![[0.92, 0.18, 0.18], [0.96, 0.96, 0.96], [0.16, 0.44, 0.95]],
            },
            "mlp" | "lipophilicity" => ColorKeyLegend {
                title: "Lipophilicity MLP".to_string(),
                min_label: "Hydrophilic".to_string(),
                max_label: "Lipophilic".to_string(),
                stops: vec![[0.04, 0.56, 0.58], [0.96, 0.96, 0.96], [0.72, 0.53, 0.06]],
            },
            "sap" => ColorKeyLegend {
                title: "3D SAP Hydrophobicity".to_string(),
                min_label: "-0.8".to_string(),
                max_label: "+0.8".to_string(),
                stops: vec![[0.42, 0.65, 0.95], [0.94, 0.94, 0.94], [0.94, 0.22, 0.15]],
            },
            "agg" | "aggregation" | "aggrescan" => ColorKeyLegend {
                title: "3D Aggregation Propensity".to_string(),
                min_label: "-2.0 (Soluble)".to_string(),
                max_label: "+2.0 (Agg-Prone)".to_string(),
                stops: vec![[0.24, 0.54, 0.96], [0.95, 0.95, 0.95], [0.94, 0.20, 0.16]],
            },
            "bfactor" | "b" => {
                let (lo, hi) = self.active_molecule().map_or((10.0, 80.0), |mol| {
                    let info = inspect_structure_info(mol);
                    (info.bfactor_min, info.bfactor_max.max(info.bfactor_min + 10.0))
                });
                ColorKeyLegend {
                    title: "X-Ray B-Factor (A^2)".to_string(),
                    min_label: format!("{lo:.0} (Ordered)"),
                    max_label: format!("{hi:.0} (Mobile)"),
                    stops: vec![[0.14, 0.38, 0.94], [0.96, 0.96, 0.96], [0.92, 0.18, 0.18]],
                }
            }
            "rainbow" => ColorKeyLegend {
                title: "Sequence (N -> C)".to_string(),
                min_label: "N-term".to_string(),
                max_label: "C-term".to_string(),
                stops: vec![
                    [0.14, 0.34, 0.92],
                    [0.12, 0.74, 0.88],
                    [0.24, 0.82, 0.44],
                    [0.96, 0.82, 0.18],
                    [0.88, 0.18, 0.22],
                ],
            },
            _ => ColorKeyLegend {
                title: "AlphaFold pLDDT".to_string(),
                min_label: "50 (Low)".to_string(),
                max_label: "100 (High)".to_string(),
                stops: vec![
                    [0.98, 0.48, 0.25],
                    [0.99, 0.84, 0.22],
                    [0.39, 0.79, 0.96],
                    [0.05, 0.35, 0.86],
                ],
            },
        }
    }

    /// Select and focus a single residue `(chain, res_seq)` in 3D (used by Ramachandran & Aggregation plots).
    pub fn focus_residue_3d(&mut self, chain: char, res_seq: i32) -> String {
        let Some(mol) = self.active_molecule() else {
            return "No biomolecule loaded.".to_string();
        };
        let mut mask = vec![false; mol.atoms.len()];
        let mut sum = [0.0_f64; 3];
        let mut cnt = 0usize;
        let mut rname = String::new();
        for (i, a) in mol.atoms.iter().enumerate() {
            if a.chain == chain && a.res_seq == res_seq {
                mask[i] = true;
                sum[0] += a.coords[0];
                sum[1] += a.coords[1];
                sum[2] += a.coords[2];
                cnt += 1;
                if rname.is_empty() {
                    rname = a.res_name.clone();
                }
            }
        }
        if cnt == 0 {
            return format!("Residue {chain}:{res_seq} not found.");
        }
        let center = [sum[0] / cnt as f64, sum[1] / cnt as f64, sum[2] / cnt as f64];
        self.render_opts.selected_atoms = mask.clone();
        self.named_selections.insert("sele".to_string(), mask.clone());
        self.named_selections.insert("sel".to_string(), mask);
        self.camera.focus_on_bounds(center, 11.0);
        self.sync_active_render_opts();
        self.dirty_geometry = true;
        format!("Selected & focused {chain}:{rname}{res_seq} ({cnt} atoms).")
    }

    fn resolve_atom_spec(&self, spec: &str) -> Result<(usize, String, [f64; 3]), ScriptError> {
        let mol = self
            .active_molecule()
            .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
        let parts: Vec<&str> = spec.trim().split(':').collect();
        if parts.len() < 2 {
            return Err(ScriptError::Command(format!(
                "expected CHAIN:RES[:ATOM] (e.g. A:10:CA), got '{spec}'"
            )));
        }
        let ch = parts[0]
            .chars()
            .next()
            .ok_or_else(|| ScriptError::Command("missing chain id".to_string()))?;
        let rseq: i32 = parts[1]
            .parse()
            .map_err(|_| ScriptError::Command(format!("bad residue number in '{spec}'")))?;
        let atom_name = parts.get(2).copied();
        let idx = if let Some(an) = atom_name {
            mol.atoms
                .iter()
                .position(|a| a.chain == ch && a.res_seq == rseq && a.name.eq_ignore_ascii_case(an))
        } else {
            mol.chain(ch)
                .and_then(|c| c.residues.iter().find(|r| r.res_seq == rseq))
                .and_then(|r| r.guide_idx.or_else(|| r.atom_indices.first().copied()))
        }
        .ok_or_else(|| ScriptError::Command(format!("atom '{spec}' not found in active model")))?;
        let a = &mol.atoms[idx];
        Ok((
            idx,
            format!("{}:{}{}.{}", a.chain, a.res_name, a.res_seq, a.name),
            a.coords,
        ))
    }

    /// Record an interactive 2-atom Euclidean distance measurement with dashed line & 3D label.
    pub fn add_distance_measurement(&mut self, a: usize, b: usize) -> Result<String, ScriptError> {
        let (pa, pb, d, msg) = {
            let mol = self
                .active_molecule()
                .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
            let (Some(aa), Some(ab)) = (mol.atoms.get(a), mol.atoms.get(b)) else {
                return Err(ScriptError::Command("atom index out of range".to_string()));
            };
            let d = measure_distance(aa.coords, ab.coords);
            let la = format!("{}:{}{}.{}", aa.chain, aa.res_name, aa.res_seq, aa.name);
            let lb = format!("{}:{}{}.{}", ab.chain, ab.res_name, ab.res_seq, ab.name);
            (
                aa.coords,
                ab.coords,
                d,
                format!("Dist({la} — {lb}) = {d:.2} Å"),
            )
        };
        let mid = [
            0.5 * (pa[0] + pb[0]),
            0.5 * (pa[1] + pb[1]),
            0.5 * (pa[2] + pb[2]),
        ];
        self.render_opts.line_overlays.push(LineOverlay {
            label: msg.clone(),
            from: pa,
            to: pb,
            dashed: true,
            rgba: [0.99, 0.90, 0.22, 1.0],
        });
        self.labels_3d.push(Label3D {
            text: format!("{d:.2} A"),
            coords: mid,
            rgba: [0.99, 0.94, 0.35, 1.0],
            font_scale: 1.0,
        });
        self.measurements_log.push(msg.clone());
        self.sync_active_render_opts();
        self.dirty_geometry = true;
        Ok(msg)
    }

    /// Record an interactive 3-atom geometric angle measurement (`a - b - c`) with lines & 3D label.
    pub fn add_angle_measurement(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
    ) -> Result<String, ScriptError> {
        let (pa, pb, pc, ang, msg) = {
            let mol = self
                .active_molecule()
                .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
            let (Some(aa), Some(ab), Some(ac)) =
                (mol.atoms.get(a), mol.atoms.get(b), mol.atoms.get(c))
            else {
                return Err(ScriptError::Command("atom index out of range".to_string()));
            };
            let ang = measure_angle_deg(aa.coords, ab.coords, ac.coords);
            let msg = format!(
                "Angle({}:{}.{} - {}:{}.{} - {}:{}.{}) = {ang:.1}°",
                aa.chain,
                aa.res_seq,
                aa.name,
                ab.chain,
                ab.res_seq,
                ab.name,
                ac.chain,
                ac.res_seq,
                ac.name
            );
            (aa.coords, ab.coords, ac.coords, ang, msg)
        };
        self.render_opts.line_overlays.push(LineOverlay {
            label: msg.clone(),
            from: pa,
            to: pb,
            dashed: true,
            rgba: [0.35, 0.92, 0.98, 1.0],
        });
        self.render_opts.line_overlays.push(LineOverlay {
            label: msg.clone(),
            from: pb,
            to: pc,
            dashed: true,
            rgba: [0.35, 0.92, 0.98, 1.0],
        });
        self.labels_3d.push(Label3D {
            text: format!("{ang:.1}°"),
            coords: pb,
            rgba: [0.45, 0.96, 1.0, 1.0],
            font_scale: 1.0,
        });
        self.measurements_log.push(msg.clone());
        self.sync_active_render_opts();
        self.dirty_geometry = true;
        Ok(msg)
    }

    /// Record an interactive 4-atom signed dihedral/torsion angle measurement (`a-b-c-d`).
    pub fn add_dihedral_measurement(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        d: usize,
    ) -> Result<String, ScriptError> {
        let (pa, pb, pc, pd, tor, msg) = {
            let mol = self
                .active_molecule()
                .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
            let (Some(aa), Some(ab), Some(ac), Some(ad)) = (
                mol.atoms.get(a),
                mol.atoms.get(b),
                mol.atoms.get(c),
                mol.atoms.get(d),
            ) else {
                return Err(ScriptError::Command("atom index out of range".to_string()));
            };
            let tor = measure_dihedral_deg(aa.coords, ab.coords, ac.coords, ad.coords);
            let msg = format!(
                "Dihedral({}:{}.{} - {}:{}.{} - {}:{}.{} - {}:{}.{}) = {tor:.1}°",
                aa.chain,
                aa.res_seq,
                aa.name,
                ab.chain,
                ab.res_seq,
                ab.name,
                ac.chain,
                ac.res_seq,
                ac.name,
                ad.chain,
                ad.res_seq,
                ad.name
            );
            (aa.coords, ab.coords, ac.coords, ad.coords, tor, msg)
        };
        let mid = [
            0.5 * (pb[0] + pc[0]),
            0.5 * (pb[1] + pc[1]),
            0.5 * (pb[2] + pc[2]),
        ];
        for &(p, q) in &[(pa, pb), (pb, pc), (pc, pd)] {
            self.render_opts.line_overlays.push(LineOverlay {
                label: msg.clone(),
                from: p,
                to: q,
                dashed: true,
                rgba: [0.88, 0.52, 0.99, 1.0],
            });
        }
        self.labels_3d.push(Label3D {
            text: format!("phi {tor:.1}°"),
            coords: mid,
            rgba: [0.94, 0.68, 1.0, 1.0],
            font_scale: 1.0,
        });
        self.measurements_log.push(msg.clone());
        self.sync_active_render_opts();
        self.dirty_geometry = true;
        Ok(msg)
    }

    /// Highlight and select a PAE residue range `[start_idx..=end_idx]` in the active 3D model.
    pub fn select_pae_domain(&mut self, start_idx: usize, end_idx: usize) -> String {
        let (lo, hi, domain_res, mask) = {
            let Some(mol) = self.active_molecule() else {
                return "No active model.".to_string();
            };
            let mut polymer_res: Vec<(char, i32)> = Vec::new();
            for ch in &mol.chains {
                for r in &ch.residues {
                    if matches!(
                        r.class,
                        ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
                    ) && r.guide_idx.is_some()
                    {
                        polymer_res.push((r.chain, r.res_seq));
                    }
                }
            }
            let lo = start_idx.min(end_idx);
            let hi = start_idx
                .max(end_idx)
                .min(polymer_res.len().saturating_sub(1));
            let slice: Vec<(char, i32)> =
                polymer_res.iter().take(hi + 1).skip(lo).copied().collect();
            let domain_set: std::collections::HashSet<(char, i32)> =
                slice.iter().copied().collect();
            let mask: Vec<bool> = mol
                .atoms
                .iter()
                .map(|a| domain_set.contains(&(a.chain, a.res_seq)))
                .collect();
            (lo, hi, slice, mask)
        };
        for (ch, rs) in domain_res {
            self.render_opts
                .residue_highlights
                .insert((ch, rs), [0.16, 0.88, 0.56]);
        }
        let cnt = mask.iter().filter(|&&b| b).count();
        self.render_opts.selected_atoms = mask;
        self.sync_active_render_opts();
        self.dirty_geometry = true;
        format!(
            "Highlighted PAE domain residues {}..={} ({cnt} atoms).",
            lo + 1,
            hi + 1
        )
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
            shadow_strength: self.render_opts.shadow_strength,
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
        self.render_opts.shadow_strength = snap.shadow_strength;
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

    /// Push a transactional Undo snapshot of the active biomolecule before a 3D edit or drag stroke.
    pub fn push_undo_snapshot(&mut self) {
        if let Some(m) = self.active_model_mut() {
            m.undo_stack.push(m.molecule.clone());
            if m.undo_stack.len() > 25 {
                m.undo_stack.remove(0);
            }
            m.redo_stack.clear();
        }
    }

    /// Decode a binary GROMACS `.xtc` trajectory buffer into the active biomolecule.
    pub fn load_xtc_bytes(
        &mut self,
        bytes: &[u8],
        source_label: &str,
    ) -> Result<String, ScriptError> {
        let mol = self
            .active_molecule_mut()
            .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
        let loaded =
            load_xtc_frames(mol, bytes).map_err(|e| ScriptError::Command(e.to_string()))?;
        let t_end = mol.frames.last().map_or(0.0, |f| f.time_ps);
        self.dirty_geometry = true;
        let msg = format!(
            "Loaded {loaded} GROMACS .xtc frames (0.0..{t_end:.1} ps) from {source_label}."
        );
        self.provenance.push(ProvenanceEntry {
            kind: ProvenanceKind::MolecularDynamics,
            source_app: "biomolcraft::xtc-codec".to_string(),
            summary: msg.clone(),
        });
        Ok(msg)
    }

    /// Detect & rank 3D druggable binding cavities on the active macromolecule (`fpocket`/CASTp-style).
    pub fn detect_pockets(&mut self, max_pockets: usize) -> Result<String, ScriptError> {
        let mol = self
            .active_molecule()
            .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
        let title = mol.title.clone();
        let pockets = detect_binding_pockets(mol, max_pockets);
        if pockets.is_empty() {
            self.binding_pockets.clear();
            self.selected_pocket_id = None;
            self.dirty_geometry = true;
            return Ok(format!(
                "No buried binding pockets (> 45 Å³) detected on '{title}'."
            ));
        }

        let mut md = format!(
            "## 3D Druggable Binding Pockets — {title}\n\n| Rank | Druggability | Volume (Å³) | Enclosure | Hydrophobic % | Polar % | Center (Å) | Lining Residues |\n|---:|---:|---:|---:|---:|---:|:---|:---|\n"
        );
        for p in &pockets {
            let preview: Vec<&str> = p
                .lining_residues
                .iter()
                .take(8)
                .map(String::as_str)
                .collect();
            md.push_str(&format!(
                "| #{} | **{:.2}** | {:.0} | {:.0}% | {:.0}% | {:.0}% | `[{:.1}, {:.1}, {:.1}]` | {} |\n",
                p.pocket_id,
                p.druggability_score,
                p.volume_a3,
                p.enclosure * 100.0,
                p.hydrophobic_fraction * 100.0,
                p.polar_fraction * 100.0,
                p.center[0],
                p.center[1],
                p.center[2],
                preview.join(", ")
            ));
        }
        let top = &pockets[0];
        let gb = GridBox3D {
            center: top.center,
            size: top.size,
            spacing: 0.375,
        };
        let msg = format!(
            "Detected {} binding pocket(s) on '{title}'. Top #1: druggability={:.2}, vol={:.0} Å³, enc={:.0}%, center=[{:.1},{:.1},{:.1}] (GridBox synced).",
            pockets.len(),
            top.druggability_score,
            top.volume_a3,
            top.enclosure * 100.0,
            top.center[0],
            top.center[1],
            top.center[2]
        );
        self.binding_pockets = pockets;
        self.selected_pocket_id = Some(1);
        self.render_opts.grid_box = Some(gb);
        if let Some(ref mut vd) = self.viewdock {
            vd.editable_grid_box = Some(gb);
            vd.show_grid_box = true;
        }
        self.sync_active_render_opts();
        self.last_module_markdown = md;
        self.provenance.push(ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::pockets-3d".to_string(),
            summary: msg.clone(),
        });
        self.dirty_geometry = true;
        Ok(msg)
    }

    /// Highlight detected pocket `#rank` (1-based), snap active `DockCraft` `GridBox3D`, and focus camera.
    pub fn select_pocket_gridbox(&mut self, rank: usize) -> Result<String, ScriptError> {
        let Some(p) = self
            .binding_pockets
            .iter()
            .find(|p| p.pocket_id == rank)
            .cloned()
        else {
            return Err(ScriptError::Command(format!(
                "binding pocket #{rank} not found (run `pockets` first; {} detected)",
                self.binding_pockets.len()
            )));
        };
        let gb = GridBox3D {
            center: p.center,
            size: p.size,
            spacing: 0.375,
        };
        self.selected_pocket_id = Some(p.pocket_id);
        self.render_opts.grid_box = Some(gb);
        if let Some(ref mut vd) = self.viewdock {
            vd.editable_grid_box = Some(gb);
            vd.show_grid_box = true;
        }
        let rad = (p.size[0].max(p.size[1]).max(p.size[2]) * 0.72).max(10.0);
        self.camera.focus_on_bounds(p.center, rad);
        self.sync_active_render_opts();
        self.dirty_geometry = true;
        Ok(format!(
            "Pocket #{} selected (drug={:.2}, vol={:.0} Å³): GridBox synced -> center=[{:.2},{:.2},{:.2}] size=[{:.1},{:.1},{:.1}] Å.",
            p.pocket_id,
            p.druggability_score,
            p.volume_a3,
            gb.center[0],
            gb.center[1],
            gb.center[2],
            gb.size[0],
            gb.size[1],
            gb.size[2]
        ))
    }

    fn parse_model_id_arg(s: &str) -> Option<u32> {
        s.trim().trim_start_matches('#').parse::<u32>().ok()
    }

    fn run_sasa_query(&mut self, expr: &str) -> Result<String, ScriptError> {
        let mol = self
            .active_molecule()
            .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
            .clone();
        let trimmed = expr.trim();
        let (label, target_mask) = if !trimmed.is_empty() {
            let m = evaluate_selection(&mol, trimmed, &self.named_selections)
                .map_err(|e| ScriptError::Command(e.to_string()))?;
            (trimmed.to_string(), m)
        } else if self.render_opts.selected_atoms.iter().any(|&b| b) {
            ("selection".to_string(), self.render_opts.selected_atoms.clone())
        } else if mol.atoms.iter().any(|a| a.res_class == ResidueClass::Ligand) {
            let m: Vec<bool> = mol
                .atoms
                .iter()
                .map(|a| a.res_class == ResidueClass::Ligand)
                .collect();
            ("ligand".to_string(), m)
        } else {
            let m: Vec<bool> = mol
                .atoms
                .iter()
                .map(|a| a.res_class != ResidueClass::Water)
                .collect();
            ("all (non-water)".to_string(), m)
        };

        let rep = compute_selection_sasa(&mol, &target_mask, None);
        if rep.atom_count == 0 {
            return Err(ScriptError::Command(format!(
                "sasa: selection '{label}' matched 0 atoms"
            )));
        }
        let msg = format!(
            "SASA ('{label}', {} atoms): isolated = {:.1} Å², in complex = {:.1} Å², buried ΔSASA = {:.1} Å² ({:.1}% buried).",
            rep.atom_count,
            rep.isolated_sasa_a2,
            rep.complex_sasa_a2,
            rep.buried_sasa_a2,
            rep.buried_fraction_pct
        );
        self.last_module_markdown = format!(
            "## Solvent-Accessible Surface Area (`sasa`)\n\n- **Target:** `{label}` (`{}` atoms)\n- **Isolated SASA (alone):** `{:.2} Å²`\n- **Complex SASA (bound):** `{:.2} Å²`\n- **Buried Surface (`ΔSASA`):** `{:.2} Å²` (**`{:.1}%` buried**)\n",
            rep.atom_count,
            rep.isolated_sasa_a2,
            rep.complex_sasa_a2,
            rep.buried_sasa_a2,
            rep.buried_fraction_pct
        );
        self.measurements_log.push(msg.clone());
        Ok(msg)
    }

    fn parse_interaction_query(rest: &str, default_cutoff: f64) -> InteractionQuery {
        let mut cutoff = default_cutoff;
        let mut inter_model = false;
        let mut intra_model = true;
        let mut inter_explicit = false;
        let mut clean_tokens: Vec<String> = Vec::new();
        let raw: Vec<&str> = rest.split_whitespace().collect();
        let mut i = 0usize;
        while i < raw.len() {
            let tok = raw[i];
            let low = tok.to_ascii_lowercase();
            if low == "intermodel" {
                if let Some(v) = raw.get(i + 1) {
                    inter_model = matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "on" | "yes");
                    inter_explicit = true;
                    i += 2;
                    continue;
                }
            } else if low == "intramodel" {
                if let Some(v) = raw.get(i + 1) {
                    intra_model = matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "on" | "yes");
                    i += 2;
                    continue;
                }
            } else if low == "reveal" || low == "log" {
                if raw.get(i + 1).is_some() {
                    i += 2;
                    continue;
                }
            } else if clean_tokens.is_empty()
                && let Ok(val) = tok.parse::<f64>()
            {
                cutoff = val;
                i += 1;
                continue;
            }
            clean_tokens.push(tok.to_string());
            i += 1;
        }

        // Split on `restrict` keyword if present
        let split_pos = clean_tokens
            .iter()
            .position(|t| t.eq_ignore_ascii_case("restrict"));
        let (left, right) = if let Some(p) = split_pos {
            (&clean_tokens[..p], Some(&clean_tokens[p + 1..]))
        } else {
            (clean_tokens.as_slice(), None)
        };

        let parse_side = |toks: &[String]| -> (Option<u32>, Option<String>) {
            if toks.is_empty() {
                return (None, None);
            }
            if toks[0].starts_with('#')
                && let Some(mid) = Self::parse_model_id_arg(&toks[0])
            {
                let expr = toks[1..].join(" ");
                return (Some(mid), (!expr.trim().is_empty()).then_some(expr));
            }
            let expr = toks.join(" ");
            (None, (!expr.trim().is_empty()).then_some(expr))
        };

        let (model_a, expr_a) = parse_side(left);
        let (model_b, expr_b) = right.map(parse_side).unwrap_or((None, None));
        if !inter_explicit
            && let (Some(ma), Some(mb)) = (model_a, model_b)
            && ma != mb
        {
            inter_model = true;
            intra_model = false;
        }

        InteractionQuery {
            cutoff,
            model_a,
            expr_a,
            model_b,
            expr_b,
            inter_model,
            intra_model,
        }
    }

    #[allow(clippy::type_complexity)]
    fn resolve_interaction_targets(
        &self,
        q: &InteractionQuery,
    ) -> Result<(Biomolecule, Option<Vec<bool>>, Option<Biomolecule>, Option<Vec<bool>>), ScriptError> {
        let mol_a = if let Some(mid) = q.model_a {
            self.models
                .iter()
                .find(|m| m.model_id == mid)
                .map(|m| m.molecule.clone())
                .ok_or_else(|| ScriptError::Command(format!("model #{mid} not found")))?
        } else {
            self.active_molecule()
                .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                .clone()
        };
        let mask_a = match q.expr_a.as_deref() {
            Some(e) => Some(
                evaluate_selection(&mol_a, e, &self.named_selections)
                    .map_err(|err| ScriptError::Command(err.to_string()))?,
            ),
            None => None,
        };
        if let Some(mb_id) = q.model_b {
            let mol_b = self
                .models
                .iter()
                .find(|m| m.model_id == mb_id)
                .map(|m| m.molecule.clone())
                .ok_or_else(|| ScriptError::Command(format!("model #{mb_id} not found")))?;
            let mask_b = match q.expr_b.as_deref() {
                Some(e) => Some(
                    evaluate_selection(&mol_b, e, &self.named_selections)
                        .map_err(|err| ScriptError::Command(err.to_string()))?,
                ),
                None => None,
            };
            Ok((mol_a, mask_a, Some(mol_b), mask_b))
        } else {
            let mask_b = match q.expr_b.as_deref() {
                Some(e) => Some(
                    evaluate_selection(&mol_a, e, &self.named_selections)
                        .map_err(|err| ScriptError::Command(err.to_string()))?,
                ),
                None => {
                    // When only one selection is given (e.g. `hbonds ligand`), check it against complement!
                    mask_a
                        .as_ref()
                        .map(|ma| ma.iter().map(|&v| !v).collect::<Vec<bool>>())
                }
            };
            Ok((mol_a, mask_a, None, mask_b))
        }
    }

    /// Execute one or more semicolon-separated PyMOL/ChimeraX-compatible DSL commands.
    pub fn execute_dsl(&mut self, script: &str) -> Result<String, ScriptError> {
        let mut responses = Vec::new();
        for stmt in split_dsl_statements(script) {
            self.command_history.push(stmt.to_string());
            responses.push(self.execute_single_cmd(stmt)?);
        }
        Ok(responses.join("\n"))
    }

    fn execute_single_cmd(&mut self, stmt: &str) -> Result<String, ScriptError> {
        let cmd = dsl_statement_verb(stmt);
        let rest = stmt
            .trim()
            .split_once(char::is_whitespace)
            .map_or("", |(_, tail)| tail)
            .trim();

        match cmd.as_str() {
            "help" => Ok(render_registry_help(rest)),
            "open" | "load" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: open <path.pdb|cif|sdf|biomol.zip> [add|replace]".to_string(),
                    ));
                }
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let path_s = toks[0].trim_matches('"');
                if path_s.to_ascii_lowercase().ends_with(".biomol.zip") {
                    let (restored, sum) = load_session_zip(Path::new(path_s))?;
                    *self = restored;
                    return Ok(format!(
                        "Loaded verified portable session '{}' ({} models, {} atoms, {} scenes, volume={}, viewdock={}).",
                        path_s,
                        sum.model_count,
                        sum.total_atoms,
                        sum.scene_count,
                        sum.has_volume_mrc,
                        sum.has_viewdock
                    ));
                }
                let mode = toks.get(1).copied().unwrap_or("");
                let add_mode = mode.eq_ignore_ascii_case("add")
                    || (mode.is_empty() && !self.models.is_empty());
                if mode.eq_ignore_ascii_case("replace") || !add_mode {
                    self.load_structure_file(Path::new(path_s))
                } else {
                    self.add_structure_file(Path::new(path_s))
                }
            }
            "save_session" | "pse" | "cxs" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: save_session <out.biomol.zip>".to_string(),
                    ));
                }
                let p = Path::new(rest.trim_matches('"'));
                self.sync_active_render_opts();
                let sum = save_session_zip(self, p)?;
                Ok(format!(
                    "Saved verified portable session to {} ({} bytes, {} models, {} atoms, {} scenes, volume={}, viewdock={}).",
                    p.display(),
                    sum.archive_bytes,
                    sum.model_count,
                    sum.total_atoms,
                    sum.scene_count,
                    sum.has_volume_mrc,
                    sum.has_viewdock
                ))
            }
            "load_session" | "open_session" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: load_session <in.biomol.zip>".to_string(),
                    ));
                }
                let p = Path::new(rest.trim_matches('"'));
                let (restored, sum) = load_session_zip(p)?;
                *self = restored;
                Ok(format!(
                    "Loaded verified portable session from {} ({} bytes, {} models, {} atoms, {} scenes, volume={}, viewdock={}).",
                    p.display(),
                    sum.archive_bytes,
                    sum.model_count,
                    sum.total_atoms,
                    sum.scene_count,
                    sum.has_volume_mrc,
                    sum.has_viewdock
                ))
            }
            "fetch" => {
                let asset = fetch_structure_spec(rest).map_err(ScriptError::Command)?;
                let tag = if asset.cached_hit {
                    "cache hit"
                } else {
                    "downloaded"
                };
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
                    let mut extra = String::new();
                    if asset.provider == "AlphaFoldDB" {
                        self.render_opts.color_scheme = ColorScheme::BfactorPlddt;
                        self.render_opts.residue_colors.clear();
                        self.sync_active_render_opts();
                        if let Some(mol) = self.active_molecule() {
                            let cas: Vec<f64> = mol
                                .atoms
                                .iter()
                                .filter(|a| a.name == "CA" && a.res_class == ResidueClass::Protein)
                                .map(|a| a.b_factor)
                                .collect();
                            if !cas.is_empty() {
                                let vhi = cas.iter().filter(|&&b| b >= 90.0).count();
                                let hi = cas.iter().filter(|&&b| (70.0..90.0).contains(&b)).count();
                                let lo = cas.iter().filter(|&&b| (50.0..70.0).contains(&b)).count();
                                let vlo = cas.iter().filter(|&&b| b < 50.0).count();
                                extra.push_str(&format!(
                                    " | pLDDT Cα ({}): >=90: {vhi}, 70-90: {hi}, 50-70: {lo}, <50: {vlo}",
                                    cas.len()
                                ));
                            }
                        }
                        if let Some(ref pae_path) = asset.pae_path
                            && let Ok(raw) = std::fs::read_to_string(pae_path)
                            && let Ok(pae) = parse_pae_json(&asset.canonical_id, &raw)
                        {
                            extra.push_str(&format!(
                                " | PAE auto-loaded ({}x{}, mean {:.1} Å, {} rigid domains)",
                                pae.length,
                                pae.length,
                                pae.mean_pae,
                                pae.domain_blocks.len()
                            ));
                            self.pae = Some(pae);
                        }
                        self.dirty_geometry = true;
                    }
                    Ok(format!(
                        "Fetched {} {} ({tag}) -> {msg}{extra}",
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
                match toks
                    .first()
                    .copied()
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .as_str()
                {
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
                Ok(format!(
                    "SSAO strength set to {:.2}.",
                    self.render_opts.ssao_strength
                ))
            }
            "shadows" | "shadow" => {
                let low = rest.to_ascii_lowercase();
                let val = match low.as_str() {
                    "" | "on" | "true" => 0.65_f32,
                    "off" | "false" | "none" => 0.0_f32,
                    _ => rest.parse::<f32>().map_err(|_| {
                        ScriptError::Command("usage: shadows <on|off|0.0..1.0>".to_string())
                    })?,
                };
                self.render_opts.shadow_strength = val.clamp(0.0, 1.0);
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!(
                    "Directional PCF 3x3 soft shadows set to {:.2}.",
                    self.render_opts.shadow_strength
                ))
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
                    "putty" | "sausage" | "worm" => RepresentationKind::Putty,
                    "nucleic" | "rna" | "ladder" => RepresentationKind::NucleicLadder,
                    "sticks" | "stick" => RepresentationKind::Sticks,
                    "spheres" | "sphere" | "vdw" => RepresentationKind::Spheres,
                    "surface" | "surf" => RepresentationKind::Surface,
                    "lines" | "line" => RepresentationKind::Lines,
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown style '{other}' (use cartoon|putty|nucleic|sticks|spheres|surface|lines)"
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
                    "rainbow" | "spectrum" => ColorScheme::Rainbow,
                    "bfactor" | "b" => ColorScheme::CrystallographicBfactor,
                    "plddt" => ColorScheme::BfactorPlddt,
                    "sap" => ColorScheme::Sap3D,
                    "esp" | "electrostatic" => ColorScheme::ElectrostaticEsp,
                    "mlp" | "lipophilicity" => ColorScheme::LipophilicityMlp,
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown color scheme '{other}' (use chain|ss|cpk|nucleotide|rainbow|bfactor|plddt|sap|esp|mlp)"
                        )));
                    }
                };
                self.render_opts.color_scheme = c;
                self.render_opts.residue_colors.clear();
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Color scheme set to {}.", c.label()))
            }
            "rainbow" => {
                self.render_opts.color_scheme = ColorScheme::Rainbow;
                self.render_opts.residue_colors.clear();
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok("Colored polymer chains by N->C Rainbow (ligands kept CPK).".to_string())
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
            "transparency" | "opacity" => {
                let val_tok = rest.split_whitespace().next().unwrap_or("");
                let raw: f32 = val_tok.parse().map_err(|_| {
                    ScriptError::Command("usage: transparency <0..100> (e.g. transparency 60)".to_string())
                })?;
                let alpha = if cmd == "opacity" {
                    if raw > 1.0 {
                        (raw / 100.0).clamp(0.05, 1.0)
                    } else {
                        raw.clamp(0.05, 1.0)
                    }
                } else {
                    (1.0 - raw.clamp(0.0, 95.0) / 100.0).clamp(0.05, 1.0)
                };
                self.render_opts.surface_alpha = alpha;
                if alpha < 0.99 {
                    self.render_opts.show_surface_shell = true;
                }
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!(
                    "Surface transparency set to {:.0}% (alpha = {:.2}).",
                    (1.0 - alpha) * 100.0,
                    alpha
                ))
            }
            "sticks" | "pocket_sticks" => {
                if rest.is_empty()
                    || matches!(rest.to_ascii_lowercase().as_str(), "clear" | "off" | "none")
                {
                    self.render_opts.residue_highlights.clear();
                    self.sync_active_render_opts();
                    self.dirty_geometry = true;
                    return Ok("Cleared selective pocket/residue sticks.".to_string());
                }
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let mask = evaluate_selection(&mol, rest, &self.named_selections)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let mut res_set = std::collections::BTreeSet::new();
                for (i, &hit) in mask.iter().enumerate() {
                    if hit {
                        let a = &mol.atoms[i];
                        res_set.insert((a.chain, a.res_seq));
                    }
                }
                for &(ch, rs) in &res_set {
                    self.render_opts
                        .residue_highlights
                        .insert((ch, rs), [0.76, 0.78, 0.82]);
                }
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!(
                    "Displayed sticks for {} residue(s) matching '{rest}' over active representation.",
                    res_set.len()
                ))
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
                    return Err(ScriptError::Command(
                        "invalid numeric GridBox coords".to_string(),
                    ));
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
            "pockets" | "cavities" => {
                let low = rest.to_ascii_lowercase();
                if matches!(low.as_str(), "off" | "clear" | "hide" | "none") {
                    self.binding_pockets.clear();
                    self.selected_pocket_id = None;
                    self.dirty_geometry = true;
                    return Ok("Cleared 3D binding pocket meshes.".to_string());
                }
                let max_n = if low.is_empty() || low == "on" {
                    6usize
                } else {
                    low.parse::<usize>().unwrap_or(6).clamp(1, 16)
                };
                self.detect_pockets(max_n)
            }
            "pocket_box" => {
                let rank: usize = rest.trim().trim_start_matches('#').parse().unwrap_or(1);
                self.select_pocket_gridbox(rank)
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
                let mut res_set = std::collections::HashSet::new();
                for (i, &hit) in mask.iter().enumerate() {
                    if hit {
                        res_set.insert((mol.atoms[i].chain, mol.atoms[i].res_seq));
                    }
                }
                self.render_opts.selected_atoms = mask.clone();
                self.named_selections.insert(name.clone(), mask.clone());
                self.named_selections.insert("sel".to_string(), mask);
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!(
                    "Selection '{name}' matched {count} atoms ({} residues).",
                    res_set.len()
                ))
            }
            "info" => {
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let mut parts = rest.splitn(2, char::is_whitespace);
                let sub = parts.next().unwrap_or("summary").trim();
                let tail = parts.next().unwrap_or("").trim();
                let mask = if !tail.is_empty() {
                    Some(
                        evaluate_selection(&mol, tail, &self.named_selections)
                            .map_err(|e| ScriptError::Command(e.to_string()))?,
                    )
                } else if matches!(sub.to_ascii_lowercase().as_str(), "residues" | "atoms")
                    && !self.render_opts.selected_atoms.is_empty()
                    && self.render_opts.selected_atoms.iter().any(|&b| b)
                {
                    Some(self.render_opts.selected_atoms.clone())
                } else {
                    None
                };
                let md = render_info_markdown(&mol, sub, mask.as_deref());
                let rep = inspect_structure_info(&mol);
                self.last_module_markdown = md;
                Ok(format!(
                    "info {sub}: {} chains, {} atoms, {} internal gap(s), {} truncated sidechain(s), B-factor {:.1}/{:.1}/{:.1} Å².",
                    rep.chains.len(),
                    rep.total_atoms,
                    rep.missing_gaps.len(),
                    rep.truncated_sidechains.len(),
                    rep.bfactor_min,
                    rep.bfactor_median,
                    rep.bfactor_max
                ))
            }
            "sasa" => self.run_sasa_query(rest),
            "ramachandran" | "rama" => {
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let low = rest.trim().to_ascii_lowercase();
                let highlight_outliers = low.contains("outlier");
                let chain_filter = rest
                    .split_whitespace()
                    .find(|t| t.len() == 1 && t.chars().all(|c| c.is_ascii_alphabetic()))
                    .and_then(|t| t.chars().next())
                    .map(|c| c.to_ascii_uppercase());
                let rep = analyze_ramachandran(&mol, chain_filter);
                let md = render_ramachandran_markdown(&rep);
                if highlight_outliers {
                    for p in &rep.points {
                        if p.region == biomol_model::RamaRegion::Outlier {
                            self.render_opts
                                .residue_highlights
                                .insert((p.chain, p.res_seq), [0.98, 0.16, 0.72]);
                        }
                    }
                    self.sync_active_render_opts();
                    self.dirty_geometry = true;
                }
                let msg = format!(
                    "Ramachandran ({} residues): {:.1}% favored ({}), {:.1}% allowed ({}), {:.1}% outliers ({}), {} cis-peptides.",
                    rep.points.len(),
                    rep.favored_pct,
                    rep.favored_count,
                    rep.allowed_pct,
                    rep.allowed_count,
                    rep.outlier_pct,
                    rep.outlier_count,
                    rep.cis_peptide_count
                );
                self.ramachandran = Some(rep);
                self.last_module_markdown = md;
                self.measurements_log.push(msg.clone());
                Ok(msg)
            }
            "aggregation" | "agg" | "aggrescan" => {
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks
                    .first()
                    .is_some_and(|t| t.eq_ignore_ascii_case("rescue"))
                {
                    if toks.len() < 3 {
                        return Err(ScriptError::Command(
                            "usage: aggregation rescue <CHAIN:RESSEQ> <AA1|AA3> [dynamic]"
                                .to_string(),
                        ));
                    }
                    let (c_s, r_s) = toks[1].split_once(':').ok_or_else(|| {
                        ScriptError::Command("expected CHAIN:RESSEQ (e.g. A:42)".to_string())
                    })?;
                    let ch = c_s
                        .chars()
                        .next()
                        .ok_or_else(|| ScriptError::Command("missing chain id".to_string()))?;
                    let rseq: i32 = r_s
                        .parse()
                        .map_err(|_| ScriptError::Command("bad residue number".to_string()))?;
                    let aa = parse_aa_code(toks[2]).ok_or_else(|| {
                        ScriptError::Command(format!("invalid amino acid '{}'", toks[2]))
                    })?;
                    let dyn_mode = toks.iter().any(|t| t.eq_ignore_ascii_case("dynamic"));
                    let res = simulate_solubility_rescue(&mol, ch, rseq, aa, dyn_mode)
                        .map_err(|e| ScriptError::Command(e.to_string()))?;
                    let msg = format!(
                        "Solubility rescue {ch}:{}{rseq}{}: avg {:+.3} -> {:+.3} (Δ {:+.3}), max {:+.3} -> {:+.3}, res {:+.3} -> {:+.3} ({} clashes). Note: {}",
                        res.wild_type,
                        res.mutant,
                        res.before_avg_score,
                        res.after_avg_score,
                        res.delta_avg_score,
                        res.before_max_score,
                        res.after_max_score,
                        res.before_res_score,
                        res.after_res_score,
                        res.clashes_introduced,
                        res.caveat
                    );
                    self.last_module_markdown = format!(
                        "## Solubility Rescue Simulation: `{ch}:{}{rseq}{}`\n\n- **Global Avg Score:** `{:+.3}` → `{:+.3}` (`Δ {:+.3}`)\n- **Peak Score:** `{:+.3}` → `{:+.3}`\n- **Mutated Residue Score:** `{:+.3}` → `{:+.3}`\n- **Steric Clashes Introduced:** `{}`\n- **Scientific Trade-Off Caveat:** {}\n",
                        res.wild_type,
                        res.mutant,
                        res.before_avg_score,
                        res.after_avg_score,
                        res.delta_avg_score,
                        res.before_max_score,
                        res.after_max_score,
                        res.before_res_score,
                        res.after_res_score,
                        res.clashes_introduced,
                        res.caveat
                    );
                    self.measurements_log.push(msg.clone());
                    return Ok(msg);
                }

                let dynamic_mode = toks.iter().any(|t| t.eq_ignore_ascii_case("dynamic"));
                let rep = analyze_aggregation_3d(&mol, dynamic_mode);
                let visuals = aggregation_visual_palette(&rep);
                let md = render_aggregation_markdown(&rep);
                self.render_opts.residue_colors = visuals.residue_colors;
                self.render_opts
                    .sphere_overlays
                    .retain(|s| s.category != "AggregationPatch");
                self.render_opts
                    .sphere_overlays
                    .extend(visuals.patch_spheres);
                self.color_key = Some(self.build_color_key("agg"));
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                let msg = format!(
                    "3D Aggregation ({}): avg {:+.3}, max {:+.3}, {} exposed hotspots (>0.50), {} 3D patch(es).",
                    if dynamic_mode { "dynamic" } else { "static" },
                    rep.avg_score,
                    rep.max_score,
                    rep.hotspot_count,
                    rep.patches.len()
                );
                self.aggregation = Some(rep);
                self.last_module_markdown = md;
                self.measurements_log.push(msg.clone());
                self.provenance.push(ProvenanceEntry {
                    kind: ProvenanceKind::ComputationalPrediction,
                    source_app: "biomolcraft::aggregation-3d".to_string(),
                    summary: msg.clone(),
                });
                Ok(msg)
            }
            "clone" | "duplicate" => {
                if self.models.is_empty() {
                    return Err(ScriptError::Command("no biomolecule loaded".to_string()));
                }
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let (src_idx, custom_name) = if let Some(first) = toks.first()
                    && first.starts_with('#')
                {
                    let id = Self::parse_model_id_arg(first)
                        .ok_or_else(|| ScriptError::Command("bad model id".to_string()))?;
                    let idx = self
                        .models
                        .iter()
                        .position(|m| m.model_id == id)
                        .ok_or_else(|| ScriptError::Command(format!("model #{id} not found")))?;
                    let nm = toks.get(1).copied();
                    (idx, nm)
                } else {
                    (self.active_model_idx, toks.first().copied())
                };
                let src = self.models[src_idx].clone();
                let new_id = self.next_model_id;
                self.next_model_id += 1;
                let new_name = custom_name
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{}_copy", src.name));
                let mut cloned = src;
                cloned.model_id = new_id;
                cloned.name = new_name.clone();
                cloned.role = "clone".to_string();
                cloned.undo_stack.clear();
                cloned.redo_stack.clear();
                self.models.push(cloned);
                self.dirty_geometry = true;
                Ok(format!(
                    "Cloned #{} into #{new_id} '{new_name}' (active model remains #{}).",
                    self.models[src_idx].model_id,
                    self.models[self.active_model_idx].model_id
                ))
            }
            "rotamers" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: rotamers <CHAIN:RESSEQ> <AA1|AA3>".to_string(),
                    ));
                }
                let (c_s, r_s) = toks[0].split_once(':').ok_or_else(|| {
                    ScriptError::Command("expected CHAIN:RESSEQ (e.g. A:315)".to_string())
                })?;
                let ch = c_s
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing chain id".to_string()))?;
                let rseq: i32 = r_s.parse().map_err(|_| {
                    ScriptError::Command("invalid residue sequence number".to_string())
                })?;
                let aa = parse_aa_code(toks[1]).ok_or_else(|| {
                    ScriptError::Command(format!("invalid target amino acid '{}'", toks[1]))
                })?;
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let table = enumerate_rotamers_3d(mol, ch, rseq, aa)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let md = render_rotamers_markdown(&table);
                let best = &table.candidates[table.best_rotamer_index];
                let msg = format!(
                    "Evaluated {} Dunbrack rotamers for {ch}:{}{rseq}{} ({}): best #{} (χ1={:.0}°, χ2={:?}, lig_clash={}, prot_clash={}).",
                    table.candidates.len(),
                    table.wild_type,
                    table.mutant,
                    table.backbone_bin.label(),
                    best.rotamer_index + 1,
                    best.chi1_deg,
                    best.chi2_deg,
                    best.clashes_ligand,
                    best.clashes_protein
                );
                self.rotamer_table = Some(table);
                self.last_module_markdown = md;
                Ok(msg)
            }
            "mutate" | "swapaa" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: mutate <CHAIN:RESSEQ> <AA1|AA3> [rotamer_idx|criteria N|auto]"
                            .to_string(),
                    ));
                }
                let (c_s, r_s) = toks[0].split_once(':').ok_or_else(|| {
                    ScriptError::Command("expected CHAIN:RESSEQ (e.g. A:315)".to_string())
                })?;
                let ch = c_s
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing chain id".to_string()))?;
                let rseq: i32 = r_s.parse().map_err(|_| {
                    ScriptError::Command("invalid residue sequence number".to_string())
                })?;
                let aa = parse_aa_code(toks[1]).ok_or_else(|| {
                    ScriptError::Command(format!("invalid mutant amino acid '{}'", toks[1]))
                })?;

                // Enumerate all Dunbrack rotamers prior to mutation so GUI/Log displays full table
                if let Some(mol_ro) = self.active_molecule()
                    && let Ok(table) = enumerate_rotamers_3d(mol_ro, ch, rseq, aa)
                {
                    self.last_module_markdown = render_rotamers_markdown(&table);
                    self.rotamer_table = Some(table);
                }

                // Support both 0-based `rotamer_idx` AND ChimeraX 1-based `criteria N`
                let explicit_rot: Option<usize> = if toks.len() >= 4
                    && toks[2].eq_ignore_ascii_case("criteria")
                {
                    toks[3].parse::<usize>().ok().map(|n| n.saturating_sub(1))
                } else if let Some(&rot_s) = toks.get(2)
                    && let Ok(idx) = rot_s.parse::<usize>()
                {
                    Some(idx)
                } else {
                    None
                };

                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = if let Some(rot_idx) = explicit_rot {
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
                    .retain(|s| s.category != "Clash");
                self.render_opts
                    .sphere_overlays
                    .extend(rep.clash_overlays.clone());
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                let chi2_s = rep
                    .chi2_deg
                    .map_or_else(|| "—".to_string(), |v| format!("{v:.0}°"));
                let msg = format!(
                    "Mutated {ch}:{}{rseq}{} (Dunbrack rot {}/{}, prob {:.1}%, {}, χ1={:.0}°, χ2={chi2_s}): {} clashes (lig={}, prot={}, max_ov={:.2} Å), {} new H-bonds.",
                    rep.wild_type,
                    rep.mutant,
                    rep.rotamer_index + 1,
                    rep.total_rotamers,
                    rep.rotamer_probability_pct,
                    rep.backbone_bin.label(),
                    rep.chi1_deg,
                    rep.clash_count,
                    rep.clashes_ligand,
                    rep.clashes_protein,
                    rep.max_overlap_a,
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
                    "antibodies" | "ab" | "vhh" => {
                        run_antibody_module(&mol, None, None)
                            .map_err(ScriptError::Command)?
                            .visual
                    }
                    "ncrna" | "rna" | "mrna" => {
                        run_ncrna_module(&mol).map_err(ScriptError::Command)?.0
                    }
                    "enzymes" | "enzyme" => run_enzyme_module(&mol),
                    "adc" => {
                        run_adc_module(&mol, "ggfg", "dxd", 8.0, false)
                            .map_err(ScriptError::Command)?
                            .0
                    }
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
                        "usage: save <out.pdb|out.cif|out.craftscene.json|out.biomol.zip>"
                            .to_string(),
                    ));
                }
                let out = Path::new(rest.trim_matches('"'));
                let low = rest.to_ascii_lowercase();
                if low.ends_with(".biomol.zip") || low.ends_with(".zip") {
                    self.sync_active_render_opts();
                    let sum = save_session_zip(self, out)?;
                    Ok(format!(
                        "Saved verified portable session to {} ({} bytes, {} models, {} scenes).",
                        out.display(),
                        sum.archive_bytes,
                        sum.model_count,
                        sum.scene_count
                    ))
                } else if low.ends_with(".json") {
                    let doc = self.to_craftscene_document();
                    let text = serde_json::to_string_pretty(&doc)
                        .map_err(|e| ScriptError::Command(e.to_string()))?;
                    std::fs::write(out, text)?;
                    Ok(format!("Saved CraftScene JSON to {}.", out.display()))
                } else if low.ends_with(".cif") || low.ends_with(".mmcif") {
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
                let idx: usize = rest.parse().map_err(|_| {
                    ScriptError::Command("usage: frame <0-based index>".to_string())
                })?;
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
                        "usage: load_dcd <trajectory.dcd|trajectory.xtc>".to_string(),
                    ));
                }
                let clean = rest.trim_matches('"');
                let bytes = std::fs::read(clean)?;
                if clean.to_ascii_lowercase().ends_with(".xtc") {
                    return self.load_xtc_bytes(&bytes, clean);
                }
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let loaded = load_dcd_frames(mol, &bytes)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                let msg = format!("Loaded {loaded} binary DCD trajectory frames from {clean}.");
                self.provenance.push(ProvenanceEntry {
                    kind: ProvenanceKind::MolecularDynamics,
                    source_app: "biomolcraft::dcd-loader".to_string(),
                    summary: msg.clone(),
                });
                Ok(msg)
            }
            "load_xtc" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: load_xtc <trajectory.xtc>".to_string(),
                    ));
                }
                let clean = rest.trim_matches('"');
                let bytes = std::fs::read(clean)?;
                self.load_xtc_bytes(&bytes, clean)
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
                let ch = toks.first().and_then(|s| s.chars().next()).unwrap_or('C');
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
                let mut extras = Vec::new();
                if let Some(p) = sum.wordcraft_docx {
                    extras.push(format!("DOCX={p}"));
                }
                if let Some(p) = sum.deckcraft_pptx {
                    extras.push(format!("PPTX={p}"));
                }
                if let Some(p) = sum.gridcraft_xlsx {
                    extras.push(format!("XLSX={p}"));
                }
                let native_note = if extras.is_empty() {
                    String::new()
                } else {
                    format!(" [Native ArtCraft Office: {}]", extras.join(", "))
                };
                Ok(format!(
                    "Exported ArtCraft bundle (PNG + GridCraft TSV + WordCraft/DeckCraft MD) to {}.{native_note}",
                    sum.out_dir
                ))
            }
            "push_artcraft" | "push" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let target_s = toks.first().copied().unwrap_or("wordcraft");
                let target = ArtCraftTarget::parse(target_s).ok_or_else(|| {
                    ScriptError::Command(format!(
                        "unknown ArtCraft target '{target_s}' (use wordcraft|deckcraft|gridcraft)"
                    ))
                })?;
                let default_dir = format!("/tmp/biomolcraft_push_{}", target.app_id());
                let out_dir = toks.get(1).copied().unwrap_or(&default_dir);
                let res = self.push_artcraft_target(target, Path::new(out_dir))?;
                Ok(res.message)
            }
            "putty" => {
                let on = !matches!(rest.to_ascii_lowercase().as_str(), "off" | "false" | "0");
                self.render_opts.representation = if on {
                    RepresentationKind::Putty
                } else {
                    RepresentationKind::Cartoon
                };
                if on {
                    self.render_opts.color_scheme = ColorScheme::BfactorPlddt;
                }
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("B-Factor / Attribute Putty mode: {on}."))
            }
            "~hbonds" => {
                self.render_opts
                    .line_overlays
                    .retain(|l| !l.label.starts_with("HBond"));
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok("Cleared H-bond overlays.".to_string())
            }
            "~clashes" => {
                self.render_opts
                    .line_overlays
                    .retain(|l| !l.label.starts_with("Clash"));
                self.render_opts
                    .sphere_overlays
                    .retain(|s| !s.category.eq_ignore_ascii_case("clash"));
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok("Cleared steric clash overlays.".to_string())
            }
            "~contacts" => {
                self.render_opts
                    .line_overlays
                    .retain(|l| !l.label.starts_with("Contact"));
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok("Cleared contact overlays.".to_string())
            }
            "hbonds" => {
                if matches!(rest.to_ascii_lowercase().as_str(), "off" | "clear" | "none") {
                    return self.execute_single_cmd("~hbonds");
                }
                let q = Self::parse_interaction_query(rest, 3.5);
                let (mol_a, mask_a, mol_b_opt, mask_b) = self.resolve_interaction_targets(&q)?;
                let mut hbs = Vec::new();
                if let Some(mol_b) = mol_b_opt {
                    if q.inter_model {
                        hbs.extend(detect_hbonds_between(
                            &mol_a,
                            mask_a.as_deref(),
                            &mol_b,
                            mask_b.as_deref(),
                            q.cutoff,
                            95.0,
                        ));
                    }
                    if q.intra_model && q.model_a == q.model_b {
                        hbs.extend(detect_hbonds(
                            &mol_a,
                            mask_a.as_deref(),
                            mask_b.as_deref(),
                            q.cutoff,
                            95.0,
                        ));
                    }
                } else {
                    if q.intra_model {
                        hbs.extend(detect_hbonds(
                            &mol_a,
                            mask_a.as_deref(),
                            mask_b.as_deref(),
                            q.cutoff,
                            95.0,
                        ));
                    }
                    if q.inter_model {
                        let active_id = self.active_model().map(|m| m.model_id);
                        for other in &self.models {
                            if Some(other.model_id) != active_id && other.visible {
                                let ob = q.expr_b.as_deref().and_then(|e| {
                                    evaluate_selection(&other.molecule, e, &self.named_selections)
                                        .ok()
                                });
                                hbs.extend(detect_hbonds_between(
                                    &mol_a,
                                    mask_a.as_deref(),
                                    &other.molecule,
                                    ob.as_deref(),
                                    q.cutoff,
                                    95.0,
                                ));
                            }
                        }
                    }
                }
                self.render_opts
                    .line_overlays
                    .retain(|l| !l.label.starts_with("HBond"));
                for h in hbs.iter().take(250) {
                    self.render_opts.line_overlays.push(LineOverlay {
                        label: format!(
                            "HBond {} - {} ({:.2} A)",
                            h.label_a, h.label_b, h.distance_a
                        ),
                        from: h.coords_a,
                        to: h.coords_b,
                        dashed: true,
                        rgba: [0.24, 0.85, 0.99, 0.95],
                    });
                }
                let mut md = format!(
                    "## Hydrogen Bonds (`hbonds`, cutoff {:.1} Å, interModel={})\n\nDetected **{}** geometric H-bonds.\n\n| Donor/Acceptor A | Donor/Acceptor B | Dist (Å) | Angle (°) | Est. E (kcal/mol) |\n|---|---|---:|---:|---:|\n",
                    q.cutoff,
                    q.inter_model,
                    hbs.len()
                );
                for h in hbs.iter().take(40) {
                    md.push_str(&format!(
                        "| `{}` | `{}` | {:.2} | {:.1} | {:.2} |\n",
                        h.label_a, h.label_b, h.distance_a, h.angle_deg, h.energy_kcal
                    ));
                }
                self.last_module_markdown = md;
                let msg = format!(
                    "Detected {} hydrogen bonds (<= {:.1} Å, interModel={}).",
                    hbs.len(),
                    q.cutoff,
                    q.inter_model
                );
                self.measurements_log.push(msg.clone());
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(msg)
            }
            "clashes" => {
                if matches!(rest.to_ascii_lowercase().as_str(), "off" | "clear" | "none") {
                    return self.execute_single_cmd("~clashes");
                }
                let q = Self::parse_interaction_query(rest, 0.60);
                let (mol_a, mask_a, mol_b_opt, mask_b) = self.resolve_interaction_targets(&q)?;
                let mut clashes = Vec::new();
                if let Some(mol_b) = mol_b_opt {
                    if q.inter_model {
                        clashes.extend(detect_clashes_between(
                            &mol_a,
                            mask_a.as_deref(),
                            &mol_b,
                            mask_b.as_deref(),
                            q.cutoff,
                        ));
                    }
                    if q.intra_model && q.model_a == q.model_b {
                        clashes.extend(detect_clashes(
                            &mol_a,
                            mask_a.as_deref(),
                            mask_b.as_deref(),
                            q.cutoff,
                        ));
                    }
                } else {
                    if q.intra_model {
                        clashes.extend(detect_clashes(
                            &mol_a,
                            mask_a.as_deref(),
                            mask_b.as_deref(),
                            q.cutoff,
                        ));
                    }
                    if q.inter_model {
                        let active_id = self.active_model().map(|m| m.model_id);
                        for other in &self.models {
                            if Some(other.model_id) != active_id && other.visible {
                                let ob = q.expr_b.as_deref().and_then(|e| {
                                    evaluate_selection(&other.molecule, e, &self.named_selections)
                                        .ok()
                                });
                                clashes.extend(detect_clashes_between(
                                    &mol_a,
                                    mask_a.as_deref(),
                                    &other.molecule,
                                    ob.as_deref(),
                                    q.cutoff,
                                ));
                            }
                        }
                    }
                }
                self.render_opts
                    .line_overlays
                    .retain(|l| !l.label.starts_with("Clash"));
                self.render_opts
                    .sphere_overlays
                    .retain(|s| !s.category.eq_ignore_ascii_case("clash"));
                for c in clashes.iter().take(120) {
                    let mid = [
                        0.5 * (c.coords_a[0] + c.coords_b[0]),
                        0.5 * (c.coords_a[1] + c.coords_b[1]),
                        0.5 * (c.coords_a[2] + c.coords_b[2]),
                    ];
                    self.render_opts.sphere_overlays.push(SphereOverlay {
                        label: format!(
                            "Clash {} - {} ({:.2} A ov)",
                            c.label_a, c.label_b, c.overlap_a
                        ),
                        category: "clash".to_string(),
                        coords: mid,
                        radius_a: (0.45 + 0.35 * c.overlap_a).clamp(0.45, 1.3),
                        rgba: [0.96, 0.18, 0.24, 0.72],
                    });
                    self.render_opts.line_overlays.push(LineOverlay {
                        label: format!("Clash {:.2} A", c.overlap_a),
                        from: c.coords_a,
                        to: c.coords_b,
                        dashed: true,
                        rgba: [0.98, 0.22, 0.28, 1.0],
                    });
                }
                let mut md = format!(
                    "## Steric Clashes (`clashes`, VDW overlap >= {:.2} Å, interModel={})\n\nDetected **{}** steric clashes.\n\n| Atom A | Atom B | Dist (Å) | VDW Overlap (Å) |\n|---|---|---:|---:|\n",
                    q.cutoff,
                    q.inter_model,
                    clashes.len()
                );
                for c in clashes.iter().take(40) {
                    md.push_str(&format!(
                        "| `{}` | `{}` | {:.2} | {:.2} |\n",
                        c.label_a, c.label_b, c.distance_a, c.overlap_a
                    ));
                }
                self.last_module_markdown = md;
                let msg = format!(
                    "Detected {} steric clashes (overlap >= {:.2} Å, interModel={}).",
                    clashes.len(),
                    q.cutoff,
                    q.inter_model
                );
                self.measurements_log.push(msg.clone());
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(msg)
            }
            "contacts" => {
                if matches!(rest.to_ascii_lowercase().as_str(), "off" | "clear" | "none") {
                    return self.execute_single_cmd("~contacts");
                }
                let q = Self::parse_interaction_query(rest, 4.0);
                let (mol_a, mask_a, mol_b_opt, mask_b) = self.resolve_interaction_targets(&q)?;
                let mut contacts = Vec::new();
                if let Some(mol_b) = mol_b_opt {
                    if q.inter_model {
                        contacts.extend(detect_contacts_between(
                            &mol_a,
                            mask_a.as_deref(),
                            &mol_b,
                            mask_b.as_deref(),
                            q.cutoff,
                            250,
                        ));
                    }
                    if q.intra_model && q.model_a == q.model_b {
                        contacts.extend(detect_contacts(
                            &mol_a,
                            mask_a.as_deref(),
                            mask_b.as_deref(),
                            q.cutoff,
                            250,
                        ));
                    }
                } else {
                    if q.intra_model {
                        contacts.extend(detect_contacts(
                            &mol_a,
                            mask_a.as_deref(),
                            mask_b.as_deref(),
                            q.cutoff,
                            250,
                        ));
                    }
                    if q.inter_model {
                        let active_id = self.active_model().map(|m| m.model_id);
                        for other in &self.models {
                            if Some(other.model_id) != active_id && other.visible {
                                let ob = q.expr_b.as_deref().and_then(|e| {
                                    evaluate_selection(&other.molecule, e, &self.named_selections)
                                        .ok()
                                });
                                contacts.extend(detect_contacts_between(
                                    &mol_a,
                                    mask_a.as_deref(),
                                    &other.molecule,
                                    ob.as_deref(),
                                    q.cutoff,
                                    250,
                                ));
                            }
                        }
                    }
                }
                self.render_opts
                    .line_overlays
                    .retain(|l| !l.label.starts_with("Contact"));
                for c in contacts.iter().take(150) {
                    self.render_opts.line_overlays.push(LineOverlay {
                        label: format!(
                            "Contact {} - {} ({:.2} A)",
                            c.label_a, c.label_b, c.distance_a
                        ),
                        from: c.coords_a,
                        to: c.coords_b,
                        dashed: true,
                        rgba: [0.98, 0.72, 0.22, 0.85],
                    });
                }
                let mut md = format!(
                    "## Atomic Contacts (`contacts`, cutoff {:.1} Å, interModel={})\n\nDetected **{}** non-covalent contacts.\n\n| Atom A | Atom B | Dist (Å) |\n|---|---|---:|\n",
                    q.cutoff,
                    q.inter_model,
                    contacts.len()
                );
                for c in contacts.iter().take(40) {
                    md.push_str(&format!(
                        "| `{}` | `{}` | {:.2} |\n",
                        c.label_a, c.label_b, c.distance_a
                    ));
                }
                self.last_module_markdown = md;
                let msg = format!(
                    "Detected {} contacts (<= {:.1} Å, interModel={}).",
                    contacts.len(),
                    q.cutoff,
                    q.inter_model
                );
                self.measurements_log.push(msg.clone());
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(msg)
            }
            "distance" | "dist" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: distance <CHAIN:RES[:ATOM]> <CHAIN:RES[:ATOM]>".to_string(),
                    ));
                }
                let (ia, _, _) = self.resolve_atom_spec(toks[0])?;
                let (ib, _, _) = self.resolve_atom_spec(toks[1])?;
                self.add_distance_measurement(ia, ib)
            }
            "angle" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 3 {
                    return Err(ScriptError::Command(
                        "usage: angle <A1> <A2> <A3> (CHAIN:RES[:ATOM])".to_string(),
                    ));
                }
                let (ia, _, _) = self.resolve_atom_spec(toks[0])?;
                let (ib, _, _) = self.resolve_atom_spec(toks[1])?;
                let (ic, _, _) = self.resolve_atom_spec(toks[2])?;
                self.add_angle_measurement(ia, ib, ic)
            }
            "torsion" | "dihedral" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 4 {
                    return Err(ScriptError::Command(
                        "usage: torsion <A1> <A2> <A3> <A4> (CHAIN:RES[:ATOM])".to_string(),
                    ));
                }
                let (ia, _, _) = self.resolve_atom_spec(toks[0])?;
                let (ib, _, _) = self.resolve_atom_spec(toks[1])?;
                let (ic, _, _) = self.resolve_atom_spec(toks[2])?;
                let (id, _, _) = self.resolve_atom_spec(toks[3])?;
                self.add_dihedral_measurement(ia, ib, ic, id)
            }
            "measure" => {
                let low = rest.to_ascii_lowercase();
                if low == "clear" {
                    self.measurements_log.clear();
                    self.axes.clear();
                    self.planes.clear();
                    self.labels_3d.clear();
                    self.render_opts.line_overlays.clear();
                    self.render_opts.sphere_overlays.clear();
                    self.sync_active_render_opts();
                    self.dirty_geometry = true;
                    return Ok("Cleared all measurements, axes, planes, and 3D labels.".to_string());
                }
                if let Some(tail) = low.strip_prefix("sasa") {
                    let orig_tail = rest[4..].trim();
                    let _ = tail;
                    return self.run_sasa_query(orig_tail);
                }
                Ok(if self.measurements_log.is_empty() {
                    "No measurements recorded yet.".to_string()
                } else {
                    self.measurements_log.join("\n")
                })
            }
            "axis" => {
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let mut pts: Vec<[f64; 3]> = self
                    .render_opts
                    .selected_atoms
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| s)
                    .filter_map(|(i, _)| mol.atoms.get(i).map(|a| a.coords))
                    .collect();
                if pts.len() < 3 {
                    pts = mol
                        .chains
                        .first()
                        .map(|c| {
                            c.residues
                                .iter()
                                .filter_map(|r| r.guide_idx.map(|gi| mol.atoms[gi].coords))
                                .collect()
                        })
                        .unwrap_or_default();
                }
                let lbl = if rest.is_empty() {
                    format!("axis_{}", self.axes.len() + 1)
                } else {
                    rest.to_string()
                };
                let ax = compute_geometry_axis(&lbl, &pts)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let mut msg = format!(
                    "Defined axis '{}' (len {:.2} Å, linearity {:.3}).",
                    ax.label, ax.length_a, ax.linearity_ratio
                );
                if let Some(prev) = self.axes.last() {
                    let ang = axis_angle_deg(prev.direction, ax.direction, true);
                    msg.push_str(&format!(
                        " Inter-axis angle vs '{}' = {ang:.1}°.",
                        prev.label
                    ));
                }
                self.labels_3d.push(Label3D {
                    text: ax.label.clone(),
                    coords: ax.centroid,
                    rgba: [0.99, 0.84, 0.28, 1.0],
                    font_scale: 1.0,
                });
                self.axes.push(ax);
                self.measurements_log.push(msg.clone());
                self.dirty_geometry = true;
                Ok(msg)
            }
            "plane" => {
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let mut pts: Vec<[f64; 3]> = self
                    .render_opts
                    .selected_atoms
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| s)
                    .filter_map(|(i, _)| mol.atoms.get(i).map(|a| a.coords))
                    .collect();
                if pts.len() < 3 {
                    pts = mol
                        .atoms
                        .iter()
                        .filter(|a| a.res_class == ResidueClass::Ligand)
                        .map(|a| a.coords)
                        .collect();
                }
                if pts.len() < 3 {
                    pts = mol.atoms.iter().take(24).map(|a| a.coords).collect();
                }
                let lbl = if rest.is_empty() {
                    format!("plane_{}", self.planes.len() + 1)
                } else {
                    rest.to_string()
                };
                let pl = compute_best_fit_plane(&lbl, &pts)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let mut msg = format!(
                    "Defined plane '{}' (r={:.2} Å, rms={:.3} Å).",
                    pl.label, pl.radius_a, pl.rms_deviation_a
                );
                if let Some(prev) = self.planes.last() {
                    let ang = axis_angle_deg(prev.normal, pl.normal, true);
                    msg.push_str(&format!(
                        " Inter-plane angle vs '{}' = {ang:.1}°.",
                        prev.label
                    ));
                }
                self.labels_3d.push(Label3D {
                    text: pl.label.clone(),
                    coords: pl.centroid,
                    rgba: [0.35, 0.92, 0.98, 1.0],
                    font_scale: 1.0,
                });
                self.planes.push(pl);
                self.measurements_log.push(msg.clone());
                self.dirty_geometry = true;
                Ok(msg)
            }
            "label" => {
                if rest.eq_ignore_ascii_case("clear") {
                    self.labels_3d.clear();
                    self.labels_2d.clear();
                    self.dirty_geometry = true;
                    return Ok("Cleared all 3D & 2D labels.".to_string());
                }
                let mut parts = rest.splitn(2, char::is_whitespace);
                let spec = parts.next().unwrap_or("");
                let text = parts.next().unwrap_or("").trim();
                let (_, auto_lbl, coords) = self.resolve_atom_spec(spec)?;
                let lbl_text = if text.is_empty() {
                    auto_lbl
                } else {
                    text.to_string()
                };
                self.labels_3d.push(Label3D {
                    text: lbl_text.clone(),
                    coords,
                    rgba: [0.98, 0.96, 0.85, 1.0],
                    font_scale: 1.0,
                });
                self.dirty_geometry = true;
                Ok(format!("Added 3D label '{lbl_text}' at {spec}."))
            }
            "2dlabel" | "2dlabels" => {
                if rest.eq_ignore_ascii_case("clear") {
                    self.labels_2d.clear();
                    self.dirty_geometry = true;
                    return Ok("Cleared 2D labels.".to_string());
                }
                let toks: Vec<&str> = rest.splitn(3, char::is_whitespace).collect();
                if toks.len() < 3 {
                    return Err(ScriptError::Command(
                        "usage: 2dlabel <x_0..1> <y_0..1> <text>".to_string(),
                    ));
                }
                let x: f32 = toks[0].parse::<f32>().unwrap_or(0.04).clamp(0.02, 0.92);
                let y: f32 = toks[1].parse::<f32>().unwrap_or(0.04).clamp(0.02, 0.92);
                let text = toks[2].trim().to_string();
                let id = format!("lbl2d_{}", self.labels_2d.len() + 1);
                self.labels_2d.push(Label2D {
                    id,
                    text: text.clone(),
                    norm_pos: [x, y],
                    rgba: [0.96, 0.98, 1.0, 1.0],
                    font_scale: 1.0,
                });
                self.dirty_geometry = true;
                Ok(format!("Added 2D label '{text}' at ({x:.2}, {y:.2})."))
            }
            "key" => {
                if matches!(rest.to_ascii_lowercase().as_str(), "off" | "false" | "none") {
                    self.color_key = None;
                    self.dirty_geometry = true;
                    return Ok("Color Key legend hidden.".to_string());
                }
                let key = self.build_color_key(rest);
                let t = key.title.clone();
                self.color_key = Some(key);
                self.dirty_geometry = true;
                Ok(format!("Color Key legend enabled: {t}."))
            }
            "scalebar" => {
                if matches!(rest.to_ascii_lowercase().as_str(), "off" | "false" | "none") {
                    self.scale_bar = None;
                    self.dirty_geometry = true;
                    return Ok("Scale Bar hidden.".to_string());
                }
                let len_a: f32 = rest.parse::<f32>().unwrap_or(20.0).clamp(2.0, 200.0);
                self.scale_bar = Some(ScaleBarSpec {
                    length_a: len_a,
                    label: format!("{len_a:.0} A"),
                });
                self.dirty_geometry = true;
                Ok(format!("Calibrated Scale Bar enabled ({len_a:.0} Å)."))
            }
            "cealign" | "super" => {
                let toks: Vec<&str> = rest
                    .split([',', ' '])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: cealign <#mob:C | mob_ch> <#ref:C | ref_ch>".to_string(),
                    ));
                }
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
                    let tf = cealign_superpose(&mut mob.molecule, mc, &target_mol, tc)
                        .map_err(|e| ScriptError::Command(e.to_string()))?;
                    self.focus_camera_all_visible();
                    self.dirty_geometry = true;
                    return Ok(format!(
                        "CEAlign (seq-independent) #{m_id}:{mc} -> #{t_id}:{tc}: RMSD = {:.4} Å ({} core atoms).",
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
                let tf = cealign_superpose(mol, mc, &target_clone, tc)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                Ok(format!(
                    "CEAlign chain {mc} -> {tc}: RMSD = {:.4} Å ({} core atoms).",
                    tf.rmsd_a, tf.aligned_pairs
                ))
            }
            "msa" => {
                if self.models.is_empty() {
                    return Err(ScriptError::Command("no biomolecule loaded".to_string()));
                }
                let refs: Vec<(&str, &Biomolecule)> = self
                    .models
                    .iter()
                    .map(|m| (m.name.as_str(), &m.molecule))
                    .collect();
                let msa = build_synchronized_msa(&refs);
                let mean_cons = if msa.column_conservation.is_empty() {
                    0.0
                } else {
                    msa.column_conservation.iter().sum::<f32>()
                        / (msa.column_conservation.len() as f32)
                };
                let mut md = format!(
                    "## Synchronized Multi-Model MSA (`Match->Align`)\n\n- **Models:** `{}` | **Columns:** `{}` | **Mean Conservation:** `{:.1}%`\n\n```\n",
                    msa.rows.len(),
                    msa.num_columns,
                    mean_cons * 100.0
                );
                for row in &msa.rows {
                    let seq: String = row.aligned_chars.iter().take(80).collect();
                    md.push_str(&format!(
                        "{:<14} [{}] {}\n",
                        row.model_name, row.chain_id, seq
                    ));
                }
                md.push_str("```\n");
                let msg = format!(
                    "Built synchronized MSA: {} models x {} columns (mean conservation {:.1}%).",
                    msa.rows.len(),
                    msa.num_columns,
                    mean_cons * 100.0
                );
                self.msa = Some(msa);
                self.last_module_markdown = md;
                Ok(msg)
            }
            "pae" => {
                let pae = if rest.is_empty() || rest.eq_ignore_ascii_case("auto") {
                    let mol = self
                        .active_molecule()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    estimate_pae_proxy_from_structure(mol)
                } else {
                    let raw = std::fs::read_to_string(rest.trim_matches('"'))?;
                    parse_pae_json(rest, &raw).map_err(|e| ScriptError::Command(e.to_string()))?
                };
                let msg = format!(
                    "PAE matrix ready ('{}', {}x{}, mean {:.2} Å, max {:.2} Å, {} rigid domains, json={}).",
                    pae.label,
                    pae.length,
                    pae.length,
                    pae.mean_pae,
                    pae.max_pae,
                    pae.domain_blocks.len(),
                    pae.is_experimental_json
                );
                self.last_module_markdown = format!(
                    "## Predicted Aligned Error (PAE) Matrix\n\n- **Source:** `{}`\n- **Dimensions:** `{} x {}`\n- **Mean / Max PAE:** `{:.2} Å / {:.2} Å`\n- **Detected Rigid Domains:** `{:?}`\n",
                    pae.label, pae.length, pae.length, pae.mean_pae, pae.max_pae, pae.domain_blocks
                );
                self.pae = Some(pae);
                Ok(msg)
            }
            "morph" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let (id_a, id_b, n_frames) = if toks.len() >= 2
                    && toks[0].starts_with('#')
                    && toks[1].starts_with('#')
                {
                    let a = Self::parse_model_id_arg(toks[0])
                        .ok_or_else(|| ScriptError::Command("bad first model id".to_string()))?;
                    let b = Self::parse_model_id_arg(toks[1])
                        .ok_or_else(|| ScriptError::Command("bad second model id".to_string()))?;
                    let f: usize = toks.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
                    (a, b, f)
                } else if self.models.len() >= 2 {
                    let f: usize = toks.first().and_then(|s| s.parse().ok()).unwrap_or(20);
                    (self.models[0].model_id, self.models[1].model_id, f)
                } else {
                    return Err(ScriptError::Command(
                        "usage: morph <#1> <#2> [frames] (load at least 2 conformations first)"
                            .to_string(),
                    ));
                };
                let mol_a = self
                    .models
                    .iter()
                    .find(|m| m.model_id == id_a)
                    .map(|m| m.molecule.clone())
                    .ok_or_else(|| ScriptError::Command(format!("model #{id_a} not found")))?;
                let mol_b = self
                    .models
                    .iter()
                    .find(|m| m.model_id == id_b)
                    .map(|m| m.molecule.clone())
                    .ok_or_else(|| ScriptError::Command(format!("model #{id_b} not found")))?;
                let morphed = morph_biomolecules(&mol_a, &mol_b, n_frames)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let flen = morphed.frames.len();
                // Hide original endpoints so the morphed trajectory plays cleanly in 3D
                for m in &mut self.models {
                    if m.model_id == id_a || m.model_id == id_b {
                        m.visible = false;
                    }
                }
                let loaded_msg = self.insert_model(morphed, None, "morph", false);
                Ok(format!(
                    "Morphed #{id_a} -> #{id_b} across {flen} SHAKE-relaxed frames. {loaded_msg}"
                ))
            }
            "movie" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: movie <turn|rock|fly|morph> <out.gif|out.mp4|out.webm> [frames] [W] [H] [fps]"
                            .to_string(),
                    ));
                }
                let mode = toks[0].to_ascii_lowercase();
                let out_path = Path::new(toks[1]);
                let n_frames: usize = toks.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
                let width: u32 = toks.get(3).and_then(|s| s.parse().ok()).unwrap_or(640);
                let height: u32 = toks.get(4).and_then(|s| s.parse().ok()).unwrap_or(480);
                let fps: u16 = toks.get(5).and_then(|s| s.parse().ok()).unwrap_or(12);

                let frames_rgba = match mode.as_str() {
                    "turn" | "turntable" => {
                        let sc = self.build_viewport_scene();
                        render_turntable_frames(&sc, &self.camera, width, height, n_frames)
                            .map_err(|e| ScriptError::Command(e.to_string()))?
                    }
                    "rock" => {
                        let sc = self.build_viewport_scene();
                        render_rock_frames(&sc, &self.camera, width, height, n_frames, 30.0)
                            .map_err(|e| ScriptError::Command(e.to_string()))?
                    }
                    "fly" => {
                        let sc = self.build_viewport_scene();
                        let mut kfs: Vec<Camera3D> =
                            self.scenes.values().map(|s| s.camera.clone()).collect();
                        if kfs.len() < 2 {
                            let c0 = self.camera.clone();
                            let mut c1 = c0.clone();
                            c1.yaw += 1.25;
                            c1.distance = (c1.distance * 0.78).max(8.0);
                            kfs = vec![c0, c1];
                        }
                        let seg_f = (n_frames / (kfs.len() - 1).max(1)).max(4);
                        render_fly_frames(&sc, &kfs, width, height, seg_f)
                            .map_err(|e| ScriptError::Command(e.to_string()))?
                    }
                    "morph" | "trajectory" | "traj" => {
                        let total = self.active_molecule().map_or(0, |m| m.frames.len());
                        if total < 2 {
                            return Err(ScriptError::Command(
                                "active model has < 2 frames (run `morph`, `anm`, `md`, or `load_dcd` first)"
                                    .to_string(),
                            ));
                        }
                        let orig = self.active_molecule().map_or(0, |m| m.active_frame);
                        let mut frames = Vec::with_capacity(total);
                        for fi in 0..total {
                            if let Some(mol) = self.active_molecule_mut() {
                                mol.set_active_frame(fi);
                            }
                            let sc = self.build_viewport_scene();
                            frames.push(
                                render_scene_rgba(&sc, &self.camera, width, height, false)
                                    .map_err(|e| ScriptError::Command(e.to_string()))?,
                            );
                        }
                        if let Some(mol) = self.active_molecule_mut() {
                            mol.set_active_frame(orig);
                        }
                        frames
                    }
                    other => {
                        return Err(ScriptError::Command(format!(
                            "unknown movie mode '{other}' (use turn|rock|fly|morph)"
                        )));
                    }
                };
                let frame_cnt = frames_rgba.len();
                let bytes = write_movie_file(out_path, width, height, fps, &frames_rgba)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                Ok(format!(
                    "Exported '{mode}' movie ({frame_cnt} frames @ {width}x{height}, {fps} fps) -> {} ({bytes} bytes).",
                    out_path.display()
                ))
            }
            "swapna" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: swapna <CHAIN:RESSEQ> <A|U|G|C|T>".to_string(),
                    ));
                }
                let (c_s, r_s) = toks[0].split_once(':').ok_or_else(|| {
                    ScriptError::Command("expected CHAIN:RESSEQ (e.g. X:15)".to_string())
                })?;
                let ch = c_s
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing chain id".to_string()))?;
                let rseq: i32 = r_s
                    .parse()
                    .map_err(|_| ScriptError::Command("bad residue number".to_string()))?;
                let base = toks[1]
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing target base".to_string()))?;
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = swapna_nucleotide_3d(mol, ch, rseq, base)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.render_opts
                    .residue_highlights
                    .insert((ch, rseq), [0.16, 0.92, 0.64]);
                self.render_opts
                    .sphere_overlays
                    .extend(rep.clash_overlays.clone());
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                let msg = format!(
                    "swapna {ch}:{}{rseq}{} ({}): {} clashes, {} H-bonds, {} 3D base pairs total.",
                    rep.wild_type,
                    rep.mutant,
                    if rep.is_dna { "DNA" } else { "RNA" },
                    rep.clash_count,
                    rep.hbond_count,
                    rep.base_pairs_after
                );
                self.provenance.push(ProvenanceEntry {
                    kind: ProvenanceKind::InteractiveEdit,
                    source_app: "biomolcraft::swapna".to_string(),
                    summary: msg.clone(),
                });
                Ok(msg)
            }
            "bond_rot" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 3 {
                    return Err(ScriptError::Command(
                        "usage: bond_rot <C:R:A1> <C:R:A2> <delta_deg>".to_string(),
                    ));
                }
                let (ia, _, _) = self.resolve_atom_spec(toks[0])?;
                let (ib, _, _) = self.resolve_atom_spec(toks[1])?;
                let deg: f64 = toks[2]
                    .parse()
                    .map_err(|_| ScriptError::Command("bad angle degrees".to_string()))?;
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = rotate_bond_torsion(mol, ia, ib, deg)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                let msg = format!(
                    "bond_rot {} by {:.1}° ({} atoms): {} clashes, {} H-bonds | local score {:.2} -> {:.2} kcal/mol.",
                    rep.bond_label,
                    rep.delta_deg,
                    rep.rotated_atoms,
                    rep.clash_count_after,
                    rep.hbond_count_after,
                    rep.local_score_before_kcal,
                    rep.local_score_after_kcal
                );
                self.measurements_log.push(msg.clone());
                Ok(msg)
            }
            "tug" | "sculpt" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 4 {
                    return Err(ScriptError::Command(
                        "usage: tug <C:R:A> <dx> <dy> <dz> [radius_a]".to_string(),
                    ));
                }
                let (ia, _, _) = self.resolve_atom_spec(toks[0])?;
                let dx: f64 = toks[1].parse().unwrap_or(0.0);
                let dy: f64 = toks[2].parse().unwrap_or(0.0);
                let dz: f64 = toks[3].parse().unwrap_or(0.0);
                let rad: f64 = toks.get(4).and_then(|s| s.parse().ok()).unwrap_or(6.0);
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = tug_atom_with_sculpting(mol, ia, [dx, dy, dz], rad, 35)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                let msg = format!(
                    "Tugged {} by {:.2} Å ({} sculpted atoms): MM E {:.1} -> {:.1} kcal/mol | ΔΔG_local {:+.2} kcal/mol.",
                    rep.atom_label,
                    rep.target_shift_a,
                    rep.atoms_sculpted,
                    rep.energy_before_kcal,
                    rep.energy_after_kcal,
                    rep.local_ddg_kcal
                );
                self.measurements_log.push(msg.clone());
                Ok(msg)
            }
            "build" | "fab" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: build <helix|sheet|extended|rna> <SEQUENCE> [chain]".to_string(),
                    ));
                }
                let kind = toks[0].to_ascii_lowercase();
                let seq = toks[1];
                let chain = toks.get(2).and_then(|s| s.chars().next()).unwrap_or('A');
                let mol = if kind == "rna" || kind == "duplex" {
                    build_de_novo_rna_helix_3d(seq, chain)
                } else {
                    build_de_novo_peptide_3d(seq, &kind, chain)
                }
                .map_err(|e| ScriptError::Command(e.to_string()))?;
                let msg = self.insert_model(mol, None, "built", false);
                Ok(format!("Built 3D '{kind}' structure. {msg}"))
            }
            "addh" => {
                let ph: f64 = rest.parse().unwrap_or(7.4);
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let rep = add_polar_hydrogens_ph(mol, ph);
                self.dirty_geometry = true;
                Ok(format!(
                    "Added {} polar hydrogens at pH {:.1} (protonated HIS={}, ASP/GLU={}).",
                    rep.hydrogens_added, rep.ph, rep.protonated_his, rep.protonated_asp_glu
                ))
            }
            "altlocs" => {
                let (n_alt, md) = {
                    let mol = self
                        .active_molecule()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    let mut md = format!(
                        "## Alternate Conformations (`altlocs`)\n\nDetected **{}** atoms with multiple occupancies:\n\n",
                        mol.altlocs.len()
                    );
                    for a in mol.altlocs.iter().take(40) {
                        md.push_str(&format!(
                            "- `{}:{}{}.{}`: {:?}\n",
                            a.chain, a.res_name, a.res_seq, a.atom_name, a.alt_ids
                        ));
                    }
                    (mol.altlocs.len(), md)
                };
                if n_alt == 0 {
                    Ok(
                        "No alternate conformations (altlocs) detected in active structure."
                            .to_string(),
                    )
                } else {
                    self.last_module_markdown = md;
                    Ok(format!(
                        "Detected {n_alt} atoms with alternate conformations (altlocs)."
                    ))
                }
            }
            "renumber" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: renumber <chain> <start_res_seq>".to_string(),
                    ));
                }
                let ch = toks[0]
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing chain".to_string()))?;
                let start: i32 = toks[1]
                    .parse()
                    .map_err(|_| ScriptError::Command("bad start residue number".to_string()))?;
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let cnt = renumber_chain(mol, ch, start)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                Ok(format!(
                    "Renumbered {cnt} residues in chain {ch} starting at {start}."
                ))
            }
            "changechains" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() < 2 {
                    return Err(ScriptError::Command(
                        "usage: changechains <old_chain> <new_chain>".to_string(),
                    ));
                }
                let old_c = toks[0]
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing old chain".to_string()))?;
                let new_c = toks[1]
                    .chars()
                    .next()
                    .ok_or_else(|| ScriptError::Command("missing new chain".to_string()))?;
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let moved = rename_chain(mol, old_c, new_c)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.dirty_geometry = true;
                Ok(format!("Renamed chain {old_c} -> {new_c} ({moved} atoms)."))
            }
            "combine" => {
                let (n_vis, combined) = {
                    let vis: Vec<&Biomolecule> = self
                        .models
                        .iter()
                        .filter(|m| m.visible)
                        .map(|m| &m.molecule)
                        .collect();
                    if vis.len() < 2 {
                        return Err(ScriptError::Command(
                            "combine requires at least 2 visible loaded models".to_string(),
                        ));
                    }
                    let name = if rest.is_empty() { "combined" } else { rest };
                    let comb = combine_biomolecules(&vis, name)
                        .map_err(|e| ScriptError::Command(e.to_string()))?;
                    (vis.len(), comb)
                };
                let msg = self.insert_model(combined, None, "combined", false);
                Ok(format!("Combined {n_vis} models. {msg}"))
            }
            "delete" => {
                if rest.is_empty() {
                    return Err(ScriptError::Command(
                        "usage: delete <selection_expr> (e.g. `delete water` or `delete chain C`)"
                            .to_string(),
                    ));
                }
                let mask = {
                    let mol = self
                        .active_molecule()
                        .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                    evaluate_selection(mol, rest, &self.named_selections)
                        .map_err(|e| ScriptError::Command(e.to_string()))?
                };
                self.push_undo_snapshot();
                let mol = self
                    .active_molecule_mut()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?;
                let del = delete_atoms_by_mask(mol, &mask)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                self.render_opts.selected_atoms.clear();
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                Ok(format!("Deleted {del} atoms matching '{rest}'."))
            }
            "sym" => {
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let bio = generate_biological_assembly(&mol)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let copies = mol.biomt_ops.len();
                let msg = self.insert_model(bio, None, "bioassembly", false);
                Ok(format!(
                    "Generated Biological Assembly ({copies} BIOMT ops). {msg}"
                ))
            }
            "symexp" | "crystalcontacts" => {
                let cut: f64 = rest.parse().unwrap_or(5.0);
                let mol = self
                    .active_molecule()
                    .ok_or_else(|| ScriptError::Command("no biomolecule loaded".to_string()))?
                    .clone();
                let (mates, contacts) = generate_crystal_mates(&mol, cut)
                    .map_err(|e| ScriptError::Command(e.to_string()))?;
                let n_mates = mates.len();
                if cmd == "symexp" {
                    for (_, mate) in mates {
                        let _ = self.insert_model(mate, None, "symmate", false);
                    }
                    self.render_opts.show_unit_cell = true;
                    self.sync_active_render_opts();
                    self.focus_camera_all_visible();
                }
                self.dirty_geometry = true;
                Ok(format!(
                    "{cmd}: {n_mates} crystallographic symmetry mate(s) within {cut:.1} Å ({contacts} sampled crystal contact pairs <= 4.0 Å)."
                ))
            }
            "unitcell" => {
                let on = !matches!(rest.to_ascii_lowercase().as_str(), "off" | "false" | "0");
                self.render_opts.show_unit_cell = on;
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                let info = self
                    .active_molecule()
                    .and_then(|m| m.unit_cell.as_ref())
                    .map_or_else(
                        || "no CRYST1 metadata".to_string(),
                        |u| {
                            format!(
                                "a={:.2} b={:.2} c={:.2} Å ({:.0}/{:.0}/{:.0}°, {})",
                                u.a, u.b, u.c, u.alpha_deg, u.beta_deg, u.gamma_deg, u.space_group
                            )
                        },
                    );
                Ok(format!(
                    "Crystallographic Unit Cell wireframe: {on} ({info})."
                ))
            }
            "snfg" | "glycans" => {
                let on = !matches!(rest.to_ascii_lowercase().as_str(), "off" | "false" | "0");
                self.render_opts.show_snfg_glycans = on;
                self.sync_active_render_opts();
                self.dirty_geometry = true;
                let rep = self
                    .active_molecule()
                    .map(detect_snfg_glycans)
                    .unwrap_or_default();
                let mut md = format!(
                    "## 3D SNFG Glycans (`snfg`)\n\nDetected **{}** monosaccharide ring(s) and **{}** glycosidic / N-/O-glycan link(s):\n\n",
                    rep.nodes.len(),
                    rep.links.len()
                );
                for n in &rep.nodes {
                    md.push_str(&format!(
                        "- `{}:{}{}` -> **{}** (`{:?}`)\n",
                        n.chain, n.res_name, n.res_seq, n.canonical_name, n.shape
                    ));
                }
                for l in &rep.links {
                    md.push_str(&format!("- Link: `{}`\n", l.label));
                }
                self.last_module_markdown = md;
                Ok(format!(
                    "SNFG 3D Glycans: {on} ({} glycans, {} glycosidic/Asn links).",
                    rep.nodes.len(),
                    rep.links.len()
                ))
            }
            "vop" | "volume" => {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                let sub = toks.first().copied().unwrap_or("").to_ascii_lowercase();
                if sub == "orthoplanes" || sub == "ortho" {
                    let on = !matches!(
                        toks.get(1)
                            .copied()
                            .unwrap_or("on")
                            .to_ascii_lowercase()
                            .as_str(),
                        "off" | "false" | "0"
                    );
                    self.show_orthoplanes = on;
                    self.dirty_geometry = true;
                    return Ok(format!("Volumetric Orthoplanes (XY/XZ/YZ): {on}."));
                }
                let grid = self.active_volume.clone().ok_or_else(|| {
                    ScriptError::Command(
                        "no active volume map (run `molmap` or `open_map` first)".to_string(),
                    )
                })?;
                match sub.as_str() {
                    "gaussian" | "smooth" => {
                        let sig: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.8);
                        let g = grid
                            .volume_gaussian(sig)
                            .map_err(|e| ScriptError::Command(e.to_string()))?;
                        let lbl = g.label.clone();
                        self.active_volume = Some(g);
                        self.dirty_geometry = true;
                        Ok(format!("Applied vop gaussian ({sig:.1} Å) -> '{lbl}'."))
                    }
                    "sharpen" | "bfactor" => {
                        let amt: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.2);
                        let s = grid
                            .volume_sharpen(amt)
                            .map_err(|e| ScriptError::Command(e.to_string()))?;
                        let lbl = s.label.clone();
                        self.active_volume = Some(s);
                        self.dirty_geometry = true;
                        Ok(format!("Applied vop sharpen (amount={amt:.2}) -> '{lbl}'."))
                    }
                    "zone" | "carve" => {
                        let rad: f64 = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(5.0);
                        let centers: Vec<[f64; 3]> = self
                            .active_molecule()
                            .map(|mol| {
                                let sel: Vec<[f64; 3]> = self
                                    .render_opts
                                    .selected_atoms
                                    .iter()
                                    .enumerate()
                                    .filter(|&(_, &s)| s)
                                    .filter_map(|(i, _)| mol.atoms.get(i).map(|a| a.coords))
                                    .collect();
                                if !sel.is_empty() {
                                    sel
                                } else {
                                    let lig: Vec<[f64; 3]> = mol
                                        .atoms
                                        .iter()
                                        .filter(|a| a.res_class == ResidueClass::Ligand)
                                        .map(|a| a.coords)
                                        .collect();
                                    if !lig.is_empty() {
                                        lig
                                    } else {
                                        mol.atoms.iter().step_by(4).map(|a| a.coords).collect()
                                    }
                                }
                            })
                            .unwrap_or_default();
                        let zoned = grid
                            .volume_zone(&centers, rad)
                            .map_err(|e| ScriptError::Command(e.to_string()))?;
                        let lbl = zoned.label.clone();
                        self.active_volume = Some(zoned);
                        self.dirty_geometry = true;
                        Ok(format!(
                            "Carved volume zone within {rad:.1} Å of {} anchor atoms -> '{lbl}'.",
                            centers.len()
                        ))
                    }
                    "hide_dust" | "dust" | "hidedust" => {
                        let min_v: usize = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
                        let (cleaned, removed) = grid
                            .volume_hide_dust(self.active_volume_sigma, min_v)
                            .map_err(|e| ScriptError::Command(e.to_string()))?;
                        self.active_volume = Some(cleaned);
                        self.dirty_geometry = true;
                        Ok(format!(
                            "vop hide_dust (min_voxels={min_v}): suppressed {removed} noise island(s)."
                        ))
                    }
                    other => Err(ScriptError::Command(format!(
                        "unknown vop subcommand '{other}' (use gaussian|sharpen|zone|hide_dust|orthoplanes)"
                    ))),
                }
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

    #[test]
    fn wave5_measurements_cealign_msa_pae_morph_overlays_and_gif_movie()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut pdb1 = String::new();
        let mut pdb2 = String::new();
        let aas1 = [
            "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "SER", "THR", "TYR", "ASN", "GLN", "ASP",
        ];
        let aas2 = [
            "GLU", "LYS", "ARG", "HIS", "GLY", "PRO", "CYS", "MET", "ALA", "VAL", "LEU", "ILE",
        ];
        for i in 0..12usize {
            let t = i as f64;
            let x = t * 1.5;
            let y = 2.3 * (t * 1.745).cos();
            let z = 2.3 * (t * 1.745).sin();
            pdb1.push_str(&format!(
                "ATOM  {:5}  CA  {} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 {:5.2}           C\n",
                i * 2 + 1,
                aas1[i],
                i + 1,
                x,
                y,
                z,
                92.0 - t * 3.0
            ));
            pdb1.push_str(&format!(
                "ATOM  {:5}  O   {} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 {:5.2}           O\n",
                i * 2 + 2,
                aas1[i],
                i + 1,
                x,
                y + 1.24,
                z,
                92.0 - t * 3.0
            ));
            let y2 = if i >= 6 { y + (i - 5) as f64 * 0.9 } else { y };
            pdb2.push_str(&format!(
                "ATOM  {:5}  CA  {} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 85.00           C\n",
                i * 2 + 1,
                aas2[i],
                i + 1,
                -y2 + 14.0,
                x - 6.0,
                z + 3.0
            ));
            pdb2.push_str(&format!(
                "ATOM  {:5}  O   {} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 85.00           O\n",
                i * 2 + 2,
                aas2[i],
                i + 1,
                -(y2 + 1.24) + 14.0,
                x - 6.0,
                z + 3.0
            ));
        }
        let mut sess = BiomolSession::default();
        sess.load_structure_text("conf_a", &pdb1, None)?;
        sess.add_structure_text("conf_b", &pdb2, None)?;

        let ce_out = sess.execute_dsl("cealign #2:A #1:A")?;
        assert!(ce_out.contains("CEAlign"));

        let _ = sess.execute_dsl(
            "active #1; putty on; key plddt; scalebar 15; label A:1 N-Term; 2dlabel 0.04 0.04 Wave5; distance A:1:CA A:12:CA; angle A:1:CA A:6:CA A:12:CA; torsion A:1:CA A:4:CA A:8:CA A:12:CA; axis helix_ax; plane helix_pl; hbonds 3.5; clashes 0.6; msa; pae auto",
        )?;
        assert!(sess.color_key.is_some());
        assert!(sess.scale_bar.is_some());
        assert!(sess.msa.is_some());
        assert!(sess.pae.is_some());
        assert!(!sess.axes.is_empty() && !sess.planes.is_empty());

        let m_out = sess.execute_dsl("morph #1 #2 8")?;
        assert!(m_out.contains("Morphed #1 -> #2 across 8"));
        assert_eq!(sess.active_molecule().map(|m| m.frames.len()), Some(8));

        let gif_path =
            std::env::temp_dir().join(format!("biomol_w5_morph_{}.gif", std::process::id()));
        let mov = sess.execute_dsl(&format!("movie morph {} 8 160 120 10", gif_path.display()))?;
        assert!(mov.contains("Exported 'morph' movie"));
        let raw = std::fs::read(&gif_path)?;
        assert_eq!(&raw[..6], b"GIF89a");
        let _ = std::fs::remove_file(&gif_path);
        Ok(())
    }

    #[test]
    fn wave6_swapna_build_sym_snfg_and_vop_dsl() -> Result<(), Box<dyn std::error::Error>> {
        let mut sess = BiomolSession::default();
        let b_rna = sess.execute_dsl("build rna GCGCUA A")?;
        assert!(b_rna.contains("Built 3D 'rna'"));
        let sw = sess.execute_dsl("swapna A:1 A")?;
        assert!(sw.contains("swapna A:G1A"));

        let b_pep = sess.execute_dsl("build helix AKLTYWDS H")?;
        assert!(b_pep.contains("Built 3D 'helix'"));
        let ah = sess.execute_dsl("addh 7.4")?;
        assert!(ah.contains("Added"));

        let _ = sess
            .execute_dsl("molmap 4.0 1.5; vop sharpen 1.2; vop hide_dust 4; vop orthoplanes on")?;
        assert!(sess.show_orthoplanes);
        let sc = sess.build_viewport_scene();
        assert!(sc.transparent_meshes.len() >= 2);
        Ok(())
    }

    #[test]
    fn wave7_registry_autocomplete_portable_zip_session_and_artcraft_push()
    -> Result<(), Box<dyn std::error::Error>> {
        // 1. Unified Typed Command Registry + [TAB] Autocomplete + Palette filter
        assert!(COMMAND_REGISTRY.len() >= 65);
        let ac_cmd = autocomplete_dsl("save_s");
        assert!(
            ac_cmd
                .iter()
                .any(|c| c.insert_text.starts_with("save_session"))
        );
        let ac_sub = autocomplete_dsl("vop sh");
        assert!(ac_sub.iter().any(|c| c.insert_text == "vop sharpen"));
        let ac_chain = autocomplete_dsl("preset pub_white; push_artcraft de");
        assert!(
            ac_chain
                .iter()
                .any(|c| c.insert_text == "preset pub_white; push_artcraft deckcraft")
        );
        let pal = filter_command_palette("glycan");
        assert!(pal.iter().any(|c| c.name == "snfg"));

        // 2. Build rich multi-model + Scene + Labels + Key + ScaleBar + Volume + ViewDockX session
        let mut sess = BiomolSession::default();
        let _ = sess.execute_dsl(
            "build helix AKLTYWDS A; build rna GCGCUA B; active #1; preset pub_white; key plddt; scalebar 20; 2dlabel 0.03 0.03 Wave7; label A:1:CA Start; molmap 4.5 1.5; vop orthoplanes on; scene save f1; scene save f2",
        )?;
        assert_eq!(sess.models.len(), 2);
        assert_eq!(sess.scenes.len(), 2);
        assert!(sess.active_volume.is_some());

        // 3. Save to verified .biomol.zip, inspect, reload into clean BiomolSession, assert parity!
        let zip_path =
            std::env::temp_dir().join(format!("biomol_w7_sess_{}.biomol.zip", std::process::id()));
        let save_msg = sess.execute_dsl(&format!("save_session {}", zip_path.display()))?;
        assert!(save_msg.contains("Saved verified portable session"));

        let inspected = inspect_session_zip(&zip_path)?;
        assert_eq!(inspected.schema_version, "biomol-session/v1");
        assert_eq!(inspected.model_count, 2);
        assert_eq!(inspected.scene_count, 2);
        assert!(inspected.has_volume_mrc);
        assert!(inspected.has_color_key && inspected.has_scale_bar);

        let mut fresh = BiomolSession::default();
        let load_msg = fresh.execute_dsl(&format!("load_session {}", zip_path.display()))?;
        assert!(load_msg.contains("Loaded verified portable session"));
        assert_eq!(fresh.models.len(), 2);
        assert_eq!(fresh.scenes.len(), 2);
        assert!(fresh.active_volume.is_some());
        assert!(fresh.show_orthoplanes);
        assert_eq!(fresh.labels_3d.len(), 1);
        assert_eq!(fresh.labels_2d.len(), 1);
        assert!(fresh.color_key.is_some());
        assert!(fresh.scale_bar.is_some());
        let sc = fresh.build_viewport_scene();
        assert!(!sc.opaque_meshes.is_empty());

        // 4. ArtCraft Push (writes full PNG + TSV + WordCraft/DeckCraft MD + Craft Link check)
        let push_dir = std::env::temp_dir().join(format!("biomol_w7_push_{}", std::process::id()));
        let push_msg =
            fresh.execute_dsl(&format!("push_artcraft wordcraft {}", push_dir.display()))?;
        assert!(push_msg.contains("WordCraft"));
        assert!(push_dir.join("report_wordcraft.md").is_file());
        assert!(push_dir.join("table_gridcraft.tsv").is_file());
        assert!(push_dir.join("figure_biomolcraft.png").is_file());

        let _ = std::fs::remove_file(&zip_path);
        let _ = std::fs::remove_dir_all(&push_dir);
        Ok(())
    }

    #[test]
    fn wave8_shadows_xtc_pockets_and_session_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        use craftbio_struct::{TrajectoryFrame, write_xtc_frames};

        // Synthetic hollow protein shell with a buried cavity around origin
        let mut pdb = String::new();
        let mut serial = 1usize;
        let mut res_seq = 1i32;
        for iz in -6..=6i32 {
            for iy in -6..=6i32 {
                for ix in -6..=6i32 {
                    let x = ix as f64 * 2.2;
                    let y = iy as f64 * 2.2;
                    let z = iz as f64 * 2.2;
                    let r = (x * x + y * y + z * z).sqrt();
                    let is_mouth = z > 4.5 && (x * x + y * y).sqrt() < 2.8;
                    if (5.6..=10.0).contains(&r) && !is_mouth {
                        pdb.push_str(&format!(
                            "ATOM  {:5}  CA  PHE A{:4}    {:8.3}{:8.3}{:8.3}  1.00 20.00           C\n",
                            serial, res_seq, x, y, z
                        ));
                        serial += 1;
                        res_seq += 1;
                    }
                }
            }
        }
        pdb.push_str("END\n");

        let mut sess = BiomolSession::default();
        sess.load_structure_text("pocket_host", &pdb, None)?;

        // 1. Directional soft shadows DSL
        let sh_msg = sess.execute_dsl("shadows 0.78")?;
        assert!(sh_msg.contains("0.78"));
        assert!((sess.render_opts.shadow_strength - 0.78).abs() < 1e-4);

        // 2. 3D Binding pocket detection + pocket_box GridBox sync
        let p_msg = sess.execute_dsl("pockets 4")?;
        assert!(p_msg.contains("Detected"));
        assert!(!sess.binding_pockets.is_empty());
        let pb_msg = sess.execute_dsl("pocket_box 1")?;
        assert!(pb_msg.contains("GridBox synced"));
        assert!(sess.current_selection_box_snapshot(4.0).is_some());

        // 3. Pure-Rust GROMACS .xtc write & load via DSL
        if let Some(mol) = sess.active_molecule_mut() {
            let base: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.coords).collect();
            let shifted: Vec<[f64; 3]> = base.iter().map(|c| [c[0] + 0.5, c[1], c[2]]).collect();
            mol.frames = vec![
                TrajectoryFrame {
                    frame_index: 0,
                    time_ps: 0.0,
                    coords: base,
                },
                TrajectoryFrame {
                    frame_index: 1,
                    time_ps: 12.5,
                    coords: shifted,
                },
            ];
        }
        let xtc_bytes = write_xtc_frames(sess.active_molecule().ok_or("missing mol")?, 1000.0)?;
        let xtc_path =
            std::env::temp_dir().join(format!("biomol_w8_traj_{}.xtc", std::process::id()));
        std::fs::write(&xtc_path, &xtc_bytes)?;
        let xtc_msg = sess.execute_dsl(&format!("load_xtc {}", xtc_path.display()))?;
        assert!(xtc_msg.contains("Loaded 2 GROMACS .xtc frames"));
        let _ = std::fs::remove_file(&xtc_path);

        // 4. Verify shadow_strength & binding_pockets roundtrip inside .biomol.zip!
        let (zip_bytes, _) = encode_session_zip_bytes(&sess)?;
        let (restored, _) = decode_session_zip_bytes(&zip_bytes)?;
        assert!((restored.render_opts.shadow_strength - 0.78).abs() < 1e-4);
        assert_eq!(restored.binding_pockets.len(), sess.binding_pockets.len());
        let sc = restored.build_viewport_scene();
        assert!((sc.shadow_strength - 0.78).abs() < 1e-4);
        assert!(!sc.transparent_meshes.is_empty());
        Ok(())
    }

    #[test]
    fn wave9_prac1_to_prac3_ramachandran_and_aggregation_replay()
    -> Result<(), Box<dyn std::error::Error>> {
        // Synthetic kinase-like pocket + imatinib-like ligand (STI) + missing residue gap + truncated Lys
        let mut pdb = String::new();
        let pep = build_de_novo_peptide_3d("AAFIVLTWKAA", "helix", 'A')?;
        // Shift numbering to 310..320 so Thr is at 316, place a ligand atom 2.85 A from Thr316 OG1
        let mut og1_pos = [0.0_f64; 3];
        let mut serial = 1usize;
        for a in &pep.atoms {
            let rseq = a.res_seq + 309;
            // Deliberately drop Lys318 NZ to simulate truncated crystal sidechain (PRAC1 Part 5B)
            if rseq == 318 && a.name == "NZ" {
                continue;
            }
            if rseq == 316 && a.name == "OG1" {
                og1_pos = a.coords;
            }
            let bf = if rseq == 318 { 74.0 } else { 18.0 };
            pdb.push_str(&format!(
                "ATOM  {:5} {:^4} {:>3} {}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}\n",
                serial,
                a.name,
                a.res_name,
                a.chain,
                rseq,
                a.coords[0],
                a.coords[1],
                a.coords[2],
                bf,
                a.element
            ));
            serial += 1;
        }
        // Add a ligand STI making an H-bond at 2.85 A to Thr316 OG1
        pdb.push_str(&format!(
            "HETATM{:5}  N13 STI A 601    {:8.3}{:8.3}{:8.3}  1.00 24.00           N\n",
            serial,
            og1_pos[0] + 2.85,
            og1_pos[1],
            og1_pos[2]
        ));
        serial += 1;
        pdb.push_str(&format!(
            "HETATM{:5}  C14 STI A 601    {:8.3}{:8.3}{:8.3}  1.00 24.00           C\n",
            serial,
            og1_pos[0] + 4.15,
            og1_pos[1] + 0.4,
            og1_pos[2]
        ));
        pdb.push_str("END\n");

        let mut sess = BiomolSession::default();
        sess.load_structure_text("abl_prac", &pdb, None)?;

        // PRAC1: info, rainbow, bfactor, transparency, select b>=50
        let inf = sess.execute_dsl("info missing")?;
        assert!(inf.contains("1 truncated sidechain"));
        let rb = sess.execute_dsl("rainbow")?;
        assert!(rb.contains("Rainbow"));
        let bf = sess.execute_dsl("color bfactor; key bfactor")?;
        assert!(bf.contains("B-Factor"));
        let tr = sess.execute_dsl("surface on; transparency 60")?;
        assert!(tr.contains("60%"));
        assert!((sess.render_opts.surface_alpha - 0.40).abs() < 0.02);
        let sel_b = sess.execute_dsl("select mobile, b>=50")?;
        assert!(sel_b.contains("matched"));

        // PRAC2: pocket sticks, ligand hbonds, SASA burial %
        let st = sess.execute_dsl("sticks byres (within 4.5 of ligand) and protein")?;
        assert!(st.contains("Displayed sticks"));
        let sasa_msg = sess.execute_dsl("measure sasa ligand")?;
        assert!(sasa_msg.contains("% buried"));
        let hb_wt = sess.execute_dsl("hbonds ligand")?;
        assert!(hb_wt.contains("hydrogen bonds"));
        let _ = sess.execute_dsl("~hbonds")?;

        // PRAC3: clone wildtype #2, hide #2, enumerate Dunbrack rotamers, swapaa T->ILE, interModel check!
        let cl = sess.execute_dsl("clone wildtype; hide #2")?;
        assert!(cl.contains("Cloned #1 into #2 'wildtype'"));
        assert_eq!(sess.models.len(), 2);

        let rot_msg = sess.execute_dsl("rotamers A:316 ile")?;
        assert!(rot_msg.contains("7 Dunbrack rotamers"));
        assert!(sess.rotamer_table.is_some());

        let mut_msg = sess.execute_dsl("swapaa A:316 ile criteria 1")?;
        assert!(mut_msg.contains("Mutated A:T316I"));
        assert!(mut_msg.contains("χ2="));

        // Cross-model clash/contact check (#1 mutant vs #2 wildtype)
        let inter = sess.execute_dsl("contacts #1 resi 316 restrict #2 ligand intermodel true intramodel false")?;
        assert!(inter.contains("interModel=true"));

        // TRAB1: Ramachandran audit
        let rama_msg = sess.execute_dsl("ramachandran")?;
        assert!(rama_msg.contains("Ramachandran"));
        assert!(sess.ramachandran.is_some());

        // TRAB4 Tool 2: General 3D surface aggregation + solubility rescue simulation
        let agg_msg = sess.execute_dsl("aggregation")?;
        assert!(agg_msg.contains("3D Aggregation"));
        assert!(sess.aggregation.is_some());
        assert!(!sess.render_opts.residue_colors.is_empty());

        let rescue_msg = sess.execute_dsl("aggregation rescue A:313 D")?;
        assert!(rescue_msg.contains("Solubility rescue"));

        // Verify .biomol.zip roundtrips residue_colors cleanly
        let (zb, _) = encode_session_zip_bytes(&sess)?;
        let (restored, _) = decode_session_zip_bytes(&zb)?;
        assert_eq!(
            restored.render_opts.residue_colors.len(),
            sess.render_opts.residue_colors.len()
        );
        Ok(())
    }
}

#[cfg(test)]
mod sandbox_confinement_tests {
    use super::*;

    /// Aliases that `COMMAND_REGISTRY` advertises through `help`, `[TAB]` autocomplete and
    /// the Ctrl+K palette, but that `execute_single_cmd` has no arm for: typing them gives
    /// "unknown DSL command". They are classified as compute verbs so the capability tables
    /// remain exhaustive; they are skipped below because the gap is in the interpreter, not
    /// in the classification. Remove them from here once the interpreter accepts them.
    const REGISTRY_ONLY_ALIASES: &[&str] = &["define_axis", "define_plane", "matchalign"];

    /// The whitelist is only as good as its spelling: a typo would silently drop a verb,
    /// and a verb the interpreter no longer has would rot there. Check both against the
    /// interpreter itself rather than against the registry table.
    #[test]
    fn every_classified_verb_is_understood_by_the_interpreter() {
        for verb in COMPUTE_VERBS
            .iter()
            .chain(FILESYSTEM_VERBS.iter())
            .filter(|v| !REGISTRY_ONLY_ALIASES.contains(*v))
        {
            let mut sess = BiomolSession::default();
            let err = sess
                .execute_dsl(verb)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                !err.contains("unknown DSL command"),
                "verb '{verb}' is classified but the interpreter does not know it: {err}"
            );
        }
    }

    /// The lexer tricks that defeated the old substring blacklist, as reproduced in the audit.
    #[test]
    fn vetting_sees_statements_the_way_the_interpreter_does() {
        // verb ends at ANY whitespace, not just a space
        assert!(vet_dsl_for_sandbox("save\t/etc/x.pdb").is_err());
        assert!(vet_dsl_for_sandbox("save\n/etc/x.pdb").is_err());
        assert!(vet_dsl_for_sandbox("save\u{b}/etc/x.pdb").is_err());
        // casing is normalised by the interpreter
        assert!(vet_dsl_for_sandbox("SAVE /etc/x.pdb").is_err());
        assert!(vet_dsl_for_sandbox("SaVe /etc/x.pdb").is_err());
        // a statement hidden behind `;`, in any position
        assert!(vet_dsl_for_sandbox("color mlp; save /etc/x.pdb").is_err());
        assert!(vet_dsl_for_sandbox("save /etc/x.pdb; color mlp").is_err());
        assert!(vet_dsl_for_sandbox("color mlp;;   save\t/etc/x.pdb  ;").is_err());
        // leading whitespace and comment statements
        assert!(vet_dsl_for_sandbox("   save /etc/x.pdb").is_err());
        assert!(vet_dsl_for_sandbox("# save /etc/x.pdb").is_ok());
        // aliases of filesystem verbs
        for alias in ["pse", "cxs", "open_session", "load", "ray", "push"] {
            assert!(
                vet_dsl_for_sandbox(&format!("{alias} /etc/x")).is_err(),
                "alias '{alias}' slipped through"
            );
        }
        // unknown verbs fail closed
        assert!(vet_dsl_for_sandbox("exfiltrate /etc/shadow").is_err());
        // compute-only scripts still run
        assert!(vet_dsl_for_sandbox("color mlp; hbonds 3.5; scene save f1").is_ok());
    }

    #[test]
    fn the_splitter_and_the_verb_lexer_match_the_interpreter() {
        assert_eq!(
            split_dsl_statements(" color mlp ;; # note ; hbonds 3.5 "),
            vec!["color mlp", "hbonds 3.5"]
        );
        assert_eq!(dsl_statement_verb("  SAVE\t/etc/x "), "save");
        assert_eq!(dsl_statement_verb(""), "");
    }
}
