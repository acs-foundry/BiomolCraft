#![forbid(unsafe_code)]

//! Portable Verified BiomolCraft Sessions (`.biomol.zip` — PyMOL `.pse` / ChimeraX `.cxs` equivalent).
//!
//! Stores the complete multi-model workspace (`models/m{id}.pdb`), camera, lighting, styles,
//! overlays, `F1..F12` saved Scenes, 3D/2D labels, Color Key, Scale Bar, axes, planes,
//! measurements, MSA, PAE, optional Cryo-EM `volume.mrc`, optional `viewdock.json` (`ViewDockX`),
//! and scientific provenance inside a CRC-32 verified, path-traversal-proof, bomb-capped PKZIP archive.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use craftbio_scene::{
    ColorKeyLegend, ColorScheme, DockCraftSceneBundle, GridBox3D, Label2D, Label3D, LightingPreset,
    LineOverlay, ProvenanceEntry, RepresentationKind, ScaleBarSpec, SphereOverlay,
    read_verified_zip, read_verified_zip_bytes, write_verified_zip_bytes,
};
use craftbio_struct::{
    BindingPocket3D, GeometryAxis, GeometryPlane, MsaAlignment, PaeMatrix, parse_mrc_map,
    parse_pdb, parse_sdf_poses, write_mrc_map, write_multi_model_pdb, write_pdb,
};
use craftbio_viewport::{Camera3D, RenderOptions};
use serde::{Deserialize, Serialize};

use crate::{BiomolSession, PoseStatusTag, SavedScene, SceneModel, ScriptError, ViewDockSession};

/// Summary returned after saving, loading, or inspecting a `.biomol.zip` portable session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionZipSummary {
    pub schema_version: String,
    pub app_version: String,
    pub archive_bytes: usize,
    pub model_count: usize,
    pub total_atoms: usize,
    pub scene_count: usize,
    pub has_volume_mrc: bool,
    pub has_viewdock: bool,
    pub viewdock_poses: usize,
    pub label_3d_count: usize,
    pub label_2d_count: usize,
    pub has_color_key: bool,
    pub has_scale_bar: bool,
    pub provenance_entries: usize,
    pub model_names: Vec<String>,
}

fn default_shadow_strength() -> f32 {
    0.55
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchivedRenderOpts {
    representation: RepresentationKind,
    color_scheme: ColorScheme,
    show_surface_shell: bool,
    surface_alpha: f32,
    surface_wireframe: bool,
    show_ligands_sticks: bool,
    show_waters: bool,
    show_rna_base_pairs: bool,
    show_silhouettes: bool,
    lighting: LightingPreset,
    ssao_strength: f32,
    #[serde(default = "default_shadow_strength")]
    shadow_strength: f32,
    background: [f32; 4],
    #[serde(default)]
    residue_highlights: Vec<(char, i32, [f32; 3])>,
    #[serde(default)]
    residue_colors: Vec<(char, i32, [f32; 3])>,
    #[serde(default)]
    sphere_overlays: Vec<SphereOverlay>,
    #[serde(default)]
    line_overlays: Vec<LineOverlay>,
    grid_box: Option<GridBox3D>,
    #[serde(default)]
    show_snfg_glycans: bool,
    #[serde(default)]
    show_unit_cell: bool,
}

impl From<&RenderOptions> for ArchivedRenderOpts {
    fn from(o: &RenderOptions) -> Self {
        let mut rh: Vec<(char, i32, [f32; 3])> = o
            .residue_highlights
            .iter()
            .map(|(&(c, r), &rgb)| (c, r, rgb))
            .collect();
        rh.sort_by_key(|&(c, r, _)| (c, r));
        let mut rc: Vec<(char, i32, [f32; 3])> = o
            .residue_colors
            .iter()
            .map(|(&(c, r), &rgb)| (c, r, rgb))
            .collect();
        rc.sort_by_key(|&(c, r, _)| (c, r));
        Self {
            representation: o.representation,
            color_scheme: o.color_scheme,
            show_surface_shell: o.show_surface_shell,
            surface_alpha: o.surface_alpha,
            surface_wireframe: o.surface_wireframe,
            show_ligands_sticks: o.show_ligands_sticks,
            show_waters: o.show_waters,
            show_rna_base_pairs: o.show_rna_base_pairs,
            show_silhouettes: o.show_silhouettes,
            lighting: o.lighting,
            ssao_strength: o.ssao_strength,
            shadow_strength: o.shadow_strength,
            background: o.background,
            residue_highlights: rh,
            residue_colors: rc,
            sphere_overlays: o.sphere_overlays.clone(),
            line_overlays: o.line_overlays.clone(),
            grid_box: o.grid_box,
            show_snfg_glycans: o.show_snfg_glycans,
            show_unit_cell: o.show_unit_cell,
        }
    }
}

impl ArchivedRenderOpts {
    fn into_render_opts(self) -> RenderOptions {
        let mut residue_highlights = HashMap::with_capacity(self.residue_highlights.len());
        for (c, r, rgb) in self.residue_highlights {
            residue_highlights.insert((c, r), rgb);
        }
        let mut residue_colors = HashMap::with_capacity(self.residue_colors.len());
        for (c, r, rgb) in self.residue_colors {
            residue_colors.insert((c, r), rgb);
        }
        RenderOptions {
            representation: self.representation,
            color_scheme: self.color_scheme,
            show_surface_shell: self.show_surface_shell,
            surface_alpha: self.surface_alpha,
            surface_wireframe: self.surface_wireframe,
            show_ligands_sticks: self.show_ligands_sticks,
            show_waters: self.show_waters,
            show_rna_base_pairs: self.show_rna_base_pairs,
            show_silhouettes: self.show_silhouettes,
            lighting: self.lighting,
            ssao_strength: self.ssao_strength,
            shadow_strength: self.shadow_strength,
            background: self.background,
            residue_highlights,
            residue_colors,
            sphere_overlays: self.sphere_overlays,
            line_overlays: self.line_overlays,
            grid_box: self.grid_box,
            show_snfg_glycans: self.show_snfg_glycans,
            show_unit_cell: self.show_unit_cell,
            ..RenderOptions::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchivedModelMeta {
    model_id: u32,
    name: String,
    role: String,
    visible: bool,
    source_path: Option<String>,
    pdb_entry: String,
    atom_count: usize,
    active_frame: usize,
    render_opts: ArchivedRenderOpts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchivedViewDockState {
    bundle: DockCraftSceneBundle,
    active_pose_idx: usize,
    pose_tags: Vec<PoseStatusTag>,
    show_hyde_coronas: bool,
    show_torsion_lights: bool,
    show_interactions: bool,
    show_pharmacophores: bool,
    show_dmif_hotspots: bool,
    show_bridging_waters: bool,
    show_grid_box: bool,
    editable_grid_box: Option<GridBox3D>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionArchiveManifest {
    schema_version: String,
    app_version: String,
    active_model_idx: usize,
    next_model_id: u32,
    camera: Camera3D,
    global_render_opts: ArchivedRenderOpts,
    models: Vec<ArchivedModelMeta>,
    scenes: BTreeMap<String, SavedScene>,
    labels_3d: Vec<Label3D>,
    labels_2d: Vec<Label2D>,
    color_key: Option<ColorKeyLegend>,
    scale_bar: Option<ScaleBarSpec>,
    axes: Vec<GeometryAxis>,
    planes: Vec<GeometryPlane>,
    measurements_log: Vec<String>,
    last_module_markdown: String,
    msa: Option<MsaAlignment>,
    pae: Option<PaeMatrix>,
    volume_sigma: f64,
    volume_wireframe: bool,
    show_orthoplanes: bool,
    orthoplane_frac: [f32; 3],
    volume_label: Option<String>,
    has_volume_mrc: bool,
    has_viewdock: bool,
    viewdock_poses: usize,
    provenance: Vec<ProvenanceEntry>,
    command_history: Vec<String>,
    #[serde(default)]
    binding_pockets: Vec<BindingPocket3D>,
    #[serde(default)]
    selected_pocket_id: Option<usize>,
}

/// Encode the entire multi-model `BiomolSession` into verified `.biomol.zip` bytes.
pub fn encode_session_zip_bytes(
    session: &BiomolSession,
) -> Result<(Vec<u8>, SessionZipSummary), ScriptError> {
    let mut owned_blobs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut model_metas = Vec::with_capacity(session.models.len());
    let mut total_atoms = 0usize;
    let mut model_names = Vec::with_capacity(session.models.len());

    for m in &session.models {
        let entry_name = format!("models/m{}.pdb", m.model_id);
        let pdb_text = if m.molecule.frames.len() > 1 {
            write_multi_model_pdb(&m.molecule)
        } else {
            write_pdb(&m.molecule)
        };
        total_atoms += m.molecule.atoms.len();
        model_names.push(format!("#{} {}", m.model_id, m.name));
        model_metas.push(ArchivedModelMeta {
            model_id: m.model_id,
            name: m.name.clone(),
            role: m.role.clone(),
            visible: m.visible,
            source_path: m.source_path.clone(),
            pdb_entry: entry_name.clone(),
            atom_count: m.molecule.atoms.len(),
            active_frame: m.molecule.active_frame,
            render_opts: ArchivedRenderOpts::from(&m.render_opts),
        });
        owned_blobs.push((entry_name, pdb_text.into_bytes()));
    }

    let mut has_volume_mrc = false;
    let mut volume_label = None;
    if let Some(ref vol) = session.active_volume {
        has_volume_mrc = true;
        volume_label = Some(vol.label.clone());
        owned_blobs.push(("volume.mrc".to_string(), write_mrc_map(vol)));
    }

    let mut has_viewdock = false;
    let mut viewdock_poses = 0usize;
    if let Some(ref vd) = session.viewdock {
        has_viewdock = true;
        viewdock_poses = vd.bundle.poses.len();
        let archived_vd = ArchivedViewDockState {
            bundle: vd.bundle.clone(),
            active_pose_idx: vd.active_pose_idx,
            pose_tags: vd.pose_tags.clone(),
            show_hyde_coronas: vd.show_hyde_coronas,
            show_torsion_lights: vd.show_torsion_lights,
            show_interactions: vd.show_interactions,
            show_pharmacophores: vd.show_pharmacophores,
            show_dmif_hotspots: vd.show_dmif_hotspots,
            show_bridging_waters: vd.show_bridging_waters,
            show_grid_box: vd.show_grid_box,
            editable_grid_box: vd.editable_grid_box,
        };
        let vd_json = serde_json::to_vec_pretty(&archived_vd)
            .map_err(|e| ScriptError::Command(format!("failed to serialize ViewDockX: {e}")))?;
        owned_blobs.push(("viewdock.json".to_string(), vd_json));
    }

    let manifest = SessionArchiveManifest {
        schema_version: "biomol-session/v1".to_string(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        active_model_idx: session.active_model_idx,
        next_model_id: session.next_model_id_counter(),
        camera: session.camera.clone(),
        global_render_opts: ArchivedRenderOpts::from(&session.render_opts),
        models: model_metas,
        scenes: session.scenes.clone(),
        labels_3d: session.labels_3d.clone(),
        labels_2d: session.labels_2d.clone(),
        color_key: session.color_key.clone(),
        scale_bar: session.scale_bar.clone(),
        axes: session.axes.clone(),
        planes: session.planes.clone(),
        measurements_log: session.measurements_log.clone(),
        last_module_markdown: session.last_module_markdown.clone(),
        msa: session.msa.clone(),
        pae: session.pae.clone(),
        volume_sigma: session.active_volume_sigma,
        volume_wireframe: session.volume_wireframe,
        show_orthoplanes: session.show_orthoplanes,
        orthoplane_frac: session.orthoplane_frac,
        volume_label,
        has_volume_mrc,
        has_viewdock,
        viewdock_poses,
        provenance: session.provenance.clone(),
        command_history: session.command_history.clone(),
        binding_pockets: session.binding_pockets.clone(),
        selected_pocket_id: session.selected_pocket_id,
    };

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| ScriptError::Command(format!("failed to serialize manifest: {e}")))?;

    let mut entry_refs: Vec<(&str, &[u8])> = Vec::with_capacity(owned_blobs.len() + 1);
    entry_refs.push(("manifest.json", &manifest_bytes));
    for (name, blob) in &owned_blobs {
        entry_refs.push((name.as_str(), blob.as_slice()));
    }

    let zip_bytes =
        write_verified_zip_bytes(&entry_refs).map_err(|e| ScriptError::Command(e.to_string()))?;

    let summary = SessionZipSummary {
        schema_version: manifest.schema_version,
        app_version: manifest.app_version,
        archive_bytes: zip_bytes.len(),
        model_count: manifest.models.len(),
        total_atoms,
        scene_count: manifest.scenes.len(),
        has_volume_mrc,
        has_viewdock,
        viewdock_poses,
        label_3d_count: manifest.labels_3d.len(),
        label_2d_count: manifest.labels_2d.len(),
        has_color_key: manifest.color_key.is_some(),
        has_scale_bar: manifest.scale_bar.is_some(),
        provenance_entries: manifest.provenance.len(),
        model_names,
    };

    Ok((zip_bytes, summary))
}

/// Save a `BiomolSession` to a CRC-32 verified `.biomol.zip` file on disk.
pub fn save_session_zip(
    session: &BiomolSession,
    path: &Path,
) -> Result<SessionZipSummary, ScriptError> {
    let (bytes, summary) = encode_session_zip_bytes(session)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(summary)
}

/// Decode and CRC-32 verify a `.biomol.zip` byte buffer into a restored `BiomolSession`.
pub fn decode_session_zip_bytes(
    bytes: &[u8],
) -> Result<(BiomolSession, SessionZipSummary), ScriptError> {
    let entries =
        read_verified_zip_bytes(bytes).map_err(|e| ScriptError::Command(e.to_string()))?;
    let manifest_raw = entries
        .get("manifest.json")
        .ok_or_else(|| ScriptError::Command("archive missing manifest.json".to_string()))?;
    let manifest: SessionArchiveManifest = serde_json::from_slice(manifest_raw)
        .map_err(|e| ScriptError::Command(format!("invalid session manifest.json: {e}")))?;

    if manifest.schema_version != "biomol-session/v1" {
        return Err(ScriptError::Command(format!(
            "unsupported session schema_version '{}' (expected 'biomol-session/v1')",
            manifest.schema_version
        )));
    }

    let mut models = Vec::with_capacity(manifest.models.len());
    let mut total_atoms = 0usize;
    let mut model_names = Vec::with_capacity(manifest.models.len());
    for meta in manifest.models {
        let pdb_bytes = entries.get(&meta.pdb_entry).ok_or_else(|| {
            ScriptError::Command(format!(
                "missing model entry '{}' in archive",
                meta.pdb_entry
            ))
        })?;
        let pdb_txt = std::str::from_utf8(pdb_bytes).map_err(|_| {
            ScriptError::Command(format!("non-UTF8 PDB entry '{}'", meta.pdb_entry))
        })?;
        let mut mol =
            parse_pdb(&meta.name, pdb_txt).map_err(|e| ScriptError::Command(e.to_string()))?;
        if meta.active_frame < mol.frames.len() {
            mol.set_active_frame(meta.active_frame);
        }
        total_atoms += mol.atoms.len();
        model_names.push(format!("#{} {}", meta.model_id, meta.name));
        models.push(SceneModel {
            model_id: meta.model_id,
            name: meta.name,
            role: meta.role,
            molecule: mol,
            visible: meta.visible,
            render_opts: meta.render_opts.into_render_opts(),
            source_path: meta.source_path,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        });
    }

    let active_volume = if manifest.has_volume_mrc {
        if let Some(mrc_bytes) = entries.get("volume.mrc") {
            let lbl = manifest.volume_label.as_deref().unwrap_or("session_volume");
            Some(parse_mrc_map(lbl, mrc_bytes).map_err(|e| ScriptError::Command(e.to_string()))?)
        } else {
            None
        }
    } else {
        None
    };

    let viewdock = if manifest.has_viewdock {
        if let Some(vd_bytes) = entries.get("viewdock.json") {
            let avd: ArchivedViewDockState = serde_json::from_slice(vd_bytes).map_err(|e| {
                ScriptError::Command(format!("invalid viewdock.json in archive: {e}"))
            })?;
            let mut pose_models = Vec::with_capacity(avd.bundle.poses.len());
            for p in &avd.bundle.poses {
                let parsed = if !p.sdf_block.trim().is_empty() {
                    parse_sdf_poses(&p.sdf_block)
                        .ok()
                        .and_then(|mut v| v.drain(..).next().map(|r| r.molecule))
                } else {
                    None
                };
                pose_models.push(parsed);
            }
            let n_poses = avd.bundle.poses.len();
            let mut tags = avd.pose_tags;
            if tags.len() < n_poses {
                tags.resize(n_poses, PoseStatusTag::Unreviewed);
            }
            Some(ViewDockSession {
                bundle: avd.bundle,
                active_pose_idx: avd.active_pose_idx.min(n_poses.saturating_sub(1)),
                pose_models,
                pose_tags: tags,
                show_hyde_coronas: avd.show_hyde_coronas,
                show_torsion_lights: avd.show_torsion_lights,
                show_interactions: avd.show_interactions,
                show_pharmacophores: avd.show_pharmacophores,
                show_dmif_hotspots: avd.show_dmif_hotspots,
                show_bridging_waters: avd.show_bridging_waters,
                show_grid_box: avd.show_grid_box,
                editable_grid_box: avd.editable_grid_box,
            })
        } else {
            None
        }
    } else {
        None
    };

    let active_model_idx = if models.is_empty() {
        0
    } else {
        manifest.active_model_idx.min(models.len() - 1)
    };
    let next_id = manifest
        .next_model_id
        .max(models.iter().map(|m| m.model_id).max().unwrap_or(0) + 1);

    let summary = SessionZipSummary {
        schema_version: manifest.schema_version,
        app_version: manifest.app_version,
        archive_bytes: bytes.len(),
        model_count: models.len(),
        total_atoms,
        scene_count: manifest.scenes.len(),
        has_volume_mrc: active_volume.is_some(),
        has_viewdock: viewdock.is_some(),
        viewdock_poses: viewdock.as_ref().map_or(0, |v| v.bundle.poses.len()),
        label_3d_count: manifest.labels_3d.len(),
        label_2d_count: manifest.labels_2d.len(),
        has_color_key: manifest.color_key.is_some(),
        has_scale_bar: manifest.scale_bar.is_some(),
        provenance_entries: manifest.provenance.len(),
        model_names,
    };

    let session = BiomolSession {
        models,
        active_model_idx,
        next_model_id: next_id.max(1),
        scenes: manifest.scenes,
        viewdock,
        active_volume,
        active_volume_sigma: manifest.volume_sigma,
        volume_wireframe: manifest.volume_wireframe,
        show_orthoplanes: manifest.show_orthoplanes,
        orthoplane_frac: manifest.orthoplane_frac,
        camera: manifest.camera,
        render_opts: manifest.global_render_opts.into_render_opts(),
        named_selections: HashMap::new(),
        provenance: manifest.provenance,
        command_history: manifest.command_history,
        last_module_markdown: manifest.last_module_markdown,
        measurements_log: manifest.measurements_log,
        axes: manifest.axes,
        planes: manifest.planes,
        msa: manifest.msa,
        pae: manifest.pae,
        ramachandran: None,
        aggregation: None,
        rotamer_table: None,
        labels_3d: manifest.labels_3d,
        labels_2d: manifest.labels_2d,
        color_key: manifest.color_key,
        scale_bar: manifest.scale_bar,
        binding_pockets: manifest.binding_pockets,
        selected_pocket_id: manifest.selected_pocket_id,
        dirty_geometry: true,
    };

    Ok((session, summary))
}

/// Load and CRC-32 verify a `.biomol.zip` portable session from disk.
pub fn load_session_zip(path: &Path) -> Result<(BiomolSession, SessionZipSummary), ScriptError> {
    let raw = std::fs::read(path)?;
    decode_session_zip_bytes(&raw)
}

/// Inspect a `.biomol.zip` portable session without parsing every 3D coordinate array.
pub fn inspect_session_zip(path: &Path) -> Result<SessionZipSummary, ScriptError> {
    let raw_len = std::fs::metadata(path)?.len() as usize;
    let entries = read_verified_zip(path).map_err(|e| ScriptError::Command(e.to_string()))?;
    let manifest_raw = entries
        .get("manifest.json")
        .ok_or_else(|| ScriptError::Command("archive missing manifest.json".to_string()))?;
    let manifest: SessionArchiveManifest = serde_json::from_slice(manifest_raw)
        .map_err(|e| ScriptError::Command(format!("invalid session manifest.json: {e}")))?;

    let total_atoms = manifest.models.iter().map(|m| m.atom_count).sum();
    let model_names = manifest
        .models
        .iter()
        .map(|m| format!("#{} {}", m.model_id, m.name))
        .collect();
    Ok(SessionZipSummary {
        schema_version: manifest.schema_version,
        app_version: manifest.app_version,
        archive_bytes: raw_len,
        model_count: manifest.models.len(),
        total_atoms,
        scene_count: manifest.scenes.len(),
        has_volume_mrc: manifest.has_volume_mrc,
        has_viewdock: manifest.has_viewdock,
        viewdock_poses: manifest.viewdock_poses,
        label_3d_count: manifest.labels_3d.len(),
        label_2d_count: manifest.labels_2d.len(),
        has_color_key: manifest.color_key.is_some(),
        has_scale_bar: manifest.scale_bar.is_some(),
        provenance_entries: manifest.provenance.len(),
        model_names,
    })
}
