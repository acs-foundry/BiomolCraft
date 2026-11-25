#![forbid(unsafe_code)]

//! `biomolcraft` — Flagship Pure-Rust `wgpu 24 + egui` Multi-Model Biomolecular 3D Studio,
//! Unified Typed Command Palette (`Ctrl+K` & `[TAB]` Autocomplete), Verified `.biomol.zip`
//! Sessions, Live ArtCraft Suite Push, Interactive Modeling & Sculpting Workbench,
//! Native **`ViewDockX`**, Crystallographic Symmetry, 3D SNFG Glycans, Cryo-EM `vop`/Orthoplanes,
//! Chemical Measurements, MSA/PAE, Conformational Morphing & Movie Maker (`v1.0.0`).

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use biomol_model::{
    RamaRegion, RamaResidueKind, classify_ramachandran, tug_atom_with_sculpting,
};
use biomol_modules::{
    ModuleId, discover_installed_plugins, run_adc_module, run_antibody_module, run_enzyme_module,
    run_ncrna_module, run_pk_tmdd_module,
};
use biomol_physics::{OpenMmEnvStatus, check_openmm_biomol_env};
use biomol_rna::{NcRnaAnalysisReport, analyze_ncrna_mrna, find_rna_preset, rna_presets};
use biomol_script::{BiomolSession, PoseStatusTag, autocomplete_dsl, filter_command_palette};
use craftbio_scene::{
    ColorScheme, CraftBioLinkServer, DockCraftSceneBundle, GridBox3D, InboundLinkCommand,
    LightingPreset, RepresentationKind, discover_craft_instances,
};
use craftbio_struct::{ResidueClass, write_pdb};
use craftbio_viewport::{
    SharedViewportScene, pixels_per_angstrom_at_target, project_world_to_screen,
    show_biomol_viewport_ex,
};
use eframe::egui;

/// Path shown in the UI for the bundled pilot structures.
///
/// Rooted at the Craft data directory (`CRAFT_DATA_DIR`, else `$XDG_DATA_HOME/CraftBio`,
/// else `$HOME/.local/share/CraftBio`) so it means something on any machine.
fn example_structure_path(file_name: &str) -> String {
    let non_blank = |k: &str| {
        std::env::var(k)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let base = if let Some(dir) = non_blank("CRAFT_DATA_DIR") {
        PathBuf::from(dir)
    } else if let Some(xdg) = non_blank("XDG_DATA_HOME") {
        PathBuf::from(xdg).join("CraftBio")
    } else if let Some(home) = non_blank("HOME") {
        PathBuf::from(home).join(".local/share/CraftBio")
    } else {
        PathBuf::from("CraftBio")
    };
    base.join("campaigns/biologics_pilot")
        .join(file_name)
        .to_string_lossy()
        .into_owned()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PickMode {
    Inspect,
    Distance,
    Angle,
    Dihedral,
    SculptDrag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightModuleTab {
    ViewDockX,
    Measure,
    RamaAgg,
    MsaPae,
    MovieMorph,
    SymGlycans,
    Modeling,
    Physics,
    CryoEm,
    Antibodies,
    NcRnaMrna,
    Enzymes,
    AdcPk,
    DeNovo,
    SuitePlugins,
    Evidence,
}

#[derive(Debug, Clone, Default)]
struct StartupArgs {
    open_file: Option<PathBuf>,
    dock_bundle: Option<PathBuf>,
}

fn parse_startup_args() -> StartupArgs {
    let mut out = StartupArgs::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--dock-bundle" {
            if let Some(p) = args.next() {
                out.dock_bundle = Some(PathBuf::from(p));
            }
        } else if !arg.starts_with('-') && out.open_file.is_none() {
            out.open_file = Some(PathBuf::from(arg));
        }
    }
    out
}

pub struct BiomolCraftApp {
    dark_mode: bool,
    session: BiomolSession,
    viewport_scene: SharedViewportScene,
    link_server: Option<CraftBioLinkServer>,

    // UI state
    right_tab: RightModuleTab,
    open_path_input: String,
    fetch_id_input: String,
    console_input: String,
    console_log: Vec<String>,
    picked_atom_info: String,
    status_line: String,

    // Trajectory playback state
    playing_trajectory: bool,
    traj_fps: f32,
    last_frame_tick: Instant,

    // Physics & Cryo-EM / De Novo state
    anm_mode_sel: usize,
    anm_amp_a: f64,
    openmm_status: OpenMmEnvStatus,
    cryoem_res_a: f64,
    cryoem_sigma: f64,
    denovo_chain: String,
    denovo_hotspots: String,
    live_peers_summary: String,

    // ViewDockX ligand path for "Dock Here in DockCraft"
    dock_ligand_input: String,

    // Mutagenesis wizard state
    mut_chain: String,
    mut_res_seq: i32,
    mut_target_aa: char,
    mut_rotamer: usize,

    // ncRNA / mRNA module state
    rna_name: String,
    rna_seq: String,
    rna_report: Option<NcRnaAnalysisReport>,

    // ADC / PK module state
    adc_linker: String,
    adc_payload: String,
    adc_dar: f64,

    // Wave 5: Interactive 3D Measurement Wizard, Overlays, PAE & Movie/Morph state
    pick_mode: PickMode,
    pick_buffer: Vec<usize>,
    label2d_input: String,
    scalebar_len_a: f32,
    pae_json_input: String,
    morph_frames: usize,
    movie_mode: String,
    movie_out_path: String,
    movie_frames: usize,
    movie_width: u32,
    movie_height: u32,
    movie_fps: u16,

    // Wave 6: swapna, bond_rot, tug/sculpting, 3D Builder, Symmetry/SNFG & vop state
    swapna_base: char,
    bond_rot_atom_a: String,
    bond_rot_atom_b: String,
    bond_rot_deg: f64,
    tug_atom_spec: String,
    tug_shift: [f64; 3],
    build_kind: String,
    build_seq: String,
    sym_cutoff_a: f64,
    vop_zone_rad: f64,

    // Wave 7: Unified Command Palette (Ctrl+K), Console History, & Portable .biomol.zip Sessions
    history_cursor: Option<usize>,
    palette_open: bool,
    palette_query: String,
    session_zip_path: String,

    // Wave 8 (v1.1.0): Live Mouse Drag-to-Sculpt & GROMACS .xtc loader state
    sculpt_drag_atom: Option<usize>,
    sculpt_last_feedback: String,
    xtc_path_input: String,

    // Wave 9 (v1.2.0): Interactive Ramachandran Plot & 3D Aggregation / Solubility Rescue state
    rama_contour_kind: RamaResidueKind,
    agg_dynamic_mode: bool,
    agg_rescue_chain: String,
    agg_rescue_res: i32,
    agg_rescue_aa: char,
}

impl Default for BiomolCraftApp {
    fn default() -> Self {
        Self::with_startup(StartupArgs::default())
    }
}

impl BiomolCraftApp {
    fn with_startup(startup: StartupArgs) -> Self {
        let link_server =
            CraftBioLinkServer::start("biomolcraft", env!("CARGO_PKG_VERSION"), None).ok();
        let link_status = if link_server.is_some() {
            "Craft Link v1 socket active (bidirectional DockCraft/ChemCraft/AdmetCraft/ArtCraft)."
        } else {
            "Local standalone session."
        };

        let mut app = Self {
            dark_mode: true,
            session: BiomolSession::default(),
            viewport_scene: SharedViewportScene::default(),
            link_server,
            right_tab: RightModuleTab::Antibodies,
            open_path_input: example_structure_path("1N8Z.pdb"),
            fetch_id_input: "1CRN".to_string(),
            console_input: String::new(),
            console_log: vec![
                "BiomolCraft v1.2.0 — Interactive Ramachandran · 3D Aggregation & Rescue · Dunbrack χ1-χ4 Rotamers · SASA Burial · Rainbow/B-Factor · Auto-PAE · ViewDockX. Type 'help'.".to_string(),
                link_status.to_string(),
            ],
            picked_atom_info: "Click any atom/residue in 3D or sequence strip to inspect".to_string(),
            status_line: link_status.to_string(),
            playing_trajectory: false,
            traj_fps: 12.0,
            last_frame_tick: Instant::now(),
            anm_mode_sel: 1,
            anm_amp_a: 2.2,
            openmm_status: check_openmm_biomol_env(),
            cryoem_res_a: 4.5,
            cryoem_sigma: 1.5,
            denovo_chain: "C".to_string(),
            denovo_hotspots: "C558 C560 C573".to_string(),
            live_peers_summary: "Click 'Scan Craft Link v1 Peers' to detect live CraftBio & ArtCraft apps.".to_string(),
            dock_ligand_input: String::new(),
            mut_chain: "B".to_string(),
            mut_res_seq: 105,
            mut_target_aa: 'A',
            mut_rotamer: 0,
            rna_name: "xpt-pbuX Guanine Riboswitch (1Y26)".to_string(),
            rna_seq: find_rna_preset("guanine_riboswitch")
                .map_or_else(String::new, |p| p.sequence.to_string()),
            rna_report: None,
            adc_linker: "ggfg".to_string(),
            adc_payload: "dxd".to_string(),
            adc_dar: 8.0,
            pick_mode: PickMode::Inspect,
            pick_buffer: Vec::new(),
            label2d_input: "BiomolCraft v1.1.0".to_string(),
            scalebar_len_a: 20.0,
            pae_json_input: String::new(),
            morph_frames: 20,
            movie_mode: "turn".to_string(),
            movie_out_path: "/tmp/biomolcraft_movie.gif".to_string(),
            movie_frames: 24,
            movie_width: 800,
            movie_height: 600,
            movie_fps: 12,
            swapna_base: 'A',
            bond_rot_atom_a: "B:105:CA".to_string(),
            bond_rot_atom_b: "B:105:CB".to_string(),
            bond_rot_deg: 30.0,
            tug_atom_spec: "B:105:CA".to_string(),
            tug_shift: [0.8, 0.0, 0.0],
            build_kind: "helix".to_string(),
            build_seq: "AKLTYWDSHEAA".to_string(),
            sym_cutoff_a: 5.0,
            vop_zone_rad: 5.0,
            history_cursor: None,
            palette_open: false,
            palette_query: String::new(),
            session_zip_path: "/tmp/biomolcraft_session.biomol.zip".to_string(),
            sculpt_drag_atom: None,
            sculpt_last_feedback: "Hold Shift + Left-Drag any atom (or select Sculpt Drag mode) to tug with live MM relaxation.".to_string(),
            xtc_path_input: String::new(),
            rama_contour_kind: RamaResidueKind::General,
            agg_dynamic_mode: false,
            agg_rescue_chain: "A".to_string(),
            agg_rescue_res: 10,
            agg_rescue_aa: 'D',
        };

        if let Some(bundle_path) = startup.dock_bundle {
            if let Ok(raw) = std::fs::read_to_string(&bundle_path)
                && let Ok(bundle) = serde_json::from_str::<DockCraftSceneBundle>(&raw)
                && let Ok(msg) = app.session.load_dockcraft_bundle(bundle)
            {
                app.right_tab = RightModuleTab::ViewDockX;
                app.console_log.push(msg.clone());
                app.status_line = msg;
            }
        } else if let Some(open_p) = startup.open_file {
            app.open_path_input = open_p.display().to_string();
            let low = open_p.to_string_lossy().to_ascii_lowercase();
            if low.ends_with(".biomol.zip") {
                if let Ok((restored, sum)) = biomol_script::load_session_zip(&open_p) {
                    app.session = restored;
                    app.session_zip_path = open_p.display().to_string();
                    if sum.has_viewdock {
                        app.right_tab = RightModuleTab::ViewDockX;
                    }
                    let msg = format!(
                        "Loaded verified portable session '{}' ({} models, {} scenes, ViewDockX={}).",
                        open_p.display(),
                        sum.model_count,
                        sum.scene_count,
                        sum.has_viewdock
                    );
                    app.console_log.push(msg.clone());
                    app.status_line = msg;
                }
            } else if low.ends_with(".dockbundle.json") {
                if let Ok(raw) = std::fs::read_to_string(&open_p)
                    && let Ok(bundle) = serde_json::from_str::<DockCraftSceneBundle>(&raw)
                    && let Ok(msg) = app.session.load_dockcraft_bundle(bundle)
                {
                    app.right_tab = RightModuleTab::ViewDockX;
                    app.console_log.push(msg.clone());
                    app.status_line = msg;
                }
            } else if let Ok(msg) = app.session.load_structure_file(&open_p) {
                app.console_log.push(msg.clone());
                app.status_line = msg;
            }
        } else {
            let default_pdb = PathBuf::from(&app.open_path_input);
            if default_pdb.is_file()
                && let Ok(msg) = app.session.load_structure_file(&default_pdb)
            {
                app.console_log.push(msg.clone());
                app.status_line = msg;
            }
        }
        if let Ok(r) = analyze_ncrna_mrna(&app.rna_name, "ncRNA — Riboswitch", &app.rna_seq, None)
        {
            app.rna_report = Some(r);
        }
        app.sync_viewport_scene();
        app
    }

    fn sync_viewport_scene(&mut self) {
        if self.session.dirty_geometry {
            self.session.sync_active_render_opts();
            let data = self.session.build_viewport_scene();
            self.viewport_scene.update(data);
            self.session.dirty_geometry = false;
        }
        // Always keep synchronous `biomol.get_selection_box` snapshot fresh for DockCraft!
        if let Some(ref srv) = self.link_server {
            srv.set_selection_box(self.session.current_selection_box_snapshot(5.0));
        }
    }

    fn run_dsl(&mut self, cmd: &str) {
        match self.session.execute_dsl(cmd) {
            Ok(out) => {
                self.console_log.push(format!("> {cmd}"));
                for line in out.lines() {
                    self.console_log.push(line.to_string());
                }
                self.status_line = out.lines().last().unwrap_or("OK").to_string();
            }
            Err(e) => {
                self.console_log.push(format!("> {cmd}"));
                self.console_log.push(format!("ERROR: {e}"));
                self.status_line = format!("Error: {e}");
            }
        }
        if self.console_log.len() > 200 {
            let drain = self.console_log.len() - 200;
            self.console_log.drain(..drain);
        }
        self.sync_viewport_scene();
    }

    fn poll_craft_link_inbox(&mut self) {
        let cmds: Vec<InboundLinkCommand> = self
            .link_server
            .as_ref()
            .map(CraftBioLinkServer::drain_inbox)
            .unwrap_or_default();
        for cmd in cmds {
            match cmd {
                InboundLinkCommand::OpenPdb {
                    path,
                    source_app,
                    module_hint,
                } => {
                    self.open_path_input = path.clone();
                    match self.session.load_structure_file(&PathBuf::from(&path)) {
                        Ok(msg) => {
                            self.console_log
                                .push(format!("[Craft Link ← {source_app}] {msg}"));
                            self.status_line = msg;
                            if let Some(m) = module_hint {
                                self.run_dsl(&format!("module {m}"));
                            }
                        }
                        Err(e) => self.console_log.push(format!("[Craft Link ERROR] {e}")),
                    }
                }
                InboundLinkCommand::LoadTrajectory {
                    pdb_path,
                    dcd_path,
                    source_app,
                } => {
                    self.open_path_input = pdb_path.clone();
                    if let Ok(msg) = self.session.load_structure_file(&PathBuf::from(&pdb_path)) {
                        self.console_log
                            .push(format!("[Craft Link ← {source_app}] {msg}"));
                        self.run_dsl(&format!("load_dcd {dcd_path}"));
                        self.playing_trajectory = true;
                    }
                }
                InboundLinkCommand::LoadScene(doc) => {
                    if let Some(ref raw) = doc.inline_pdb {
                        let _ =
                            self.session
                                .load_structure_text(&doc.title, raw, doc.pdb_path.clone());
                    } else if let Some(ref p) = doc.pdb_path {
                        let _ = self.session.load_structure_file(&PathBuf::from(p));
                    }
                    self.session.render_opts.representation = doc.representation;
                    self.session.render_opts.color_scheme = doc.color_scheme;
                    self.session.render_opts.sphere_overlays = doc.sphere_overlays;
                    self.session.render_opts.line_overlays = doc.line_overlays;
                    self.session.sync_active_render_opts();
                    self.session.dirty_geometry = true;
                    self.console_log.push(format!(
                        "[Craft Link ← {}] Loaded scene '{}'",
                        doc.source_app, doc.title
                    ));
                }
                InboundLinkCommand::LoadDockCraftBundle(bundle) => {
                    let src = bundle.source_app.clone();
                    match self.session.load_dockcraft_bundle(*bundle) {
                        Ok(msg) => {
                            self.right_tab = RightModuleTab::ViewDockX;
                            self.console_log.push(format!("[Craft Link ← {src}] {msg}"));
                            self.status_line = msg;
                        }
                        Err(e) => {
                            self.console_log
                                .push(format!("[Craft Link ← {src} ERROR] {e}"));
                        }
                    }
                }
                InboundLinkCommand::ImportArtCraftBundle(payload) => {
                    let msg = format!(
                        "[Craft Link ← {}] ArtCraft bundle '{}' staged at {}",
                        payload.source_app, payload.title, payload.bundle_dir
                    );
                    self.console_log.push(msg.clone());
                    self.status_line = msg;
                }
                InboundLinkCommand::RunDsl {
                    command,
                    source_app,
                } => {
                    self.console_log
                        .push(format!("[Craft Link ← {source_app}] DSL: {command}"));
                    self.run_dsl(&command);
                }
            }
        }
        self.sync_viewport_scene();
    }

    fn spawn_sibling(&mut self, bin: &str) {
        match Command::new(bin).spawn() {
            Ok(_) => self.status_line = format!("Launched {bin}"),
            Err(e) => self.status_line = format!("Could not launch {bin}: {e}"),
        }
    }

    fn dock_here_in_dockcraft(&mut self) {
        let Some(mol) = self.session.active_molecule().cloned() else {
            self.status_line = "Load a receptor structure first.".to_string();
            return;
        };
        let Some(snap) = self.session.current_selection_box_snapshot(5.0) else {
            self.status_line =
                "Select binding-site residues or set a GridBox in ViewDockX first.".to_string();
            return;
        };
        let tmp_dir = std::env::temp_dir().join("biomolcraft_dock_handoff");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let rec_path = tmp_dir.join("receptor.pdb");
        if let Err(e) = std::fs::write(&rec_path, write_pdb(&mol)) {
            self.status_line = format!("Failed to write temp receptor: {e}");
            return;
        }
        let lig_path_str = self.dock_ligand_input.trim().to_string();
        if lig_path_str.is_empty() {
            // Launch DockCraft GUI preloaded with receptor so user can pick library/ligand interactively
            match Command::new("dockcraft").arg(&rec_path).spawn() {
                Ok(_) => {
                    self.status_line =
                        "Launched DockCraft GUI with active receptor — click 'Box ← BiomolCraft' to sync GridBox!"
                            .to_string();
                }
                Err(e) => self.status_line = format!("Could not launch dockcraft: {e}"),
            }
            return;
        }
        let out_bundle = tmp_dir.join("viewdock_result.dockbundle.json");
        let out_sdf = tmp_dir.join("docked_poses.sdf");
        let gb = snap.grid_box;
        let center_s = format!(
            "{:.3},{:.3},{:.3}",
            gb.center[0], gb.center[1], gb.center[2]
        );
        let size_s = format!("{:.2},{:.2},{:.2}", gb.size[0], gb.size[1], gb.size[2]);
        let res = Command::new("dockcraft-cli")
            .args([
                "dock",
                "--receptor",
                &rec_path.display().to_string(),
                "--ligand",
                &lig_path_str,
                "--center",
                &center_s,
                "--size",
                &size_s,
                "--rerank-hyde",
                "--out-sdf",
                &out_sdf.display().to_string(),
                "--out-dock-bundle",
                &out_bundle.display().to_string(),
            ])
            .output();
        match res {
            Ok(out) if out.status.success() && out_bundle.is_file() => {
                self.run_dsl(&format!("load_dock {}", out_bundle.display()));
                self.right_tab = RightModuleTab::ViewDockX;
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                self.status_line = format!("dockcraft-cli dock failed: {err}");
                self.console_log.push(self.status_line.clone());
            }
            Err(e) => self.status_line = format!("Could not invoke dockcraft-cli: {e}"),
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // Ctrl+K toggles the unified Command Palette anytime (even while editing text!)
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::K)) {
            self.palette_open = !self.palette_open;
            if self.palette_open {
                self.palette_query.clear();
            }
            return;
        }
        if self.palette_open && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.palette_open = false;
            return;
        }
        if ctx.wants_keyboard_input() {
            return;
        }
        let shift = ctx.input(|i| i.modifiers.shift);
        let fkeys = [
            (egui::Key::F1, "f1"),
            (egui::Key::F2, "f2"),
            (egui::Key::F3, "f3"),
            (egui::Key::F4, "f4"),
            (egui::Key::F5, "f5"),
            (egui::Key::F6, "f6"),
            (egui::Key::F7, "f7"),
            (egui::Key::F8, "f8"),
            (egui::Key::F9, "f9"),
            (egui::Key::F10, "f10"),
            (egui::Key::F11, "f11"),
            (egui::Key::F12, "f12"),
        ];
        for (k, name) in fkeys {
            if ctx.input(|i| i.key_pressed(k)) {
                if shift {
                    self.run_dsl(&format!("scene save {name}"));
                } else {
                    self.run_dsl(&format!("scene recall {name}"));
                }
            }
        }
        // Instant ViewDockX pose scrubbing with Up / Down arrow keys!
        if let Some(ref vd) = self.session.viewdock
            && !vd.bundle.poses.is_empty()
        {
            let cur = vd.active_pose_idx;
            let len = vd.bundle.poses.len();
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                let next = (cur + 1) % len;
                let _ = self.session.select_dock_pose(next);
                self.sync_viewport_scene();
            } else if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                let prev = if cur == 0 { len - 1 } else { cur - 1 };
                let _ = self.session.select_dock_pose(prev);
                self.sync_viewport_scene();
            }
        }
    }
}

impl eframe::App for BiomolCraftApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_craft_link_inbox();
        self.handle_shortcuts(ctx);

        // Smooth trajectory animation playback for ANM modes, Rust MD, and OpenMM DCD trajectories
        if self.playing_trajectory {
            let interval =
                Duration::from_secs_f32((1.0 / self.traj_fps.clamp(1.0, 60.0)).max(0.016));
            if self.last_frame_tick.elapsed() >= interval {
                if let Some(mol) = self.session.active_molecule_mut()
                    && mol.frames.len() > 1
                {
                    let next = (mol.active_frame + 1) % mol.frames.len();
                    mol.set_active_frame(next);
                    self.session.dirty_geometry = true;
                    self.last_frame_tick = Instant::now();
                } else {
                    self.playing_trajectory = false;
                }
                self.sync_viewport_scene();
            }
            ctx.request_repaint_after(interval);
        }
        ctx.set_visuals(if self.dark_mode {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });

        // Top Maestro-style Menu Bar (`File | View | Modeling | Modules | Suite`)
        egui::TopBottomPanel::top("biomol_menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Load Pilot: 1N8Z (Trastuzumab – HER2)").clicked() {
                        self.open_path_input = example_structure_path("1N8Z.pdb");
                        self.run_dsl(&format!("open {} replace", self.open_path_input));
                        ui.close_menu();
                    }
                    if ui
                        .button("Load Pilot: 5JDS (Envafolimab VHH – PD-L1)")
                        .clicked()
                    {
                        self.open_path_input = example_structure_path("5JDS.pdb");
                        self.run_dsl(&format!("open {} replace", self.open_path_input));
                        ui.close_menu();
                    }
                    if ui
                        .button("Load Pilot: 1Y26 (Guanine Riboswitch ncRNA)")
                        .clicked()
                    {
                        self.open_path_input = example_structure_path("1Y26.pdb");
                        self.run_dsl(&format!("open {} replace", self.open_path_input));
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui
                        .button("Render 4K PNG (/tmp/biomolcraft_4k.png)")
                        .clicked()
                    {
                        self.run_dsl("png /tmp/biomolcraft_4k.png 3840 2160");
                        ui.close_menu();
                    }
                    if ui
                        .button("Export Binary glTF 2.0 (/tmp/biomolcraft.glb)")
                        .clicked()
                    {
                        self.run_dsl("export_mesh /tmp/biomolcraft.glb");
                        ui.close_menu();
                    }
                    if ui
                        .button("Export Wavefront OBJ (/tmp/biomolcraft.obj)")
                        .clicked()
                    {
                        self.run_dsl("export_mesh /tmp/biomolcraft.obj");
                        ui.close_menu();
                    }
                    if ui
                        .button("Save CraftScene (/tmp/biomolcraft.craftscene.json)")
                        .clicked()
                    {
                        self.run_dsl("save /tmp/biomolcraft.craftscene.json");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Save Portable Session (.biomol.zip)").clicked() {
                        let p = self.session_zip_path.clone();
                        self.run_dsl(&format!("save_session {p}"));
                        ui.close_menu();
                    }
                    if ui.button("Load Portable Session (.biomol.zip)").clicked() {
                        let p = self.session_zip_path.clone();
                        self.run_dsl(&format!("load_session {p}"));
                        ui.close_menu();
                    }
                    if ui.button("Command Palette (Ctrl+K)").clicked() {
                        self.palette_open = true;
                        self.palette_query.clear();
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Preset: ChimeraX Soft (White + SSAO)").clicked() {
                        self.run_dsl("preset chimerax_soft");
                        ui.close_menu();
                    }
                    if ui.button("Preset: PyMOL Publication White").clicked() {
                        self.run_dsl("preset pub_white");
                        ui.close_menu();
                    }
                    if ui.button("Preset: Dark Studio Rim").clicked() {
                        self.run_dsl("preset pub_dark");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui
                        .button(if self.dark_mode {
                            "Light UI Chrome"
                        } else {
                            "Dark UI Chrome"
                        })
                        .clicked()
                    {
                        self.dark_mode = !self.dark_mode;
                        ui.close_menu();
                    }
                    if ui.button("Center / Reset Camera").clicked() {
                        self.run_dsl("reset");
                        ui.close_menu();
                    }
                    if ui
                        .checkbox(&mut self.session.camera.orthographic, "Orthographic Camera")
                        .changed()
                    {
                        ui.close_menu();
                    }
                });
                ui.menu_button("Modeling", |ui| {
                    if ui.button("Undo Last 3D Edit").clicked() {
                        self.run_dsl("undo");
                        ui.close_menu();
                    }
                    if ui.button("Redo 3D Edit").clicked() {
                        self.run_dsl("redo");
                        ui.close_menu();
                    }
                    if ui.button("Minimize Local MM (80 steps)").clicked() {
                        self.run_dsl("minimize 80");
                        ui.close_menu();
                    }
                });
                ui.menu_button("Modules", |ui| {
                    if ui.button("ViewDockX (DockCraft Poses & HYDE)").clicked() {
                        self.right_tab = RightModuleTab::ViewDockX;
                        ui.close_menu();
                    }
                    ui.separator();
                    for (tab, id) in [
                        (RightModuleTab::Antibodies, ModuleId::AntibodiesVhh),
                        (RightModuleTab::NcRnaMrna, ModuleId::NcRnaMrna),
                        (RightModuleTab::Enzymes, ModuleId::EnzymesCatalysis),
                        (RightModuleTab::AdcPk, ModuleId::AdcsBioconjugates),
                    ] {
                        if ui.button(id.label()).clicked() {
                            self.right_tab = tab;
                            self.run_dsl(&format!("module {}", id.short_key()));
                            ui.close_menu();
                        }
                    }
                });
                ui.menu_button("Suite (CraftBio & ArtCraft)", |ui| {
                    for (lbl, bin) in [
                        ("DockCraft (SBDD Docking & FEP)", "dockcraft"),
                        ("ChemCraft (Small-Mol, Synthesis & NRPs)", "chemcraft"),
                        ("AdmetCraft (ADMET & PBPK)", "admetcraft"),
                    ] {
                        if ui.button(lbl).clicked() {
                            self.spawn_sibling(bin);
                            ui.close_menu();
                        }
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("⌘ Ctrl+K Palette")
                        .on_hover_text("Open Unified BiomolCraft Command Palette (Ctrl+K)")
                        .clicked()
                    {
                        self.palette_open = !self.palette_open;
                        self.palette_query.clear();
                    }
                    ui.label(
                        egui::RichText::new(
                            "BiomolCraft v1.1.0 · Soft Shadows · .xtc · 3D Pockets · Drag-to-Sculpt · ViewDockX",
                        )
                        .strong()
                        .color(egui::Color32::from_rgb(56, 189, 248)),
                    );
                });
            });
        });

        // Bottom Dock: Clickable Sequence Strip + Trajectory Scrubber + PyMOL/ChimeraX Command Console
        egui::TopBottomPanel::bottom("biomol_bottom")
            .resizable(true)
            .default_height(155.0)
            .show(ctx, |ui| {
                let mut click_target: Option<(char, i32, char)> = None;
                let mut requested_frame: Option<usize> = None;
                if let Some(mol) = self.session.active_molecule() {
                    ui.horizontal(|ui| {
                        ui.strong("Sequence Strip:");
                        if mol.frames.len() > 1 {
                            let play_lbl = if self.playing_trajectory {
                                "⏸ Pause"
                            } else {
                                "▶ Play"
                            };
                            if ui.small_button(play_lbl).clicked() {
                                self.playing_trajectory = !self.playing_trajectory;
                                self.last_frame_tick = Instant::now();
                            }
                            let mut cur = mol.active_frame;
                            let max_f = mol.frames.len().saturating_sub(1);
                            if ui
                                .add(egui::Slider::new(&mut cur, 0..=max_f).text("Frame"))
                                .changed()
                            {
                                requested_frame = Some(cur);
                            }
                            ui.label(format!("({}/{})", cur + 1, mol.frames.len()));
                        }
                        ui.small(&self.picked_atom_info);
                    });
                    egui::ScrollArea::horizontal()
                        .max_height(34.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                for ch in &mol.chains {
                                    if !matches!(
                                        ch.dominant_class,
                                        ResidueClass::Protein
                                            | ResidueClass::Rna
                                            | ResidueClass::Dna
                                    ) {
                                        continue;
                                    }
                                    ui.colored_label(
                                        egui::Color32::from_rgb(56, 189, 248),
                                        format!("[{}]", ch.id),
                                    );
                                    for r in ch.residues.iter().take(140) {
                                        if matches!(
                                            r.class,
                                            ResidueClass::Protein
                                                | ResidueClass::Rna
                                                | ResidueClass::Dna
                                        ) && ui
                                            .small_button(format!("{}", r.code))
                                            .on_hover_text(format!(
                                                "{}:{}{}",
                                                r.chain, r.res_name, r.res_seq
                                            ))
                                            .clicked()
                                        {
                                            click_target = Some((r.chain, r.res_seq, r.code));
                                        }
                                    }
                                }
                            });
                        });
                }
                if let Some(f_idx) = requested_frame
                    && let Some(mol) = self.session.active_molecule_mut()
                {
                    mol.set_active_frame(f_idx);
                    self.session.dirty_geometry = true;
                    self.sync_viewport_scene();
                }
                if let Some((c, rseq, code)) = click_target {
                    self.mut_chain = c.to_string();
                    self.mut_res_seq = rseq;
                    self.picked_atom_info = format!("Selected {c}:{code}{rseq}");
                    self.run_dsl(&format!("select clicked, chain {c} and resi {rseq}"));
                }

                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(66.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in self.console_log.iter().rev().take(12).rev() {
                            ui.monospace(line);
                        }
                    });
                let mut execute_now: Option<String> = None;
                let suggestions = if self.console_input.trim().is_empty() {
                    Vec::new()
                } else {
                    autocomplete_dsl(&self.console_input)
                };
                ui.horizontal(|ui| {
                    ui.strong("DSL>");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.console_input)
                            .desired_width((ui.available_width() - 125.0).max(160.0))
                            .hint_text(
                                "[TAB] autocomplete · ↑/↓ history · Ctrl+K palette · e.g. save_session /tmp/s.biomol.zip",
                            ),
                    );
                    if resp.has_focus() {
                        // [TAB] Autocomplete from unified COMMAND_REGISTRY
                        if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Tab))
                            && let Some(first) = suggestions.first()
                        {
                            self.console_input = first.insert_text.clone();
                        }
                        // Up / Down arrow command history recall
                        let hlen = self.session.command_history.len();
                        if hlen > 0 {
                            if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                                let next = match self.history_cursor {
                                    None => hlen - 1,
                                    Some(0) => 0,
                                    Some(idx) => idx - 1,
                                };
                                self.history_cursor = Some(next);
                                self.console_input = self.session.command_history[next].clone();
                            } else if ui.input(|i| i.key_pressed(egui::Key::ArrowDown))
                                && let Some(idx) = self.history_cursor
                            {
                                if idx + 1 < hlen {
                                    self.history_cursor = Some(idx + 1);
                                    self.console_input =
                                        self.session.command_history[idx + 1].clone();
                                } else {
                                    self.history_cursor = None;
                                    self.console_input.clear();
                                }
                            }
                        }
                    }
                    if (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        || ui.button("Run").clicked()
                    {
                        let cmd = std::mem::take(&mut self.console_input);
                        self.history_cursor = None;
                        if !cmd.trim().is_empty() {
                            execute_now = Some(cmd);
                        }
                    }
                    if ui
                        .small_button("Ctrl+K")
                        .on_hover_text("Open Command Palette")
                        .clicked()
                    {
                        self.palette_open = true;
                        self.palette_query.clear();
                    }
                });
                if !suggestions.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        ui.small("[TAB]:");
                        for item in suggestions.iter().take(5) {
                            if ui
                                .small_button(&item.display_label)
                                .on_hover_text(item.description)
                                .clicked()
                            {
                                self.console_input = item.insert_text.clone();
                            }
                        }
                    });
                }
                if let Some(cmd) = execute_now {
                    self.run_dsl(&cmd);
                }
            });

        // Unified `Ctrl+K` Command Palette Modal (driven by `biomol_script::COMMAND_REGISTRY`)
        if self.palette_open {
            let mut close_pal = false;
            let mut chosen_cmd: Option<(String, bool)> = None;
            egui::Window::new("BiomolCraft Command Palette (Ctrl+K)")
                .collapsible(false)
                .resizable(true)
                .default_width(580.0)
                .default_height(380.0)
                .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 56.0))
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        let r = ui.add(
                            egui::TextEdit::singleline(&mut self.palette_query)
                                .desired_width(ui.available_width() - 62.0)
                                .hint_text("Type command, category, or keyword (e.g. session, glycan, vop, push, morph)..."),
                        );
                        r.request_focus();
                        if ui.button("Close").clicked() {
                            close_pal = true;
                        }
                    });
                    ui.separator();
                    let matches = filter_command_palette(&self.palette_query);
                    if ui.input(|i| i.key_pressed(egui::Key::Enter))
                        && let Some(first) = matches.first()
                    {
                        let run_direct = !first.syntax.contains('<') && !first.syntax.contains('[');
                        chosen_cmd = Some((
                            if run_direct {
                                first.name.to_string()
                            } else {
                                format!("{} ", first.name)
                            },
                            run_direct,
                        ));
                    }
                    ui.small(format!(
                        "{} matching command(s) — click to insert/run, Esc to close",
                        matches.len()
                    ));
                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            for spec in matches.iter().take(40) {
                                let run_direct =
                                    !spec.syntax.contains('<') && !spec.syntax.contains('[');
                                ui.horizontal(|ui| {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(56, 189, 248),
                                        format!("[{}]", spec.category),
                                    );
                                    if ui.button(spec.syntax).clicked() {
                                        chosen_cmd = Some((
                                            if run_direct {
                                                spec.name.to_string()
                                            } else {
                                                format!("{} ", spec.name)
                                            },
                                            run_direct,
                                        ));
                                    }
                                });
                                ui.small(spec.description);
                                ui.separator();
                            }
                        });
                });
            if close_pal {
                self.palette_open = false;
            }
            if let Some((txt, run_direct)) = chosen_cmd {
                self.palette_open = false;
                if run_direct {
                    self.run_dsl(&txt);
                } else {
                    self.console_input = txt;
                }
            }
        }

        // Left Panel: Multi-Model Panel (#1..#N), Online Fetch, Representations, Colors, Side View & Scenes
        egui::SidePanel::left("biomol_scene_tree")
            .resizable(true)
            .default_width(270.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Models & Fetch");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.open_path_input)
                                .desired_width(142.0),
                        );
                        if ui.small_button("Open").clicked() {
                            let p = self.open_path_input.clone();
                            self.run_dsl(&format!("open {p} replace"));
                        }
                        if ui.small_button("+Add").clicked() {
                            let p = self.open_path_input.clone();
                            self.run_dsl(&format!("open {p} add"));
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.fetch_id_input)
                                .desired_width(165.0)
                                .hint_text("1N8Z | af:P04626 | pubchem:2244"),
                        );
                        if ui.small_button("Fetch").clicked() {
                            let id = self.fetch_id_input.clone();
                            self.run_dsl(&format!("fetch {id}"));
                        }
                    });

                    ui.separator();
                    ui.strong(format!("Model Panel ({})", self.session.models.len()));
                    let mut toggle_vis: Option<(usize, bool)> = None;
                    let mut set_active: Option<usize> = None;
                    let mut focus_idx: Option<usize> = None;
                    let mut close_id: Option<u32> = None;
                    for (idx, m) in self.session.models.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let mut v = m.visible;
                            if ui.checkbox(&mut v, "").changed() {
                                toggle_vis = Some((idx, v));
                            }
                            let is_act = idx == self.session.active_model_idx;
                            if ui
                                .selectable_label(is_act, format!("#{} {}", m.model_id, m.name))
                                .clicked()
                            {
                                set_active = Some(idx);
                            }
                            if ui.small_button("◎").on_hover_text("Focus camera").clicked() {
                                focus_idx = Some(idx);
                            }
                            if ui.small_button("×").on_hover_text("Close model").clicked() {
                                close_id = Some(m.model_id);
                            }
                        });
                    }
                    if let Some((i, v)) = toggle_vis
                        && let Some(m) = self.session.models.get_mut(i)
                    {
                        m.visible = v;
                        self.session.dirty_geometry = true;
                        self.sync_viewport_scene();
                    }
                    if let Some(i) = set_active
                        && let Some(m) = self.session.models.get(i)
                    {
                        let mid = m.model_id;
                        self.run_dsl(&format!("active #{mid}"));
                    }
                    if let Some(i) = focus_idx
                        && let Some(m) = self.session.models.get(i)
                    {
                        let mol_clone = m.molecule.clone();
                        self.session.camera.focus_on_biomolecule(&mol_clone);
                    }
                    if let Some(mid) = close_id {
                        self.run_dsl(&format!("close #{mid}"));
                    }

                    ui.separator();
                    ui.strong("Representation (Active Model)");
                    for r in [
                        RepresentationKind::Cartoon,
                        RepresentationKind::Putty,
                        RepresentationKind::NucleicLadder,
                        RepresentationKind::Sticks,
                        RepresentationKind::Spheres,
                        RepresentationKind::Surface,
                    ] {
                        if ui
                            .selectable_value(
                                &mut self.session.render_opts.representation,
                                r,
                                r.label(),
                            )
                            .clicked()
                        {
                            self.session.dirty_geometry = true;
                            self.sync_viewport_scene();
                        }
                    }
                    ui.add_space(4.0);
                    ui.strong("Color By");
                    for c in [
                        ColorScheme::Chain,
                        ColorScheme::SecondaryStructure,
                        ColorScheme::Rainbow,
                        ColorScheme::Nucleotide,
                        ColorScheme::ElementCpk,
                        ColorScheme::CrystallographicBfactor,
                        ColorScheme::BfactorPlddt,
                        ColorScheme::Sap3D,
                        ColorScheme::ElectrostaticEsp,
                        ColorScheme::LipophilicityMlp,
                    ] {
                        if ui
                            .selectable_value(
                                &mut self.session.render_opts.color_scheme,
                                c,
                                c.label(),
                            )
                            .clicked()
                        {
                            self.session.render_opts.residue_colors.clear();
                            self.session.dirty_geometry = true;
                            self.sync_viewport_scene();
                        }
                    }
                    ui.add_space(4.0);
                    if ui
                        .checkbox(
                            &mut self.session.render_opts.show_surface_shell,
                            "Smooth SES Surface Mesh",
                        )
                        .changed()
                        || ui
                            .checkbox(
                                &mut self.session.render_opts.surface_wireframe,
                                "Wireframe Mesh (isomesh)",
                            )
                            .changed()
                        || ui
                            .checkbox(
                                &mut self.session.render_opts.show_rna_base_pairs,
                                "3D RNA Base-Pair Rungs",
                            )
                            .changed()
                        || ui
                            .checkbox(
                                &mut self.session.render_opts.show_silhouettes,
                                "ChimeraX Depth Silhouettes",
                            )
                            .changed()
                        || ui
                            .checkbox(
                                &mut self.session.render_opts.show_snfg_glycans,
                                "3D SNFG Glycans (NAG/MAN/FUC)",
                            )
                            .changed()
                        || ui
                            .checkbox(
                                &mut self.session.render_opts.show_unit_cell,
                                "Unit Cell Box (CRYST1)",
                            )
                            .changed()
                    {
                        self.session.dirty_geometry = true;
                        self.sync_viewport_scene();
                    }

                    ui.separator();
                    egui::CollapsingHeader::new("Side View (Clip · SSAO · Lighting)")
                        .default_open(true)
                        .show(ui, |ui| {
                            let mut changed = false;
                            ui.horizontal(|ui| {
                                ui.label("Light:");
                                egui::ComboBox::from_id_salt("lighting_preset_cb")
                                    .selected_text(self.session.render_opts.lighting.label())
                                    .show_ui(ui, |ui| {
                                        for lp in [
                                            LightingPreset::Soft,
                                            LightingPreset::PublicationWhite,
                                            LightingPreset::Studio,
                                            LightingPreset::Full,
                                            LightingPreset::Flat,
                                        ] {
                                            if ui
                                                .selectable_value(
                                                    &mut self.session.render_opts.lighting,
                                                    lp,
                                                    lp.label(),
                                                )
                                                .clicked()
                                            {
                                                changed = true;
                                            }
                                        }
                                    });
                            });
                            changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.session.render_opts.ssao_strength,
                                        0.0..=1.0,
                                    )
                                    .text("SSAO"),
                                )
                                .changed();
                            changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.session.render_opts.shadow_strength,
                                        0.0..=1.0,
                                    )
                                    .text("Shadows"),
                                )
                                .changed();
                            changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.session.render_opts.surface_alpha,
                                        0.2..=1.0,
                                    )
                                    .text("Surf α"),
                                )
                                .changed();
                            ui.checkbox(
                                &mut self.session.camera.clip_slab_enabled,
                                "Enable Near/Far Clip Slab",
                            );
                            if self.session.camera.clip_slab_enabled {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.session.camera.clip_near_a,
                                        1.0..=100.0,
                                    )
                                    .text("Front Å"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.session.camera.clip_far_a,
                                        1.0..=120.0,
                                    )
                                    .text("Back Å"),
                                );
                            }
                            if changed {
                                self.session.dirty_geometry = true;
                                self.sync_viewport_scene();
                            }
                        });

                    ui.separator();
                    egui::CollapsingHeader::new("Scenes (F1–F6 · Shift+Fn Save)")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for s in ["f1", "f2", "f3", "f4", "f5", "f6"] {
                                    let exists = self.session.scenes.contains_key(s);
                                    let lbl = if exists {
                                        format!("● {}", s.to_uppercase())
                                    } else {
                                        s.to_uppercase()
                                    };
                                    if ui.small_button(lbl).clicked() {
                                        if exists {
                                            self.run_dsl(&format!("scene recall {s}"));
                                        } else {
                                            self.run_dsl(&format!("scene save {s}"));
                                        }
                                    }
                                }
                            });
                        });

                    ui.separator();
                    if let Some(mol) = self.session.active_molecule() {
                        ui.strong(format!("Chains ({})", mol.chains.len()));
                        for ch in &mol.chains {
                            let badge = match ch.dominant_class {
                                ResidueClass::Protein => "[PROT]",
                                ResidueClass::Rna => "[RNA]",
                                ResidueClass::Dna => "[DNA]",
                                ResidueClass::Ligand => "[LIG]",
                                ResidueClass::Ion => "[ION]",
                                ResidueClass::Water => "[WAT]",
                            };
                            ui.small(format!(
                                "{badge} Chain {} — {} res",
                                ch.id,
                                ch.residues.len()
                            ));
                        }
                    }
                });
            });

        // Right Dockable Biomolecular Module Panel
        egui::SidePanel::right("biomol_modules_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (t, lbl) in [
                        (RightModuleTab::ViewDockX, "ViewDockX"),
                        (RightModuleTab::Measure, "Measure"),
                        (RightModuleTab::RamaAgg, "Rama / Agg"),
                        (RightModuleTab::MsaPae, "MSA / PAE"),
                        (RightModuleTab::MovieMorph, "Movie / Morph"),
                        (RightModuleTab::SymGlycans, "Sym / SNFG"),
                        (RightModuleTab::Modeling, "Model 3D"),
                        (RightModuleTab::Physics, "Physics"),
                        (RightModuleTab::CryoEm, "Cryo-EM"),
                        (RightModuleTab::Antibodies, "Ab / VHH"),
                        (RightModuleTab::NcRnaMrna, "ncRNA/mRNA"),
                        (RightModuleTab::Enzymes, "Enzymes"),
                        (RightModuleTab::AdcPk, "ADC / PK"),
                        (RightModuleTab::DeNovo, "De Novo"),
                        (RightModuleTab::SuitePlugins, "Suite/Plugins"),
                        (RightModuleTab::Evidence, "Evidence"),
                    ] {
                        ui.selectable_value(&mut self.right_tab, t, lbl);
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| match self.right_tab {
                    RightModuleTab::ViewDockX => self.ui_viewdockx_module(ui),
                    RightModuleTab::Measure => self.ui_measure_module(ui),
                    RightModuleTab::RamaAgg => self.ui_rama_agg_module(ui),
                    RightModuleTab::MsaPae => self.ui_msa_pae_module(ui),
                    RightModuleTab::MovieMorph => self.ui_movie_morph_module(ui),
                    RightModuleTab::SymGlycans => self.ui_sym_glycans_module(ui),
                    RightModuleTab::Modeling => self.ui_modeling_module(ui),
                    RightModuleTab::Physics => self.ui_physics_module(ui),
                    RightModuleTab::CryoEm => self.ui_cryoem_module(ui),
                    RightModuleTab::Antibodies => self.ui_antibody_module(ui),
                    RightModuleTab::NcRnaMrna => self.ui_ncrna_module(ui),
                    RightModuleTab::Enzymes => self.ui_enzyme_module(ui),
                    RightModuleTab::AdcPk => self.ui_adc_pk_module(ui),
                    RightModuleTab::DeNovo => self.ui_denovo_module(ui),
                    RightModuleTab::SuitePlugins => self.ui_suite_plugins_module(ui),
                    RightModuleTab::Evidence => self.ui_evidence_panel(ui),
                });
            });

        // Center Primary 3D GPU Biomolecular Viewport (PCF Soft Shadows + SSAO + Drag-to-Sculpt + HUD)
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let sculpt_drag_active = self.pick_mode == PickMode::SculptDrag
                    || ctx.input(|i| i.modifiers.shift)
                    || self.sculpt_drag_atom.is_some();
                let resp = show_biomol_viewport_ex(
                    ui,
                    &mut self.session.camera,
                    &self.viewport_scene,
                    self.session.render_opts.show_silhouettes,
                    true,
                    sculpt_drag_active,
                );
                // Paint live 3D/2D Labels, Color Key Legend & Calibrated Scale Bar directly over GPU viewport!
                self.paint_viewport_hud(ui, resp.rect);

                // Interactive Mouse 3D Drag-to-Sculpt (Shift + Left-Drag or SculptDrag mode)
                if sculpt_drag_active
                    && resp.drag_started_by(egui::PointerButton::Primary)
                    && let Some(pos) = resp.interact_pointer_pos()
                    && let Some(mol) = self.session.active_molecule()
                {
                    let rect = resp.rect;
                    let nx = ((pos.x - rect.left()) / rect.width().max(1.0)) * 2.0 - 1.0;
                    let ny = 1.0 - ((pos.y - rect.top()) / rect.height().max(1.0)) * 2.0;
                    let aspect = rect.width() / rect.height().max(1.0);
                    if let Some(ai) = self.session.camera.pick_atom(mol, nx, ny, aspect) {
                        let a = &mol.atoms[ai];
                        let lbl = format!("{}:{}{}.{}", a.chain, a.res_name, a.res_seq, a.name);
                        self.session.push_undo_snapshot();
                        self.sculpt_drag_atom = Some(ai);
                        self.picked_atom_info = format!("Dragging {lbl} (live MM sculpting)...");
                    }
                }
                if let Some(ai) = self.sculpt_drag_atom
                    && resp.dragged_by(egui::PointerButton::Primary)
                {
                    let d = resp.drag_delta();
                    if d.x * d.x + d.y * d.y > 0.25 {
                        let cam = self.session.camera.clone();
                        let h_px = resp.rect.height().max(1.0);
                        if let Some(mol) = self.session.active_molecule_mut()
                            && let Some(atom) = mol.atoms.get(ai)
                        {
                            let raw_disp =
                                cam.unproject_screen_delta_at_point(atom.coords, d.x, d.y, h_px);
                            let len = (raw_disp[0].powi(2)
                                + raw_disp[1].powi(2)
                                + raw_disp[2].powi(2))
                            .sqrt();
                            let disp = if len > 1.1 {
                                let k = 1.1 / len;
                                [raw_disp[0] * k, raw_disp[1] * k, raw_disp[2] * k]
                            } else {
                                raw_disp
                            };
                            if let Ok(rep) = tug_atom_with_sculpting(mol, ai, disp, 5.5, 22) {
                                let fb = format!(
                                    "Sculpt {} | {} atoms | MM E: {:.1}->{:.1} kcal/mol | ΔΔG_local: {:+.2}",
                                    rep.atom_label,
                                    rep.atoms_sculpted,
                                    rep.energy_before_kcal,
                                    rep.energy_after_kcal,
                                    rep.local_ddg_kcal
                                );
                                self.picked_atom_info = fb.clone();
                                self.sculpt_last_feedback = fb.clone();
                                self.status_line = fb;
                                self.session.dirty_geometry = true;
                                self.sync_viewport_scene();
                            }
                        }
                    }
                }
                if self.sculpt_drag_atom.is_some()
                    && (resp.drag_stopped_by(egui::PointerButton::Primary)
                        || !ctx.input(|i| i.pointer.primary_down()))
                {
                    self.sculpt_drag_atom = None;
                    self.console_log.push(self.sculpt_last_feedback.clone());
                }

                if resp.clicked()
                    && let Some(pos) = resp.interact_pointer_pos()
                    && let Some(mol) = self.session.active_molecule()
                {
                    let rect = resp.rect;
                    let nx = ((pos.x - rect.left()) / rect.width().max(1.0)) * 2.0 - 1.0;
                    let ny = 1.0 - ((pos.y - rect.top()) / rect.height().max(1.0)) * 2.0;
                    let aspect = rect.width() / rect.height().max(1.0);
                    if let Some(ai) = self.session.camera.pick_atom(mol, nx, ny, aspect) {
                        let a = &mol.atoms[ai];
                        self.mut_chain = a.chain.to_string();
                        self.mut_res_seq = a.res_seq;
                        self.tug_atom_spec = format!("{}:{}:{}", a.chain, a.res_seq, a.name);
                        self.picked_atom_info = format!(
                            "Picked {}:{}{}.{} ({}, B={:.1})",
                            a.chain, a.res_name, a.res_seq, a.name, a.element, a.b_factor
                        );
                        match self.pick_mode {
                            PickMode::Inspect | PickMode::SculptDrag => {}
                            PickMode::Distance => {
                                self.pick_buffer.push(ai);
                                if self.pick_buffer.len() >= 2 {
                                    let (p0, p1) = (self.pick_buffer[0], self.pick_buffer[1]);
                                    self.pick_buffer.clear();
                                    if let Ok(msg) = self.session.add_distance_measurement(p0, p1) {
                                        self.console_log.push(msg.clone());
                                        self.status_line = msg;
                                        self.sync_viewport_scene();
                                    }
                                }
                            }
                            PickMode::Angle => {
                                self.pick_buffer.push(ai);
                                if self.pick_buffer.len() >= 3 {
                                    let (p0, p1, p2) = (
                                        self.pick_buffer[0],
                                        self.pick_buffer[1],
                                        self.pick_buffer[2],
                                    );
                                    self.pick_buffer.clear();
                                    if let Ok(msg) = self.session.add_angle_measurement(p0, p1, p2)
                                    {
                                        self.console_log.push(msg.clone());
                                        self.status_line = msg;
                                        self.sync_viewport_scene();
                                    }
                                }
                            }
                            PickMode::Dihedral => {
                                self.pick_buffer.push(ai);
                                if self.pick_buffer.len() >= 4 {
                                    let (p0, p1, p2, p3) = (
                                        self.pick_buffer[0],
                                        self.pick_buffer[1],
                                        self.pick_buffer[2],
                                        self.pick_buffer[3],
                                    );
                                    self.pick_buffer.clear();
                                    if let Ok(msg) =
                                        self.session.add_dihedral_measurement(p0, p1, p2, p3)
                                    {
                                        self.console_log.push(msg.clone());
                                        self.status_line = msg;
                                        self.sync_viewport_scene();
                                    }
                                }
                            }
                        }
                    }
                }
            });
    }
}

impl BiomolCraftApp {
    fn ui_viewdockx_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("ViewDockX · DockCraft Craft Link v1");
        ui.horizontal(|ui| {
            if ui.button("Fit GridBox to Selection").clicked() {
                self.run_dsl("gridbox from_selection 5.0");
            }
            if ui.button("Dock Here in DockCraft").clicked() {
                self.dock_here_in_dockcraft();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Ligand SDF/SMI:");
            ui.add(
                egui::TextEdit::singleline(&mut self.dock_ligand_input)
                    .desired_width(210.0)
                    .hint_text("optional path (empty = open DockCraft GUI)"),
            );
        });

        ui.separator();
        ui.strong("3D Binding Pocket Detector (In-House 14-Ray Burial)");
        ui.horizontal(|ui| {
            if ui.button("Detect 3D Pockets").clicked() {
                self.run_dsl("pockets 6");
            }
            if !self.session.binding_pockets.is_empty() && ui.small_button("Hide Pockets").clicked()
            {
                self.run_dsl("pockets off");
            }
        });
        if !self.session.binding_pockets.is_empty() {
            let mut pick_pocket: Option<usize> = None;
            let mut send_pocket_to_dock: Option<usize> = None;
            for p in &self.session.binding_pockets {
                let is_sel = self.session.selected_pocket_id == Some(p.pocket_id);
                ui.horizontal(|ui| {
                    let lbl = format!(
                        "#{}: Drug {:.2} · {:.0} Å³ · Enc {:.0}%",
                        p.pocket_id,
                        p.druggability_score,
                        p.volume_a3,
                        p.enclosure * 100.0
                    );
                    if ui.selectable_label(is_sel, lbl).clicked() {
                        pick_pocket = Some(p.pocket_id);
                    }
                    if ui
                        .small_button("→ DockCraft")
                        .on_hover_text("Snap GridBox to this pocket and launch/sync DockCraft")
                        .clicked()
                    {
                        send_pocket_to_dock = Some(p.pocket_id);
                    }
                });
            }
            if let Some(pid) = pick_pocket {
                self.run_dsl(&format!("pocket_box {pid}"));
            }
            if let Some(pid) = send_pocket_to_dock {
                self.run_dsl(&format!("pocket_box {pid}"));
                self.dock_here_in_dockcraft();
            }
        }

        let mut dirty = false;
        let mut select_pose: Option<usize> = None;
        let mut cycle_tag: Option<usize> = None;

        if let Some(ref mut vd) = self.session.viewdock {
            ui.separator();
            ui.strong(format!(
                "{} — {} pose(s) (↑/↓ keys scrub poses)",
                vd.bundle.title,
                vd.bundle.poses.len()
            ));
            ui.horizontal_wrapped(|ui| {
                dirty |= ui
                    .checkbox(&mut vd.show_hyde_coronas, "HYDE Coronas")
                    .changed();
                dirty |= ui
                    .checkbox(&mut vd.show_torsion_lights, "Torsion Lights")
                    .changed();
                dirty |= ui
                    .checkbox(&mut vd.show_interactions, "Interactions")
                    .changed();
                dirty |= ui
                    .checkbox(&mut vd.show_pharmacophores, "Pharmacophores")
                    .changed();
                dirty |= ui
                    .checkbox(&mut vd.show_dmif_hotspots, "ΔMIF Selectivity")
                    .changed();
                dirty |= ui
                    .checkbox(&mut vd.show_bridging_waters, "Bridging Waters")
                    .changed();
                dirty |= ui.checkbox(&mut vd.show_grid_box, "3D GridBox").changed();
            });

            if let Some(ref mut gb) = vd.editable_grid_box {
                egui::CollapsingHeader::new("Editable 3D GridBox (Synced → DockCraft)")
                    .default_open(false)
                    .show(ui, |ui| {
                        for (ax, lbl) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(format!("C{lbl}"));
                                dirty |= ui
                                    .add(egui::DragValue::new(&mut gb.center[ax]).speed(0.25))
                                    .changed();
                                ui.label(format!("S{lbl}"));
                                dirty |= ui
                                    .add(
                                        egui::DragValue::new(&mut gb.size[ax])
                                            .speed(0.25)
                                            .range(10.0..=64.0),
                                    )
                                    .changed();
                            });
                        }
                    });
            }

            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("viewdock_poses_scroll")
                .max_height(260.0)
                .show(ui, |ui| {
                    for (idx, p) in vd.bundle.poses.iter().enumerate() {
                        let is_sel = idx == vd.active_pose_idx;
                        let tag = vd
                            .pose_tags
                            .get(idx)
                            .copied()
                            .unwrap_or(PoseStatusTag::Unreviewed);
                        ui.horizontal(|ui| {
                            if ui.small_button(tag.short_label()).clicked() {
                                cycle_tag = Some(idx);
                            }
                            let hyde_s = p
                                .hyde_kcal
                                .map_or_else(|| "—".to_string(), |v| format!("{v:.2}"));
                            let pb_s = match p.posebusters_valid {
                                Some(true) => "PB✓",
                                Some(false) => "PB!",
                                None => "",
                            };
                            let lbl = format!(
                                "#{} {} | V:{:.2} H:{} {}",
                                p.pose_id, p.name, p.vina_kcal, hyde_s, pb_s
                            );
                            if ui.selectable_label(is_sel, lbl).clicked() {
                                select_pose = Some(idx);
                            }
                        });
                    }
                });

            if let Some(active) = vd.bundle.poses.get(vd.active_pose_idx) {
                ui.separator();
                ui.strong(format!("Active Pose: {}", active.name));
                if let Some(ki) = active.estimated_ki_nm {
                    ui.small(format!(
                        "Vina {:.2} | HYDE {:?} | Est. Ki {:.1} nM | LE {:.3}",
                        active.vina_kcal,
                        active.hyde_kcal,
                        ki,
                        active.ligand_efficiency.unwrap_or(0.0)
                    ));
                }
                if let Some(tc) = active.torsion_counts {
                    ui.small(format!(
                        "Torsion Lights: {} Green · {} Yellow · {} Red | Interactions: {}",
                        tc[0],
                        tc[1],
                        tc[2],
                        active.interaction_lines.len()
                    ));
                }
                if let Some(ref pbs) = active.posebusters_summary {
                    ui.small(format!("PoseBusters: {pbs}"));
                }
            }
        } else {
            ui.separator();
            ui.small(
                "No ViewDockX bundle loaded yet.\n\
                 • From DockCraft GUI: click 'BiomolCraft' to stream poses, HYDE coronas, torsion lights, pharmacophores & ΔMIF live over Craft Link v1.\n\
                 • From CLI: `dockcraft-cli dock ... --out-dock-bundle out.json` then `load_dock out.json`.\n\
                 • Or select pocket residues here and click 'Box ← BiomolCraft' inside DockCraft!",
            );
            if let Some(ref mut gb) = self.session.render_opts.grid_box {
                ui.separator();
                ui.strong("Custom 3D GridBox (Active → DockCraft)");
                for (ax, lbl) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("C{lbl}"));
                        dirty |= ui
                            .add(egui::DragValue::new(&mut gb.center[ax]).speed(0.25))
                            .changed();
                        ui.label(format!("S{lbl}"));
                        dirty |= ui
                            .add(
                                egui::DragValue::new(&mut gb.size[ax])
                                    .speed(0.25)
                                    .range(10.0..=64.0),
                            )
                            .changed();
                    });
                }
            } else if ui
                .button("+ Create Default GridBox at Camera Target")
                .clicked()
            {
                let t = self.session.camera.target;
                self.session.render_opts.grid_box = Some(GridBox3D {
                    center: [f64::from(t[0]), f64::from(t[1]), f64::from(t[2])],
                    size: [20.0, 20.0, 20.0],
                    spacing: 0.375,
                });
                dirty = true;
            }
        }

        if let Some(idx) = cycle_tag
            && let Some(ref mut vd) = self.session.viewdock
            && let Some(tag) = vd.pose_tags.get_mut(idx)
        {
            *tag = tag.cycle();
        }
        if let Some(idx) = select_pose {
            let _ = self.session.select_dock_pose(idx);
            dirty = true;
        }
        if dirty {
            self.session.dirty_geometry = true;
            self.sync_viewport_scene();
        }
    }

    fn ui_modeling_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Interactive 3D Mutagenesis & MM Sculpting");
        ui.horizontal(|ui| {
            ui.label("Chain:");
            ui.add(egui::TextEdit::singleline(&mut self.mut_chain).desired_width(36.0));
            ui.label("Res:");
            ui.add(egui::DragValue::new(&mut self.mut_res_seq));
        });
        ui.horizontal(|ui| {
            ui.label("Mutate to:");
            egui::ComboBox::from_id_salt("mut_aa_cb")
                .selected_text(self.mut_target_aa.to_string())
                .show_ui(ui, |ui| {
                    for aa in "ACDEFGHIKLMNPQRSTVWY".chars() {
                        ui.selectable_value(&mut self.mut_target_aa, aa, aa.to_string());
                    }
                });
            ui.label("Rotamer:");
            ui.add(egui::DragValue::new(&mut self.mut_rotamer).range(0..=6));
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Apply Mutation (Auto Repack)").clicked() {
                let cmd = format!(
                    "mutate {}:{} {}",
                    self.mut_chain, self.mut_res_seq, self.mut_target_aa
                );
                self.run_dsl(&cmd);
            }
            if ui.button("Use Selected Rotamer").clicked() {
                let cmd = format!(
                    "mutate {}:{} {} {}",
                    self.mut_chain, self.mut_res_seq, self.mut_target_aa, self.mut_rotamer
                );
                self.run_dsl(&cmd);
            }
            if ui.button("List Dunbrack Rotamers (χ1/χ2)").clicked() {
                let cmd = format!(
                    "rotamers {}:{} {}",
                    self.mut_chain, self.mut_res_seq, self.mut_target_aa
                );
                self.run_dsl(&cmd);
            }
            if ui.button("Clone WT Copy").clicked() {
                self.run_dsl("clone wildtype");
            }
        });
        let mut click_rotamer: Option<(char, i32, char, usize)> = None;
        if let Some(ref rt) = self.session.rotamer_table {
            ui.separator();
            ui.strong(format!(
                "Dunbrack Rotamers {}{}{} ({}, {} rot — click to apply):",
                rt.wild_type,
                rt.res_seq,
                rt.mutant,
                rt.backbone_bin.label(),
                rt.candidates.len()
            ));
            for c in &rt.candidates {
                let c2 = c
                    .chi2_deg
                    .map_or_else(|| "—".to_string(), |v| format!("{v:.0}°"));
                let star = if c.rotamer_index == rt.best_rotamer_index {
                    "★ "
                } else {
                    ""
                };
                let lbl = format!(
                    "{star}#{} ({:.0}%) χ1={:.0}° χ2={c2} | Lig:{} Prot:{} (ov {:.2}Å)",
                    c.rotamer_index + 1,
                    c.probability_pct,
                    c.chi1_deg,
                    c.clashes_ligand,
                    c.clashes_protein,
                    c.max_overlap_a
                );
                if ui.small_button(lbl).clicked() {
                    click_rotamer = Some((rt.chain, rt.res_seq, rt.mutant, c.rotamer_index));
                }
            }
        }
        if let Some((ch, rseq, aa, idx)) = click_rotamer {
            self.mut_rotamer = idx;
            self.run_dsl(&format!("mutate {ch}:{rseq} {aa} {idx}"));
        }
        ui.horizontal(|ui| {
            if ui.button("Minimize Local MM (80 steps)").clicked() {
                self.run_dsl("minimize 80");
            }
            if ui.button("AddH (pH 7.4)").clicked() {
                self.run_dsl("addh 7.4");
            }
            if ui.button("Undo").clicked() {
                self.run_dsl("undo");
            }
            if ui.button("Redo").clicked() {
                self.run_dsl("redo");
            }
        });

        ui.separator();
        ui.strong("3D RNA/DNA Mutagenesis (swapna)");
        ui.horizontal(|ui| {
            ui.label(format!("Target {}:{}:", self.mut_chain, self.mut_res_seq));
            for b in ['A', 'U', 'G', 'C', 'T'] {
                ui.selectable_value(&mut self.swapna_base, b, b.to_string());
            }
            if ui.small_button("Apply swapna").clicked() {
                let cmd = format!(
                    "swapna {}:{} {}",
                    self.mut_chain, self.mut_res_seq, self.swapna_base
                );
                self.run_dsl(&cmd);
            }
        });

        ui.separator();
        ui.strong("Bond Torsion Rotation (bond_rot) & Tug Sculpting");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.bond_rot_atom_a).desired_width(82.0));
            ui.label("—");
            ui.add(egui::TextEdit::singleline(&mut self.bond_rot_atom_b).desired_width(82.0));
            ui.add(
                egui::DragValue::new(&mut self.bond_rot_deg)
                    .range(-180.0..=180.0)
                    .suffix("°"),
            );
            if ui.small_button("Rotate").clicked() {
                let cmd = format!(
                    "bond_rot {} {} {:.1}",
                    self.bond_rot_atom_a, self.bond_rot_atom_b, self.bond_rot_deg
                );
                self.run_dsl(&cmd);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Tug:");
            ui.add(egui::TextEdit::singleline(&mut self.tug_atom_spec).desired_width(82.0));
            for (k, lbl) in ["dX", "dY", "dZ"].iter().enumerate() {
                ui.add(
                    egui::DragValue::new(&mut self.tug_shift[k])
                        .speed(0.1)
                        .range(-5.0..=5.0)
                        .prefix(*lbl),
                );
            }
            if ui.small_button("Pull").clicked() {
                let cmd = format!(
                    "tug {} {:.2} {:.2} {:.2}",
                    self.tug_atom_spec, self.tug_shift[0], self.tug_shift[1], self.tug_shift[2]
                );
                self.run_dsl(&cmd);
            }
        });
        ui.horizontal(|ui| {
            let is_sculpt = self.pick_mode == PickMode::SculptDrag;
            if ui
                .selectable_label(is_sculpt, "Mouse Drag-to-Sculpt (or hold Shift+Drag)")
                .clicked()
            {
                self.pick_mode = if is_sculpt {
                    PickMode::Inspect
                } else {
                    PickMode::SculptDrag
                };
            }
        });
        ui.small(&self.sculpt_last_feedback);

        ui.separator();
        ui.strong("3D Structure Builder (Peptide Helix/Sheet · A-RNA Duplex)");
        ui.horizontal(|ui| {
            for k in ["helix", "sheet", "extended", "rna"] {
                ui.selectable_value(&mut self.build_kind, k.to_string(), k);
            }
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.build_seq)
                    .desired_width(210.0)
                    .hint_text("AA or RNA sequence..."),
            );
            if ui.small_button("Build 3D").clicked() {
                let cmd = format!("build {} {}", self.build_kind, self.build_seq);
                self.run_dsl(&cmd);
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Altlocs Audit").clicked() {
                self.run_dsl("altlocs");
            }
            if ui.small_button("Combine Visible Models").clicked() {
                self.run_dsl("combine");
            }
            if ui.small_button("Delete Waters").clicked() {
                self.run_dsl("delete water");
            }
        });
    }

    fn ui_physics_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Hybrid Biomolecular Physics (ANM · GB/SA · MD)");
        ui.horizontal(|ui| {
            ui.label("ANM Mode:");
            ui.add(egui::DragValue::new(&mut self.anm_mode_sel).range(1..=6));
            ui.label("Amp (Å):");
            ui.add(
                egui::DragValue::new(&mut self.anm_amp_a)
                    .range(0.5..=6.0)
                    .speed(0.2),
            );
        });
        ui.horizontal(|ui| {
            if ui.button("Compute & Animate ANM Mode").clicked() {
                let cmd = format!("anm {} {:.1}", self.anm_mode_sel, self.anm_amp_a);
                self.run_dsl(&cmd);
                self.playing_trajectory = true;
            }
            if ui.button("GB/SA + Dipole + ΔG_bind").clicked() {
                self.run_dsl("gbsa");
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Run Pure-Rust Langevin MD (120 steps)").clicked() {
                self.run_dsl("md 120");
                self.playing_trajectory = true;
            }
        });
        ui.separator();
        ui.strong("GROMACS `.xtc` & CHARMM/OpenMM `.dcd` Trajectory Loader");
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.xtc_path_input)
                    .desired_width(205.0)
                    .hint_text("/path/to/traj.xtc or traj.dcd"),
            );
            if ui.small_button("Load Traj").clicked() {
                let p = self.xtc_path_input.trim().to_string();
                if !p.is_empty() {
                    let verb = if p.to_ascii_lowercase().ends_with(".xtc") {
                        "load_xtc"
                    } else {
                        "load_dcd"
                    };
                    self.run_dsl(&format!("{verb} {p}"));
                    self.playing_trajectory = true;
                }
            }
        });
        ui.separator();
        ui.strong(format!(
            "Tier-2 OpenMM GPU Runtime: {} ({})",
            if self.openmm_status.available {
                self.openmm_status
                    .openmm_version
                    .as_deref()
                    .unwrap_or("Detected")
            } else {
                "Not Installed"
            },
            if self.openmm_status.cuda_ready {
                "CUDA RTX Ready"
            } else {
                "CPU Fallback"
            }
        ));
        ui.small(
            "Run full explicit/GBn2 GPU MD via `biomolcraft-cli md --engine openmm --pdb <file> --platform CUDA` and load the `.dcd` via `load_dcd` or Craft Link.",
        );
        ui.separator();
        ui.monospace(&self.session.last_module_markdown);
    }

    fn ui_antibody_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Antibodies & VHH Module (IMGT · TAP · Ala-Scan)");
        if ui
            .button("Analyze & Paint CDRs / Ala-Scan Hotspots in 3D")
            .clicked()
            && let Some(mol) = self.session.active_molecule().cloned()
        {
            match run_antibody_module(&mol, None, None) {
                Ok(b) => {
                    self.session.render_opts.residue_highlights = b.visual.residue_highlights;
                    self.session.render_opts.sphere_overlays = b.visual.sphere_overlays;
                    self.session.render_opts.line_overlays = b.visual.line_overlays;
                    self.session.last_module_markdown = b.visual.summary_markdown;
                    self.session.provenance.extend(b.visual.provenance);
                    self.session.dirty_geometry = true;
                    self.sync_viewport_scene();
                }
                Err(e) => self.status_line = e,
            }
        }
        ui.separator();
        ui.monospace(&self.session.last_module_markdown);
    }

    fn ui_ncrna_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("ncRNA, mRNA, siRNA & Riboswitch Module");
        ui.horizontal_wrapped(|ui| {
            for p in rna_presets() {
                if ui.small_button(p.id).clicked() {
                    self.rna_name = p.name.to_string();
                    self.rna_seq = p.sequence.to_string();
                    if let Ok(rep) = analyze_ncrna_mrna(p.name, p.class, p.sequence, None) {
                        self.rna_report = Some(rep);
                    }
                }
            }
        });
        ui.add(
            egui::TextEdit::multiline(&mut self.rna_seq)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .code_editor(),
        );
        ui.horizontal(|ui| {
            if ui.button("Fold 2D (Zuker MFE)").clicked()
                && let Ok(rep) =
                    analyze_ncrna_mrna(&self.rna_name, "ncRNA/mRNA", &self.rna_seq, None)
            {
                self.rna_report = Some(rep);
            }
            if ui.button("Sync Active 3D RNA Chains").clicked()
                && let Some(mol) = self.session.active_molecule().cloned()
                && let Ok((vis, mut reps)) = run_ncrna_module(&mol)
            {
                self.session.render_opts.representation = RepresentationKind::NucleicLadder;
                self.session.render_opts.color_scheme = ColorScheme::Nucleotide;
                self.session.render_opts.residue_highlights = vis.residue_highlights;
                self.session.last_module_markdown = vis.summary_markdown;
                self.rna_report = reps.pop();
                self.session.dirty_geometry = true;
                self.sync_viewport_scene();
            }
        });
        if let Some(ref r) = self.rna_report {
            ui.separator();
            ui.strong(format!(
                "MFE: {:.2} kcal/mol | {} pairs ({:.1}%) | {} hairpins | GC {:.1}% | U {:.1}%",
                r.fold.mfe_kcal_mol,
                r.fold.pairs.len(),
                r.fold.paired_fraction_pct,
                r.fold.hairpin_count,
                r.fold.gc_percent,
                r.fold.uridine_percent
            ));
            ui.label(format!(
                "5'-30nt MFE: {:.2} | G4: {} | TLR7/8: {} | Kozak: {}",
                r.first_30nt_mfe_kcal_mol,
                r.g_quadruplex_motifs.len(),
                r.tlr7_8_motifs_count,
                r.has_kozak_aug
            ));
            ui.monospace(&r.fold.dot_bracket);
            ui.small(&r.m1psi_recommendation);
            ui.small(format!(
                "LNP: {} (N/P={:.1}, RNA {:.2} kDa)",
                r.lnp_spec.molar_ratio_ion_dspc_chol_peg,
                r.lnp_spec.np_charge_ratio,
                r.lnp_spec.rna_mw_kda
            ));
        }
    }

    fn ui_enzyme_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Enzymes, Metals & Catalytic Triads");
        if ui
            .button("Detect Active Site / Metal Coordination in 3D")
            .clicked()
            && let Some(mol) = self.session.active_molecule().cloned()
        {
            let vis = run_enzyme_module(&mol);
            self.session.render_opts.residue_highlights = vis.residue_highlights;
            self.session.render_opts.sphere_overlays = vis.sphere_overlays;
            self.session.render_opts.line_overlays = vis.line_overlays;
            self.session.last_module_markdown = vis.summary_markdown;
            self.session.dirty_geometry = true;
            self.sync_viewport_scene();
        }
        ui.separator();
        ui.monospace(&self.session.last_module_markdown);
    }

    fn ui_adc_pk_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("ADCs, 3D Conjugation Sites & Biologics PK/TMDD");
        ui.horizontal(|ui| {
            ui.label("Linker:");
            ui.add(egui::TextEdit::singleline(&mut self.adc_linker).desired_width(60.0));
            ui.label("Payload:");
            ui.add(egui::TextEdit::singleline(&mut self.adc_payload).desired_width(60.0));
            ui.label("DAR:");
            ui.add(
                egui::DragValue::new(&mut self.adc_dar)
                    .range(1.0..=10.0)
                    .speed(0.5),
            );
        });
        ui.horizontal(|ui| {
            if ui.button("Map 3D ADC Sites + AdmetCraft").clicked()
                && let Some(mol) = self.session.active_molecule().cloned()
                && let Ok((vis, _)) = run_adc_module(
                    &mol,
                    &self.adc_linker,
                    &self.adc_payload,
                    self.adc_dar,
                    true,
                )
            {
                self.session.render_opts.residue_highlights = vis.residue_highlights;
                self.session.render_opts.sphere_overlays = vis.sphere_overlays;
                self.session.last_module_markdown = vis.summary_markdown;
                self.session.dirty_geometry = true;
                self.sync_viewport_scene();
            }
            if ui.button("Simulate IgG1 PK/TMDD").clicked()
                && let Ok((_, md)) = run_pk_tmdd_module(
                    "Biomol_IgG1",
                    biomol_modules::BiologicsFormat::Igg1Wt,
                    420.0,
                    21.0,
                )
            {
                self.session.last_module_markdown = md;
            }
        });
        ui.separator();
        ui.monospace(&self.session.last_module_markdown);
    }

    fn ui_cryoem_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Cryo-EM MRC2014 & OpenDX Volumetric Maps");
        ui.horizontal(|ui| {
            ui.label("MolMap Res (Å):");
            ui.add(
                egui::DragValue::new(&mut self.cryoem_res_a)
                    .range(2.0..=12.0)
                    .speed(0.25),
            );
            ui.label("Contour (σ):");
            ui.add(
                egui::DragValue::new(&mut self.cryoem_sigma)
                    .range(0.5..=5.0)
                    .speed(0.1),
            );
        });
        if ui
            .checkbox(
                &mut self.session.volume_wireframe,
                "Wireframe Isomesh Density",
            )
            .changed()
        {
            self.session.dirty_geometry = true;
            self.sync_viewport_scene();
        }
        ui.horizontal(|ui| {
            if ui.button("Synthesize MolMap + Isosurface Mesh").clicked() {
                let cmd = format!("molmap {:.1} {:.1}", self.cryoem_res_a, self.cryoem_sigma);
                self.run_dsl(&cmd);
            }
            if ui.button("Rigid Fit-in-Map (fitmap)").clicked() {
                self.run_dsl("fitmap");
            }
        });
        ui.separator();
        ui.strong("Volumetric Operations (vop & Orthoplanes)");
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("vop Gaussian (1.8 Å)").clicked() {
                self.run_dsl("vop gaussian 1.8");
            }
            if ui.small_button("vop Sharpen").clicked() {
                self.run_dsl("vop sharpen 1.2");
            }
            if ui.small_button("Hide Dust").clicked() {
                self.run_dsl("vop hide_dust 12");
            }
        });
        ui.horizontal(|ui| {
            ui.label("Zone Radius (Å):");
            ui.add(
                egui::DragValue::new(&mut self.vop_zone_rad)
                    .range(2.0..=20.0)
                    .speed(0.5),
            );
            if ui.small_button("Carve Volume Zone").clicked() {
                let r = self.vop_zone_rad;
                self.run_dsl(&format!("vop zone {r:.1}"));
            }
        });
        let mut ortho_dirty = ui
            .checkbox(
                &mut self.session.show_orthoplanes,
                "Show XY / XZ / YZ Orthoplanes",
            )
            .changed();
        if self.session.show_orthoplanes {
            for (k, lbl) in ["X Slice", "Y Slice", "Z Slice"].iter().enumerate() {
                ortho_dirty |= ui
                    .add(
                        egui::Slider::new(&mut self.session.orthoplane_frac[k], 0.05..=0.95)
                            .text(*lbl),
                    )
                    .changed();
            }
        }
        if ortho_dirty {
            self.session.dirty_geometry = true;
            self.sync_viewport_scene();
        }
        ui.small(
            "Load external `.mrc`/`.map`/`.dx` volumes via DSL: `open_map /path/to/map.mrc 1.5`",
        );
        ui.separator();
        ui.monospace(&self.session.last_module_markdown);
    }

    fn ui_denovo_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("De Novo & Co-Fold Epitope Staging (0 GB Weights)");
        ui.horizontal(|ui| {
            ui.label("Target Chain:");
            ui.add(egui::TextEdit::singleline(&mut self.denovo_chain).desired_width(36.0));
            if ui.button("+ Add Picked Residue").clicked() {
                let tag = format!("{}{}", self.mut_chain, self.mut_res_seq);
                if !self.denovo_hotspots.contains(&tag) {
                    if !self.denovo_hotspots.is_empty() {
                        self.denovo_hotspots.push(' ');
                    }
                    self.denovo_hotspots.push_str(&tag);
                }
            }
        });
        ui.label("3D Epitope Hotspots:");
        ui.add(egui::TextEdit::singleline(&mut self.denovo_hotspots).desired_width(f32::INFINITY));
        if ui
            .button("Paint Hotspots in 3D & Stage RFdiffusion/MPNN/Boltz-2")
            .clicked()
        {
            let cmd = format!("denovo {} {}", self.denovo_chain, self.denovo_hotspots);
            self.run_dsl(&cmd);
        }
        ui.separator();
        ui.monospace(&self.session.last_module_markdown);
    }

    fn ui_suite_plugins_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Craft Link v1 · Portable Sessions · Live ArtCraft Push · Plugins");
        ui.strong("Portable Verified Session (.biomol.zip — PyMOL .pse / ChimeraX .cxs):");
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.session_zip_path)
                    .desired_width(210.0)
                    .hint_text("/tmp/session.biomol.zip"),
            );
            if ui.small_button("Save").clicked() {
                let p = self.session_zip_path.clone();
                self.run_dsl(&format!("save_session {p}"));
            }
            if ui.small_button("Load").clicked() {
                let p = self.session_zip_path.clone();
                self.run_dsl(&format!("load_session {p}"));
            }
        });
        ui.separator();
        ui.strong("Live Push to ArtCraft Suite (Craft Link v1 + Full Bundle):");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Push Figure → WordCraft").clicked() {
                self.run_dsl("push_artcraft wordcraft /tmp/biomolcraft_push_wordcraft");
            }
            if ui.button("Push Slide → DeckCraft").clicked() {
                self.run_dsl("push_artcraft deckcraft /tmp/biomolcraft_push_deckcraft");
            }
            if ui.button("Push Table → GridCraft").clicked() {
                self.run_dsl("push_artcraft gridcraft /tmp/biomolcraft_push_gridcraft");
            }
        });
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Scan Craft Link v1 Peers").clicked() {
                match discover_craft_instances(None, None) {
                    Ok(peers) => {
                        if peers.is_empty() {
                            self.live_peers_summary = "No live peers detected.".to_string();
                        } else {
                            let list: Vec<String> = peers
                                .iter()
                                .map(|p| format!("{} v{} (pid {})", p.app, p.app_version, p.pid))
                                .collect();
                            self.live_peers_summary = format!("Live peers: {}", list.join(", "));
                        }
                    }
                    Err(e) => self.live_peers_summary = format!("Scan error: {e}"),
                }
            }
            if ui.button("Export Offline Mol* HTML").clicked() {
                self.run_dsl("export_html /tmp/biomolcraft_dossier.html");
            }
        });
        if ui
            .button("Export ArtCraft Bundle (WordCraft / DeckCraft / GridCraft)")
            .clicked()
        {
            self.run_dsl("export_artcraft /tmp/biomolcraft_artcraft");
        }
        ui.small(&self.live_peers_summary);
        ui.separator();
        ui.strong("Installed Plugins ($CRAFT_DATA_DIR/plugins):");
        let plugins = discover_installed_plugins();
        if plugins.is_empty() {
            ui.small("No external .json or .wasm plugins installed yet.");
        } else {
            for p in plugins {
                ui.label(format!("• [{}] {}: {}", p.kind, p.file_name, p.summary));
            }
        }
    }

    fn ui_evidence_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Scientific Provenance & Evidence Log");
        for ev in self.session.provenance.iter().rev() {
            ui.label(format!(
                "• [{:?}] ({}) {}",
                ev.kind, ev.source_app, ev.summary
            ));
        }
    }

    fn paint_viewport_hud(&self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter_at(rect);
        let w = rect.width().max(1.0) as u32;
        let h = rect.height().max(1.0) as u32;
        let pill_bg = egui::Color32::from_rgba_unmultiplied(14, 18, 28, 212);

        // 1. Projected 3D Labels
        for lbl in &self.session.labels_3d {
            let p = [
                lbl.coords[0] as f32,
                lbl.coords[1] as f32,
                lbl.coords[2] as f32,
            ];
            if let Some((sx, sy, _)) = project_world_to_screen(&self.session.camera, p, w, h) {
                let pos = egui::pos2(rect.left() + sx + 6.0, rect.top() + sy - 14.0);
                let col = egui::Color32::from_rgb(
                    (lbl.rgba[0].clamp(0.0, 1.0) * 255.0) as u8,
                    (lbl.rgba[1].clamp(0.0, 1.0) * 255.0) as u8,
                    (lbl.rgba[2].clamp(0.0, 1.0) * 255.0) as u8,
                );
                let galley = painter.layout_no_wrap(
                    lbl.text.clone(),
                    egui::FontId::monospace(12.5 * lbl.font_scale.clamp(0.7, 2.0)),
                    col,
                );
                let bg_rect = egui::Rect::from_min_size(
                    pos - egui::vec2(5.0, 2.0),
                    galley.size() + egui::vec2(10.0, 4.0),
                );
                painter.rect_filled(bg_rect, 4.0, pill_bg);
                painter.galley(pos, galley, col);
            }
        }

        // 2. 2D Screen-Space Labels
        for lbl in &self.session.labels_2d {
            let pos = egui::pos2(
                rect.left() + lbl.norm_pos[0].clamp(0.02, 0.92) * rect.width(),
                rect.top() + lbl.norm_pos[1].clamp(0.02, 0.92) * rect.height(),
            );
            let col = egui::Color32::from_rgb(
                (lbl.rgba[0].clamp(0.0, 1.0) * 255.0) as u8,
                (lbl.rgba[1].clamp(0.0, 1.0) * 255.0) as u8,
                (lbl.rgba[2].clamp(0.0, 1.0) * 255.0) as u8,
            );
            let galley = painter.layout_no_wrap(
                lbl.text.clone(),
                egui::FontId::proportional(14.0 * lbl.font_scale.clamp(0.7, 2.5)),
                col,
            );
            let bg_rect = egui::Rect::from_min_size(
                pos - egui::vec2(6.0, 3.0),
                galley.size() + egui::vec2(12.0, 6.0),
            );
            painter.rect_filled(bg_rect, 4.0, pill_bg);
            painter.galley(pos, galley, col);
        }

        // 3. Publication Color Key Legend (bottom-left)
        if let Some(ref key) = self.session.color_key
            && key.stops.len() >= 2
        {
            let bar_w = (rect.width() * 0.24).clamp(150.0, 280.0);
            let box_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + 14.0, rect.bottom() - 64.0),
                egui::vec2(bar_w + 20.0, 50.0),
            );
            painter.rect_filled(box_rect, 5.0, pill_bg);
            painter.text(
                box_rect.left_top() + egui::vec2(10.0, 6.0),
                egui::Align2::LEFT_TOP,
                &key.title,
                egui::FontId::proportional(11.5),
                egui::Color32::WHITE,
            );
            let segs = 32usize;
            let bar_left = box_rect.left() + 10.0;
            let bar_top = box_rect.top() + 22.0;
            let step_w = bar_w / (segs as f32);
            let stop_n = (key.stops.len() - 1) as f32;
            for s in 0..segs {
                let u = (s as f32) / ((segs - 1).max(1) as f32);
                let fi = u * stop_n;
                let idx = (fi.floor() as usize).min(key.stops.len() - 2);
                let fr = fi - idx as f32;
                let ca = key.stops[idx];
                let cb = key.stops[idx + 1];
                let col = egui::Color32::from_rgb(
                    ((ca[0] + (cb[0] - ca[0]) * fr).clamp(0.0, 1.0) * 255.0) as u8,
                    ((ca[1] + (cb[1] - ca[1]) * fr).clamp(0.0, 1.0) * 255.0) as u8,
                    ((ca[2] + (cb[2] - ca[2]) * fr).clamp(0.0, 1.0) * 255.0) as u8,
                );
                let seg_r = egui::Rect::from_min_size(
                    egui::pos2(bar_left + (s as f32) * step_w, bar_top),
                    egui::vec2(step_w + 0.8, 10.0),
                );
                painter.rect_filled(seg_r, 0.0, col);
            }
            painter.text(
                egui::pos2(bar_left, bar_top + 13.0),
                egui::Align2::LEFT_TOP,
                &key.min_label,
                egui::FontId::monospace(10.5),
                egui::Color32::LIGHT_GRAY,
            );
            painter.text(
                egui::pos2(bar_left + bar_w, bar_top + 13.0),
                egui::Align2::RIGHT_TOP,
                &key.max_label,
                egui::FontId::monospace(10.5),
                egui::Color32::LIGHT_GRAY,
            );
        }

        // 4. Calibrated Ångström Scale Bar (bottom-right)
        if let Some(ref sb) = self.session.scale_bar {
            let ppa = pixels_per_angstrom_at_target(&self.session.camera, h);
            let bar_px = (sb.length_a.max(1.0) * ppa).clamp(48.0, rect.width() * 0.34);
            let box_w = bar_px.max(76.0) + 20.0;
            let box_rect = egui::Rect::from_min_size(
                egui::pos2(rect.right() - box_w - 14.0, rect.bottom() - 52.0),
                egui::vec2(box_w, 38.0),
            );
            painter.rect_filled(box_rect, 5.0, pill_bg);
            painter.text(
                egui::pos2(box_rect.center().x, box_rect.top() + 5.0),
                egui::Align2::CENTER_TOP,
                &sb.label,
                egui::FontId::monospace(11.5),
                egui::Color32::WHITE,
            );
            let y = box_rect.bottom() - 10.0;
            let x0 = box_rect.center().x - bar_px * 0.5;
            let x1 = box_rect.center().x + bar_px * 0.5;
            let stroke = egui::Stroke::new(2.5_f32, egui::Color32::WHITE);
            painter.line_segment([egui::pos2(x0, y), egui::pos2(x1, y)], stroke);
            painter.line_segment([egui::pos2(x0, y - 4.0), egui::pos2(x0, y + 4.0)], stroke);
            painter.line_segment([egui::pos2(x1, y - 4.0), egui::pos2(x1, y + 4.0)], stroke);
        }
    }

    fn ui_measure_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Chemical Analysis · 3D Measurements · Overlays");
        ui.strong("3D Viewport Pick Mode (Click Atoms in 3D):");
        ui.horizontal_wrapped(|ui| {
            for (m, lbl) in [
                (PickMode::Inspect, "Inspect"),
                (PickMode::Distance, "Distance (2)"),
                (PickMode::Angle, "Angle (3)"),
                (PickMode::Dihedral, "Dihedral (4)"),
                (PickMode::SculptDrag, "Sculpt Drag (Shift)"),
            ] {
                if ui.selectable_value(&mut self.pick_mode, m, lbl).clicked() {
                    self.pick_buffer.clear();
                }
            }
        });
        if !self.pick_buffer.is_empty() {
            ui.small(format!(
                "Picked {} atom(s) — click remaining atom(s) in 3D viewport...",
                self.pick_buffer.len()
            ));
        }

        ui.separator();
        ui.strong("Interactions & Geometry (ChimeraX / PyMOL)");
        ui.horizontal_wrapped(|ui| {
            if ui.button("H-Bonds (3.5 Å)").clicked() {
                self.run_dsl("hbonds 3.5");
            }
            if ui.button("Steric Clashes").clicked() {
                self.run_dsl("clashes 0.60");
            }
            if ui.button("Contacts (4.0 Å)").clicked() {
                self.run_dsl("contacts 4.0");
            }
            if ui.button("SASA Burial (%)").clicked() {
                self.run_dsl("sasa");
            }
            if ui.button("Pocket Sticks (<4 Å)").clicked() {
                self.run_dsl("sticks byres (within 4.0 of ligand) and protein");
            }
            if ui.button("Info: Missing/Gaps").clicked() {
                self.run_dsl("info missing");
            }
            if ui.button("Define Axis (PCA)").clicked() {
                self.run_dsl("axis");
            }
            if ui.button("Define Plane (PCA)").clicked() {
                self.run_dsl("plane");
            }
            if ui.button("Putty / Sausage Mode").clicked() {
                self.run_dsl("putty on");
            }
            if ui.button("Clear Measurements").clicked() {
                self.pick_buffer.clear();
                self.run_dsl("measure clear");
            }
        });

        ui.separator();
        ui.strong("Publication Annotations (GUI + 4K PNG)");
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Key: Auto").clicked() {
                self.run_dsl("key auto");
            }
            if ui.small_button("Key: B-Factor").clicked() {
                self.run_dsl("key bfactor");
            }
            if ui.small_button("Key: Rainbow").clicked() {
                self.run_dsl("key rainbow");
            }
            if ui.small_button("Key: ESP").clicked() {
                self.run_dsl("key esp");
            }
            if ui.small_button("Key: MLP").clicked() {
                self.run_dsl("key mlp");
            }
            if ui.small_button("Key: pLDDT").clicked() {
                self.run_dsl("key plddt");
            }
            if ui.small_button("Key: Agg").clicked() {
                self.run_dsl("key agg");
            }
            if ui.small_button("Key: Off").clicked() {
                self.run_dsl("key off");
            }
        });
        ui.horizontal(|ui| {
            ui.label("Scale Bar (Å):");
            ui.add(egui::DragValue::new(&mut self.scalebar_len_a).range(5.0..=100.0));
            if ui.small_button("Show").clicked() {
                let len = self.scalebar_len_a;
                self.run_dsl(&format!("scalebar {len:.0}"));
            }
            if ui.small_button("Hide").clicked() {
                self.run_dsl("scalebar off");
            }
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.label2d_input)
                    .desired_width(185.0)
                    .hint_text("2D banner text..."),
            );
            if ui.small_button("+2D Label").clicked() {
                let txt = self.label2d_input.clone();
                if !txt.trim().is_empty() {
                    self.run_dsl(&format!("2dlabel 0.03 0.03 {txt}"));
                }
            }
            if ui.small_button("Clear Labels").clicked() {
                self.run_dsl("label clear");
            }
        });

        ui.separator();
        ui.strong(format!(
            "Measurements Log ({}):",
            self.session.measurements_log.len()
        ));
        for line in self.session.measurements_log.iter().rev().take(15) {
            ui.monospace(line);
        }
        if !self.session.last_module_markdown.is_empty() {
            ui.separator();
            ui.monospace(&self.session.last_module_markdown);
        }
    }

    fn ui_rama_agg_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Ramachandran (φ/ψ) · 3D Surface Aggregation");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Compute Ramachandran").clicked() {
                self.run_dsl("ramachandran");
            }
            if ui.button("Highlight Outliers in 3D").clicked() {
                self.run_dsl("ramachandran outliers");
            }
            if ui.small_button("Info: Gaps/Missing").clicked() {
                self.run_dsl("info missing");
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Basin Contour:");
            for kind in [
                RamaResidueKind::General,
                RamaResidueKind::Glycine,
                RamaResidueKind::Proline,
                RamaResidueKind::PreProline,
            ] {
                ui.selectable_value(&mut self.rama_contour_kind, kind, kind.label());
            }
        });

        // Interactive 2D [-180..+180] x [-180..+180] Ramachandran Plot synced with 3D!
        let mut clicked_res: Option<(char, i32)> = None;
        let side = 252.0_f32;
        let (resp, painter) = ui.allocate_painter(egui::vec2(side, side), egui::Sense::click());
        let rect = resp.rect;
        let grid_n = 36usize;
        let cell = side / (grid_n as f32);
        for gy in 0..grid_n {
            // Top = +180 deg, Bottom = -180 deg
            let psi = 180.0 - ((gy as f64 + 0.5) / grid_n as f64) * 360.0;
            for gx in 0..grid_n {
                let phi = -180.0 + ((gx as f64 + 0.5) / grid_n as f64) * 360.0;
                let (reg, _) = classify_ramachandran(self.rama_contour_kind, phi, psi);
                let col = match reg {
                    RamaRegion::Favored => egui::Color32::from_rgb(20, 92, 96),
                    RamaRegion::Allowed => egui::Color32::from_rgb(24, 54, 84),
                    RamaRegion::Outlier => egui::Color32::from_rgb(18, 22, 32),
                };
                let cr = egui::Rect::from_min_size(
                    egui::pos2(
                        rect.left() + gx as f32 * cell,
                        rect.top() + gy as f32 * cell,
                    ),
                    egui::vec2(cell + 0.5, cell + 0.5),
                );
                painter.rect_filled(cr, 0.0, col);
            }
        }
        // Zero-axes crosshairs and border
        let axis_stroke = egui::Stroke::new(
            1.0_f32,
            egui::Color32::from_rgba_unmultiplied(180, 196, 220, 110),
        );
        painter.line_segment(
            [
                egui::pos2(rect.center().x, rect.top()),
                egui::pos2(rect.center().x, rect.bottom()),
            ],
            axis_stroke,
        );
        painter.line_segment(
            [
                egui::pos2(rect.left(), rect.center().y),
                egui::pos2(rect.right(), rect.center().y),
            ],
            axis_stroke,
        );
        painter.rect_stroke(
            rect,
            2.0_f32,
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(100, 116, 139)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            egui::pos2(rect.left() + 4.0, rect.top() + 3.0),
            egui::Align2::LEFT_TOP,
            "ψ +180°",
            egui::FontId::monospace(9.5),
            egui::Color32::LIGHT_GRAY,
        );
        painter.text(
            egui::pos2(rect.right() - 4.0, rect.bottom() - 3.0),
            egui::Align2::RIGHT_BOTTOM,
            "φ +180°",
            egui::FontId::monospace(9.5),
            egui::Color32::LIGHT_GRAY,
        );

        let mut hover_tip: Option<String> = None;
        let pointer_pos = resp.hover_pos();
        if let Some(ref rama) = self.session.ramachandran {
            let mut nearest_click: Option<(f32, char, i32)> = None;
            let click_pos = if resp.clicked() {
                resp.interact_pointer_pos()
            } else {
                None
            };
            for p in &rama.points {
                let px = rect.left() + (((p.phi_deg + 180.0) / 360.0) as f32).clamp(0.0, 1.0) * side;
                let py = rect.top() + (((180.0 - p.psi_deg) / 360.0) as f32).clamp(0.0, 1.0) * side;
                let center = egui::pos2(px, py);
                let col = match p.region {
                    RamaRegion::Favored => egui::Color32::from_rgb(74, 222, 128),
                    RamaRegion::Allowed => egui::Color32::from_rgb(250, 204, 21),
                    RamaRegion::Outlier => egui::Color32::from_rgb(244, 63, 94),
                };
                let r = if p.region == RamaRegion::Outlier { 4.2 } else { 2.8 };
                if p.kind == RamaResidueKind::Proline {
                    painter.rect_filled(
                        egui::Rect::from_center_size(center, egui::vec2(r * 2.0, r * 2.0)),
                        0.0,
                        col,
                    );
                } else {
                    painter.circle_filled(center, r, col);
                }
                if let Some(hp) = pointer_pos
                    && hp.distance(center) < 7.0
                {
                    hover_tip = Some(format!(
                        "{}:{}{} ({}) | φ={:.1}° ψ={:.1}° | {}",
                        p.chain,
                        p.res_name,
                        p.res_seq,
                        p.kind.label(),
                        p.phi_deg,
                        p.psi_deg,
                        p.region.label()
                    ));
                }
                if let Some(cp) = click_pos {
                    let d = cp.distance(center);
                    if d < 12.0 && nearest_click.is_none_or(|(bd, _, _)| d < bd) {
                        nearest_click = Some((d, p.chain, p.res_seq));
                    }
                }
            }
            if let Some((_, ch, rs)) = nearest_click {
                clicked_res = Some((ch, rs));
            }
            ui.small(format!(
                "Evaluated {} res: {:.1}% Favored ({}) · {:.1}% Allowed ({}) · {:.1}% Outliers ({}) · {} cis",
                rama.points.len(),
                rama.favored_pct,
                rama.favored_count,
                rama.allowed_pct,
                rama.allowed_count,
                rama.outlier_pct,
                rama.outlier_count,
                rama.cis_peptide_count
            ));
            if let Some(tip) = hover_tip {
                ui.colored_label(egui::Color32::from_rgb(125, 211, 252), tip);
            } else {
                ui.small("Click any dot to select & focus that residue in the 3D viewport.");
            }
        } else {
            ui.small("Click 'Compute Ramachandran' to plot φ/ψ dihedrals for the active model.");
        }

        ui.separator();
        ui.strong("3D Surface Aggregation & Solubility Rescue (Aggrescan3D-style)");
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.agg_dynamic_mode, "Dynamic (B-factor)");
            if ui.button("Run 3D Aggregation Scan").clicked() {
                let cmd = if self.agg_dynamic_mode {
                    "aggregation dynamic"
                } else {
                    "aggregation static"
                };
                self.run_dsl(cmd);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Rescue:");
            ui.add(egui::TextEdit::singleline(&mut self.agg_rescue_chain).desired_width(28.0));
            ui.add(egui::DragValue::new(&mut self.agg_rescue_res));
            ui.label("→");
            egui::ComboBox::from_id_salt("agg_rescue_aa_cb")
                .selected_text(self.agg_rescue_aa.to_string())
                .width(46.0)
                .show_ui(ui, |ui| {
                    for aa in ['D', 'E', 'K', 'R', 'S', 'Q', 'N', 'A'] {
                        ui.selectable_value(&mut self.agg_rescue_aa, aa, aa.to_string());
                    }
                });
            if ui.small_button("Simulate").clicked() {
                let dyn_flag = if self.agg_dynamic_mode { " dynamic" } else { "" };
                let cmd = format!(
                    "aggregation rescue {}:{} {}{dyn_flag}",
                    self.agg_rescue_chain, self.agg_rescue_res, self.agg_rescue_aa
                );
                self.run_dsl(&cmd);
            }
        });

        let mut pick_agg_res: Option<(char, i32)> = None;
        if let Some(ref agg) = self.session.aggregation {
            ui.small(format!(
                "Avg: {:+.3} · Max: {:+.3} · Total+: {:.2} · Hotspots (>0.50): {} · Patches: {}",
                agg.avg_score,
                agg.max_score,
                agg.total_positive_score,
                agg.hotspot_count,
                agg.patches.len()
            ));
            let mut top: Vec<_> = agg
                .residue_scores
                .iter()
                .filter(|r| r.score_3d > 0.35)
                .collect();
            top.sort_by(|a, b| b.score_3d.total_cmp(&a.score_3d));
            if !top.is_empty() {
                ui.small("Top exposed hotspots (click to focus & load into Rescue):");
                ui.horizontal_wrapped(|ui| {
                    for r in top.iter().take(10) {
                        if ui
                            .small_button(format!(
                                "{}:{}{} ({:+.2})",
                                r.chain, r.res_name, r.res_seq, r.score_3d
                            ))
                            .clicked()
                        {
                            pick_agg_res = Some((r.chain, r.res_seq));
                        }
                    }
                });
            }
        }

        if let Some((ch, rs)) = pick_agg_res {
            self.agg_rescue_chain = ch.to_string();
            self.agg_rescue_res = rs;
            clicked_res = Some((ch, rs));
        }
        if let Some((ch, rs)) = clicked_res {
            self.mut_chain = ch.to_string();
            self.mut_res_seq = rs;
            let msg = self.session.focus_residue_3d(ch, rs);
            self.console_log.push(msg.clone());
            self.status_line = msg;
            self.sync_viewport_scene();
        }
        if !self.session.last_module_markdown.is_empty() {
            ui.separator();
            ui.monospace(&self.session.last_module_markdown);
        }
    }

    fn ui_msa_pae_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Synchronized MSA & Interactive 2D PAE Plot");
        ui.horizontal(|ui| {
            if ui.button("Build Synchronized MSA (All Models)").clicked() {
                self.run_dsl("msa");
            }
            if ui.button("Estimate Structural PAE").clicked() {
                self.run_dsl("pae auto");
            }
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.pae_json_input)
                    .desired_width(215.0)
                    .hint_text("/path/to/alphafold_or_boltz_pae.json"),
            );
            if ui.small_button("Load PAE JSON").clicked() {
                let p = self.pae_json_input.clone();
                if !p.trim().is_empty() {
                    self.run_dsl(&format!("pae {p}"));
                }
            }
        });

        // Synchronized clickable MSA rows
        let mut msa_select: Option<(char, i32)> = None;
        if let Some(ref msa) = self.session.msa {
            ui.separator();
            ui.strong(format!(
                "Match→Align MSA ({} models × {} cols):",
                msa.rows.len(),
                msa.num_columns
            ));
            egui::ScrollArea::horizontal()
                .id_salt("msa_scroll")
                .max_height(130.0)
                .show(ui, |ui| {
                    for row in &msa.rows {
                        ui.horizontal(|ui| {
                            ui.monospace(format!(
                                "{:<10}[{}]",
                                row.model_name.chars().take(10).collect::<String>(),
                                row.chain_id
                            ));
                            for (ci, &ch) in row.aligned_chars.iter().take(90).enumerate() {
                                let cons = msa.column_conservation.get(ci).copied().unwrap_or(0.0);
                                let col = if cons >= 0.99 {
                                    egui::Color32::from_rgb(52, 211, 153)
                                } else if cons >= 0.5 {
                                    egui::Color32::from_rgb(250, 204, 21)
                                } else {
                                    egui::Color32::GRAY
                                };
                                if ui
                                    .add(
                                        egui::Button::new(
                                            egui::RichText::new(ch.to_string())
                                                .monospace()
                                                .color(col),
                                        )
                                        .frame(false),
                                    )
                                    .clicked()
                                    && let Some(rseq) = row.res_seqs.get(ci).copied().flatten()
                                {
                                    msa_select = Some((row.chain_id, rseq));
                                }
                            }
                        });
                    }
                });
        }
        if let Some((ch, rs)) = msa_select {
            self.run_dsl(&format!("select msa_col, chain {ch} and resi {rs}"));
        }

        // Interactive 2D AlphaFold / Boltz PAE Error Heatmap
        let mut clicked_domain: Option<(usize, usize)> = None;
        if let Some(ref pae) = self.session.pae {
            ui.separator();
            ui.strong(format!(
                "2D PAE Matrix ({}×{}, mean {:.1} Å, {} domains — click to highlight in 3D):",
                pae.length,
                pae.length,
                pae.mean_pae,
                pae.domain_blocks.len()
            ));
            ui.horizontal_wrapped(|ui| {
                for (di, &(s, e)) in pae.domain_blocks.iter().enumerate() {
                    if ui
                        .small_button(format!("Domain {} ({}–{})", di + 1, s + 1, e + 1))
                        .clicked()
                    {
                        clicked_domain = Some((s, e));
                    }
                }
            });

            let side = 230.0_f32;
            let (resp, painter) = ui.allocate_painter(egui::vec2(side, side), egui::Sense::click());
            let rect = resp.rect;
            let n = pae.length.max(1);
            let bins = n.min(64);
            let cell = side / (bins as f32);
            let max_v = pae.max_pae.max(25.0);
            for by in 0..bins {
                let ry = (by * n) / bins;
                for bx in 0..bins {
                    let rx = (bx * n) / bins;
                    let v = pae.values.get(ry * n + rx).copied().unwrap_or(max_v);
                    // Classic AlphaFold green PAE palette: 0 A = dark emerald -> 30 A = white
                    let t = (v / max_v).clamp(0.0, 1.0);
                    let r = (12.0 + 240.0 * t) as u8;
                    let g = (108.0 + 145.0 * t) as u8;
                    let b = (68.0 + 184.0 * t) as u8;
                    let c_rect = egui::Rect::from_min_size(
                        egui::pos2(
                            rect.left() + (bx as f32) * cell,
                            rect.top() + (by as f32) * cell,
                        ),
                        egui::vec2(cell + 0.5, cell + 0.5),
                    );
                    painter.rect_filled(c_rect, 0.0, egui::Color32::from_rgb(r, g, b));
                }
            }
            if resp.clicked()
                && let Some(pos) = resp.interact_pointer_pos()
            {
                let rx = (((pos.x - rect.left()) / side).clamp(0.0, 0.999) * (n as f32)) as usize;
                let ry = (((pos.y - rect.top()) / side).clamp(0.0, 0.999) * (n as f32)) as usize;
                let span = 12usize;
                clicked_domain = Some((
                    rx.min(ry).saturating_sub(span),
                    (rx.max(ry) + span).min(n - 1),
                ));
            }
        }
        if let Some((s, e)) = clicked_domain {
            let msg = self.session.select_pae_domain(s, e);
            self.console_log.push(msg.clone());
            self.status_line = msg;
            self.sync_viewport_scene();
        }
    }

    fn ui_movie_morph_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("CEAlign · Conformational Morphing · Movie Maker");
        ui.strong("Sequence-Independent Superposition & Morphing");
        ui.horizontal(|ui| {
            ui.label("Morph Frames:");
            ui.add(egui::DragValue::new(&mut self.morph_frames).range(4..=60));
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("CEAlign #2 → #1 (0% Seq ID)").clicked() {
                self.run_dsl("cealign #2:A #1:A");
            }
            if ui.button("Morph #1 ↔ #2 (SHAKE)").clicked() {
                let f = self.morph_frames;
                self.run_dsl(&format!("morph #1 #2 {f}"));
                self.playing_trajectory = true;
            }
        });

        ui.separator();
        ui.strong("Movie Maker (Pure-Rust .gif & ffmpeg H.264 .mp4)");
        ui.horizontal(|ui| {
            ui.label("Mode:");
            for m in ["turn", "rock", "fly", "morph"] {
                ui.selectable_value(&mut self.movie_mode, m.to_string(), m);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Output:");
            ui.add(
                egui::TextEdit::singleline(&mut self.movie_out_path)
                    .desired_width(235.0)
                    .hint_text("/tmp/movie.gif or /tmp/movie.mp4"),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Frames:");
            ui.add(egui::DragValue::new(&mut self.movie_frames).range(6..=120));
            ui.label("FPS:");
            ui.add(egui::DragValue::new(&mut self.movie_fps).range(4..=30));
        });
        ui.horizontal(|ui| {
            ui.label("W:");
            ui.add(egui::DragValue::new(&mut self.movie_width).range(240..=1920));
            ui.label("H:");
            ui.add(egui::DragValue::new(&mut self.movie_height).range(180..=1080));
        });
        if ui.button("Export Animated Movie (.gif / .mp4)").clicked() {
            let cmd = format!(
                "movie {} {} {} {} {} {}",
                self.movie_mode,
                self.movie_out_path,
                self.movie_frames,
                self.movie_width,
                self.movie_height,
                self.movie_fps
            );
            self.run_dsl(&cmd);
        }
        ui.small(
            "Tip: Save 2+ Scenes (Shift+F1, Shift+F2) then select 'fly' for smooth camera flights!",
        );
    }

    fn ui_sym_glycans_module(&mut self, ui: &mut egui::Ui) {
        ui.heading("Crystallographic Symmetry & 3D SNFG Glycans");
        if let Some(mol) = self.session.active_molecule() {
            if let Some(ref uc) = mol.unit_cell {
                ui.strong(format!("CRYST1 Space Group: {}", uc.space_group));
                ui.small(format!(
                    "a={:.2} Å · b={:.2} Å · c={:.2} Å | α={:.1}° β={:.1}° γ={:.1}°",
                    uc.a, uc.b, uc.c, uc.alpha_deg, uc.beta_deg, uc.gamma_deg
                ));
            } else {
                ui.small("No CRYST1 unit-cell metadata in active model.");
            }
            ui.small(format!(
                "REMARK 350 BIOMT ops: {} | AltLoc records: {}",
                mol.biomt_ops.len(),
                mol.altlocs.len()
            ));
        }
        ui.horizontal(|ui| {
            ui.label("Symexp Cutoff (Å):");
            ui.add(
                egui::DragValue::new(&mut self.sym_cutoff_a)
                    .range(2.0..=20.0)
                    .speed(0.5),
            );
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Bio-Assembly (sym)").clicked() {
                self.run_dsl("sym");
            }
            if ui.button("Crystal Mates (symexp)").clicked() {
                let c = self.sym_cutoff_a;
                self.run_dsl(&format!("symexp {c:.1}"));
            }
            if ui.button("Crystal Contacts").clicked() {
                let c = self.sym_cutoff_a;
                self.run_dsl(&format!("crystalcontacts {c:.1}"));
            }
            if ui.button("Unit Cell On/Off").clicked() {
                let next = !self.session.render_opts.show_unit_cell;
                self.run_dsl(if next { "unitcell on" } else { "unitcell off" });
            }
        });

        ui.separator();
        ui.strong("3D SNFG Glycans (GlcNAc Cubes · Man Spheres · Fuc Cones)");
        ui.horizontal(|ui| {
            if ui.button("Detect & Show SNFG 3D Glycans").clicked() {
                self.run_dsl("snfg on");
            }
            if ui.small_button("Hide SNFG").clicked() {
                self.run_dsl("snfg off");
            }
        });
        if !self.session.last_module_markdown.is_empty() {
            ui.separator();
            ui.monospace(&self.session.last_module_markdown);
        }
    }
}

fn main() -> eframe::Result<()> {
    let startup = parse_startup_args();
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1520.0, 920.0])
            .with_title(
                "BiomolCraft v1.2.0 — CraftBio Biomolecular 3D Studio · Ramachandran · 3D Aggregation · Dunbrack Rotamers · ViewDockX",
            ),
        ..Default::default()
    };
    eframe::run_native(
        "BiomolCraft",
        options,
        Box::new(move |_cc| Ok(Box::new(BiomolCraftApp::with_startup(startup)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biomolcraft_gui_state_initializes_cleanly_with_rna_and_scene() {
        let app = BiomolCraftApp::default();
        assert!(app.rna_report.is_some());
        if let Some(srv) = app.link_server {
            srv.shutdown();
        }
    }
}
