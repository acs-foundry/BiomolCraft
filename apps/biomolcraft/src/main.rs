#![forbid(unsafe_code)]

//! `biomolcraft` — Flagship Pure-Rust `wgpu 24 + egui` Multi-Model Biomolecular 3D Studio,
//! Interactive Modeling Workbench, Native **`ViewDockX`** (`DockCraft` Craft Link v1),
//! Chemical Measurements, MSA/PAE, Conformational Morphing & Movie Maker (`v0.5.0`).

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use biomol_modules::{
    ModuleId, discover_installed_plugins, run_adc_module, run_antibody_module, run_enzyme_module,
    run_ncrna_module, run_pk_tmdd_module,
};
use biomol_physics::{OpenMmEnvStatus, check_openmm_biomol_env};
use biomol_rna::{NcRnaAnalysisReport, analyze_ncrna_mrna, find_rna_preset, rna_presets};
use biomol_script::{BiomolSession, PoseStatusTag};
use craftbio_scene::{
    ColorScheme, CraftBioLinkServer, DockCraftSceneBundle, GridBox3D, InboundLinkCommand,
    LightingPreset, RepresentationKind, discover_craft_instances,
};
use craftbio_struct::{ResidueClass, write_pdb};
use craftbio_viewport::{
    SharedViewportScene, pixels_per_angstrom_at_target, project_world_to_screen,
    show_biomol_viewport,
};
use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PickMode {
    Inspect,
    Distance,
    Angle,
    Dihedral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightModuleTab {
    ViewDockX,
    Measure,
    MsaPae,
    MovieMorph,
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
            open_path_input: "/home/acs/Datos/CraftBio/campaigns/biologics_pilot/1N8Z.pdb"
                .to_string(),
            fetch_id_input: "1CRN".to_string(),
            console_input: String::new(),
            console_log: vec![
                "BiomolCraft v0.5.0 — Multi-Model wgpu Depth32F+SSAO Studio · Measure/Putty · MSA/PAE · Morph/Movie · ViewDockX. Type 'help'.".to_string(),
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
            label2d_input: "BiomolCraft v0.5.0".to_string(),
            scalebar_len_a: 20.0,
            pae_json_input: String::new(),
            morph_frames: 20,
            movie_mode: "turn".to_string(),
            movie_out_path: "/tmp/biomolcraft_movie.gif".to_string(),
            movie_frames: 24,
            movie_width: 800,
            movie_height: 600,
            movie_fps: 12,
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
            if let Ok(msg) = app.session.load_structure_file(&open_p) {
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
                        self.open_path_input =
                            "/home/acs/Datos/CraftBio/campaigns/biologics_pilot/1N8Z.pdb"
                                .to_string();
                        self.run_dsl(&format!("open {} replace", self.open_path_input));
                        ui.close_menu();
                    }
                    if ui
                        .button("Load Pilot: 5JDS (Envafolimab VHH – PD-L1)")
                        .clicked()
                    {
                        self.open_path_input =
                            "/home/acs/Datos/CraftBio/campaigns/biologics_pilot/5JDS.pdb"
                                .to_string();
                        self.run_dsl(&format!("open {} replace", self.open_path_input));
                        ui.close_menu();
                    }
                    if ui
                        .button("Load Pilot: 1Y26 (Guanine Riboswitch ncRNA)")
                        .clicked()
                    {
                        self.open_path_input =
                            "/home/acs/Datos/CraftBio/campaigns/biologics_pilot/1Y26.pdb"
                                .to_string();
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
                    ui.label(
                        egui::RichText::new(
                            "BiomolCraft v0.5.0 · SSAO · Measure · PAE · Morph · ViewDockX",
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
                ui.horizontal(|ui| {
                    ui.strong("DSL>");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.console_input)
                            .desired_width(ui.available_width() - 75.0)
                            .hint_text(
                                "e.g. fetch 1crn; preset pub_white; color mlp; surface on; scene save f1",
                            ),
                    );
                    if (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        || ui.button("Run").clicked()
                    {
                        let cmd = std::mem::take(&mut self.console_input);
                        if !cmd.trim().is_empty() {
                            self.run_dsl(&cmd);
                        }
                    }
                });
            });

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
                        ColorScheme::Nucleotide,
                        ColorScheme::ElementCpk,
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
                        (RightModuleTab::MsaPae, "MSA / PAE"),
                        (RightModuleTab::MovieMorph, "Movie / Morph"),
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
                    RightModuleTab::MsaPae => self.ui_msa_pae_module(ui),
                    RightModuleTab::MovieMorph => self.ui_movie_morph_module(ui),
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

        // Center Primary 3D GPU Biomolecular Viewport (Offscreen Depth32Float + SSAO + 2D/3D HUD)
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let resp = show_biomol_viewport(
                    ui,
                    &mut self.session.camera,
                    &self.viewport_scene,
                    self.session.render_opts.show_silhouettes,
                    true,
                );
                // Paint live 3D/2D Labels, Color Key Legend & Calibrated Scale Bar directly over GPU viewport!
                self.paint_viewport_hud(ui, resp.rect);

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
                        self.picked_atom_info = format!(
                            "Picked {}:{}{}.{} ({}, B={:.1})",
                            a.chain, a.res_name, a.res_seq, a.name, a.element, a.b_factor
                        );
                        match self.pick_mode {
                            PickMode::Inspect => {}
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
            ui.add(egui::DragValue::new(&mut self.mut_rotamer).range(0..=2));
        });
        ui.horizontal(|ui| {
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
        });
        ui.horizontal(|ui| {
            if ui.button("Minimize Local MM (80 steps)").clicked() {
                self.run_dsl("minimize 80");
            }
            if ui.button("Undo").clicked() {
                self.run_dsl("undo");
            }
            if ui.button("Redo").clicked() {
                self.run_dsl("redo");
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
        ui.heading("Craft Link v1 · ArtCraft Export · WASM/JSON Plugins");
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
            if ui.small_button("Key: ESP").clicked() {
                self.run_dsl("key esp");
            }
            if ui.small_button("Key: MLP").clicked() {
                self.run_dsl("key mlp");
            }
            if ui.small_button("Key: pLDDT").clicked() {
                self.run_dsl("key plddt");
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
}

fn main() -> eframe::Result<()> {
    let startup = parse_startup_args();
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1520.0, 920.0])
            .with_title(
                "BiomolCraft v0.5.0 — CraftBio Biomolecular 3D Studio, PAE, Morph & ViewDockX",
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
