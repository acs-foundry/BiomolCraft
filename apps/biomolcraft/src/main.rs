#![forbid(unsafe_code)]

//! `biomolcraft` — Flagship Pure-Rust `wgpu 24 + egui` Biomolecular 3D Studio, Interactive
//! Modeling Workbench & Modular Design Suite (Schrödinger Maestro / BioLuminate / ChimeraX /
//! PyMOL replacement aligned with `storytold/craft-libs` Craft Link v1).

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use biomol_modules::{
    ModuleId, run_adc_module, run_antibody_module, run_enzyme_module, run_ncrna_module,
    run_pk_tmdd_module,
};
use biomol_physics::{OpenMmEnvStatus, check_openmm_biomol_env};
use biomol_rna::{NcRnaAnalysisReport, analyze_ncrna_mrna, find_rna_preset, rna_presets};
use biomol_script::BiomolSession;
use craftbio_scene::{ColorScheme, CraftBioLinkServer, InboundLinkCommand, RepresentationKind};
use craftbio_struct::ResidueClass;
use craftbio_viewport::{SharedViewportScene, show_biomol_viewport};
use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightModuleTab {
    Modeling,
    Physics,
    Antibodies,
    NcRnaMrna,
    Enzymes,
    AdcPk,
    Evidence,
}

pub struct BiomolCraftApp {
    dark_mode: bool,
    session: BiomolSession,
    viewport_scene: SharedViewportScene,
    link_server: Option<CraftBioLinkServer>,

    // UI state
    right_tab: RightModuleTab,
    open_path_input: String,
    console_input: String,
    console_log: Vec<String>,
    picked_atom_info: String,
    status_line: String,

    // Trajectory playback state
    playing_trajectory: bool,
    traj_fps: f32,
    last_frame_tick: Instant,

    // Physics module state
    anm_mode_sel: usize,
    anm_amp_a: f64,
    openmm_status: OpenMmEnvStatus,

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
}

impl Default for BiomolCraftApp {
    fn default() -> Self {
        let link_server =
            CraftBioLinkServer::start("biomolcraft", env!("CARGO_PKG_VERSION"), None).ok();
        let link_status = if link_server.is_some() {
            "Craft Link v1 socket active (ready for BioCraft/DockCraft/ChemCraft/ArtCraft)."
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
            console_input: String::new(),
            console_log: vec![
                "BiomolCraft v0.2.0 — Pure-Rust wgpu 3D Biomolecular Studio + Hybrid GPU Physics. Type 'help' for DSL.".to_string(),
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
        };

        // Load pilot 1N8Z structure automatically if present on disk
        let default_pdb = PathBuf::from(&app.open_path_input);
        if default_pdb.is_file()
            && let Ok(msg) = app.session.load_structure_file(&default_pdb)
        {
            app.console_log.push(msg.clone());
            app.status_line = msg;
        }
        if let Ok(r) = analyze_ncrna_mrna(&app.rna_name, "ncRNA — Riboswitch", &app.rna_seq, None)
        {
            app.rna_report = Some(r);
        }
        app.sync_viewport_scene();
        app
    }
}

impl BiomolCraftApp {
    fn sync_viewport_scene(&mut self) {
        if self.session.dirty_geometry {
            let data = self.session.build_viewport_scene();
            self.viewport_scene.update(data);
            self.session.dirty_geometry = false;
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
                    self.session.dirty_geometry = true;
                    self.console_log.push(format!(
                        "[Craft Link ← {}] Loaded scene '{}'",
                        doc.source_app, doc.title
                    ));
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
}

impl eframe::App for BiomolCraftApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_craft_link_inbox();
        // Smooth trajectory animation playback for ANM modes, Rust MD, and OpenMM DCD trajectories
        if self.playing_trajectory {
            let interval =
                Duration::from_secs_f32((1.0 / self.traj_fps.clamp(1.0, 60.0)).max(0.016));
            if self.last_frame_tick.elapsed() >= interval {
                if let Some(ref mut mol) = self.session.molecule
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
                        self.run_dsl(&format!("open {}", self.open_path_input));
                        ui.close_menu();
                    }
                    if ui
                        .button("Load Pilot: 5JDS (Envafolimab VHH – PD-L1)")
                        .clicked()
                    {
                        self.open_path_input =
                            "/home/acs/Datos/CraftBio/campaigns/biologics_pilot/5JDS.pdb"
                                .to_string();
                        self.run_dsl(&format!("open {}", self.open_path_input));
                        ui.close_menu();
                    }
                    if ui
                        .button("Load Pilot: 1Y26 (Guanine Riboswitch ncRNA)")
                        .clicked()
                    {
                        self.open_path_input =
                            "/home/acs/Datos/CraftBio/campaigns/biologics_pilot/1Y26.pdb"
                                .to_string();
                        self.run_dsl(&format!("open {}", self.open_path_input));
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
                        .button("Save CraftScene (/tmp/biomolcraft.craftscene.json)")
                        .clicked()
                    {
                        self.run_dsl("save /tmp/biomolcraft.craftscene.json");
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui
                        .button(if self.dark_mode {
                            "Light Studio Theme"
                        } else {
                            "Dark Studio Theme"
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
                        ("BioCraft (Dedicated Biologics Studio)", "biocraft"),
                        ("ChemCraft (Small-Mol, Synthesis & NRPs)", "chemcraft"),
                        ("DockCraft (SBDD Docking & FEP)", "dockcraft"),
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
                        egui::RichText::new("BiomolCraft v0.1.0 · CraftBio 3D Biomolecular Studio")
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
                // Sequence strip & interactive trajectory timeline synced with 3D
                let mut click_target: Option<(char, i32, char)> = None;
                let mut requested_frame: Option<usize> = None;
                if let Some(ref mol) = self.session.molecule {
                    ui.horizontal(|ui| {
                        ui.strong("Sequence Strip:");
                        if mol.frames.len() > 1 {
                            let play_lbl = if self.playing_trajectory { "⏸ Pause" } else { "▶ Play" };
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
                    egui::ScrollArea::horizontal().max_height(34.0).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for ch in &mol.chains {
                                if !matches!(
                                    ch.dominant_class,
                                    ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
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
                                        ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
                                    ) && ui
                                        .small_button(format!("{}", r.code))
                                        .on_hover_text(format!("{}:{}{}", r.chain, r.res_name, r.res_seq))
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
                    && let Some(ref mut mol) = self.session.molecule
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
                // PyMOL / ChimeraX Command Console
                egui::ScrollArea::vertical().max_height(66.0).stick_to_bottom(true).show(ui, |ui| {
                    for line in self.console_log.iter().rev().take(12).rev() {
                        ui.monospace(line);
                    }
                });
                ui.horizontal(|ui| {
                    ui.strong("DSL>");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.console_input)
                            .desired_width(ui.available_width() - 75.0)
                            .hint_text("e.g. style cartoon; color esp; module antibodies; mutate B:105 A; minimize 60"),
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

        // Left Panel: Scene Hierarchy, Representations, Colors & Quick Open
        egui::SidePanel::left("biomol_scene_tree")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Scene Hierarchy");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.open_path_input).desired_width(155.0),
                    );
                    if ui.button("Open").clicked() {
                        let p = self.open_path_input.clone();
                        self.run_dsl(&format!("open {p}"));
                    }
                });
                ui.separator();

                ui.strong("Representation");
                for r in [
                    RepresentationKind::Cartoon,
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
                ui.add_space(6.0);
                ui.strong("Color By");
                for c in [
                    ColorScheme::Chain,
                    ColorScheme::SecondaryStructure,
                    ColorScheme::Nucleotide,
                    ColorScheme::ElementCpk,
                    ColorScheme::BfactorPlddt,
                    ColorScheme::Sap3D,
                    ColorScheme::ElectrostaticEsp,
                ] {
                    if ui
                        .selectable_value(&mut self.session.render_opts.color_scheme, c, c.label())
                        .clicked()
                    {
                        self.session.dirty_geometry = true;
                        self.sync_viewport_scene();
                    }
                }
                ui.add_space(6.0);
                if ui
                    .checkbox(
                        &mut self.session.render_opts.show_surface_shell,
                        "Translucent Surface Shell",
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
                            "ChimeraX Silhouettes",
                        )
                        .changed()
                {
                    self.session.dirty_geometry = true;
                    self.sync_viewport_scene();
                }

                ui.separator();
                if let Some(ref mol) = self.session.molecule {
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

        // Right Dockable Biomolecular Module Panel
        egui::SidePanel::right("biomol_modules_panel")
            .resizable(true)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (t, lbl) in [
                        (RightModuleTab::Modeling, "Model 3D"),
                        (RightModuleTab::Physics, "Physics"),
                        (RightModuleTab::Antibodies, "Ab / VHH"),
                        (RightModuleTab::NcRnaMrna, "ncRNA/mRNA"),
                        (RightModuleTab::Enzymes, "Enzymes"),
                        (RightModuleTab::AdcPk, "ADC / PK"),
                        (RightModuleTab::Evidence, "Evidence"),
                    ] {
                        ui.selectable_value(&mut self.right_tab, t, lbl);
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| match self.right_tab {
                    RightModuleTab::Modeling => self.ui_modeling_module(ui),
                    RightModuleTab::Physics => self.ui_physics_module(ui),
                    RightModuleTab::Antibodies => self.ui_antibody_module(ui),
                    RightModuleTab::NcRnaMrna => self.ui_ncrna_module(ui),
                    RightModuleTab::Enzymes => self.ui_enzyme_module(ui),
                    RightModuleTab::AdcPk => self.ui_adc_pk_module(ui),
                    RightModuleTab::Evidence => self.ui_evidence_panel(ui),
                });
            });

        // Center Primary 3D GPU Biomolecular Viewport (80% visual focus)
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
                if resp.clicked()
                    && let Some(pos) = resp.interact_pointer_pos()
                    && let Some(ref mol) = self.session.molecule
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
                    }
                }
            });
    }
}

impl BiomolCraftApp {
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
        ui.heading("Antibodies & VHH Module (BioCraft Engine)");
        if ui
            .button("Analyze & Paint CDRs / Ala-Scan Hotspots in 3D")
            .clicked()
            && let Some(ref mol) = self.session.molecule
        {
            match run_antibody_module(mol, None, None) {
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
                && let Some(ref mol) = self.session.molecule
                && let Ok((vis, mut reps)) = run_ncrna_module(mol)
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
            && let Some(ref mol) = self.session.molecule
        {
            let vis = run_enzyme_module(mol);
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
                && let Some(ref mol) = self.session.molecule
                && let Ok((vis, _)) =
                    run_adc_module(mol, &self.adc_linker, &self.adc_payload, self.adc_dar, true)
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

    fn ui_evidence_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Scientific Provenance & Evidence Log");
        for ev in self.session.provenance.iter().rev() {
            ui.label(format!(
                "• [{:?}] ({}) {}",
                ev.kind, ev.source_app, ev.summary
            ));
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 900.0])
            .with_title("BiomolCraft v0.1.0 — CraftBio Biomolecular 3D Studio"),
        ..Default::default()
    };
    eframe::run_native(
        "BiomolCraft",
        options,
        Box::new(|_cc| Ok(Box::new(BiomolCraftApp::default()))),
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
