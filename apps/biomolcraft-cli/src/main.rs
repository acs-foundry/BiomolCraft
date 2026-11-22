#![forbid(unsafe_code)]

//! `biomolcraft-cli` — Headless Biomolecular 3D Studio CLI, Craft Link v1 Client,
//! and `storytold/craft-libs::craft-mcp-server` MCP Server.

mod mcp;

use std::path::{Path, PathBuf};

use biomol_enzyme::{analyze_enzyme_engineering, render_enzyme_markdown};
use biomol_model::{
    audit_structure_prep, minimize_local_mm, mutate_and_repack_best_rotamer, mutate_residue_3d,
};
use biomol_modules::{
    discover_installed_plugins, run_adc_module, run_antibody_module, run_cryoem_molmap_module,
    run_cryoem_volume_module, run_denovo_cofold_module, run_enzyme_engineering_module,
    run_enzyme_module, run_ncrna_module, run_physics_anm_module, run_physics_gbsa_module,
};
use biomol_physics::{
    OpenMmBiomolConfig, check_openmm_biomol_env, check_xtb_env, compute_gbsa_energy,
    render_gbsa_markdown, render_rmsd_trace_svg, run_local_langevin_md, run_openmm_biomol_md,
};
use biomol_rna::{
    analyze_ncrna_mrna, find_rna_preset, render_rna_arc_svg, render_rna_markdown, rna_presets,
};
use biomol_script::BiomolSession;
use clap::{Parser, Subcommand};
use craftbio_report::{export_artcraft_bundle, write_biomol_html_dossier};
use craftbio_sandbox::{DEFAULT_MAX_FILE_BYTES, PathSandbox, inspect_wasm_plugin};
use craftbio_scene::{
    CraftSceneDocument, discover_craft_instances, push_scene_via_craft_link,
    send_dsl_via_craft_link, send_trajectory_via_craft_link,
};
use craftbio_struct::{
    load_dcd_frames, matchmaker_align, parse_dx_map, parse_mrc_map, parse_structure_auto,
    write_mrc_map, write_multi_model_pdb, write_pdb,
};
use craftbio_viewport::{render_scene_png_bytes, write_scene_png};

#[derive(Parser)]
#[command(
    name = "biomolcraft-cli",
    version,
    about = "BiomolCraft: Pure-Rust Biomolecular 3D Studio, Modeling, ncRNA/mRNA, Modules & Craft Link CLI/MCP"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect a PDB/mmCIF structure (chains, Protein/RNA/DNA/Ligand/Ion counts, disulfides, 3D RNA base pairs).
    Inspect {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Render a publication-grade 3D biomolecular PNG headlessly (up to 4K 3840x2160).
    Render {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(
            long,
            default_value = "cartoon",
            help = "cartoon | nucleic | sticks | spheres | surface | lines"
        )]
        style: String,
        #[arg(
            long,
            default_value = "chain",
            help = "chain | ss | cpk | nucleotide | plddt | sap | esp"
        )]
        color: String,
        #[arg(
            long,
            help = "Optional module overlay: antibodies | ncrna | enzymes | adc"
        )]
        module: Option<String>,
        #[arg(long)]
        surface: bool,
        #[arg(long, default_value_t = 1600)]
        width: u32,
        #[arg(long, default_value_t = 1200)]
        height: u32,
        #[arg(long)]
        transparent: bool,
        #[arg(long)]
        out_png: PathBuf,
    },
    /// Sequence-guided MatchMaker / Kabsch 3D structural alignment (Proteins & RNA/DNA).
    Align {
        #[arg(long)]
        mobile: PathBuf,
        #[arg(long, default_value_t = 'A')]
        mobile_chain: char,
        #[arg(long)]
        target: PathBuf,
        #[arg(long, default_value_t = 'A')]
        target_chain: char,
        #[arg(long)]
        out_pdb: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// 3D point mutagenesis with Dunbrack rotamers, clash evaluation, and optional local MM relaxation.
    Mutate {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        chain: char,
        #[arg(long)]
        res: i32,
        #[arg(long)]
        aa: char,
        #[arg(long)]
        rotamer: Option<usize>,
        #[arg(long)]
        minimize_steps: Option<usize>,
        #[arg(long)]
        out_pdb: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Run a biomolecular module (antibodies | ncrna | enzymes | adc) on a 3D structure.
    Module {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, help = "antibodies | ncrna | enzymes | adc")]
        name: String,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_scene: Option<PathBuf>,
    },
    /// Fold & audit ncRNA / mRNA / siRNA / Riboswitch (Zuker MFE 2D, G-quadruplex, TLR/m1Psi, LNP, SVG arc diagram).
    RnaFold {
        #[arg(long)]
        preset: Option<String>,
        #[arg(long)]
        seq: Option<String>,
        #[arg(long)]
        pdb: Option<PathBuf>,
        #[arg(long)]
        list_presets: bool,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_svg: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Execute a PyMOL/ChimeraX-compatible BiomolCraft DSL script string or macro file.
    Script {
        #[arg(long)]
        pdb: Option<PathBuf>,
        #[arg(long)]
        dsl: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Run pure-Rust ANM Normal Mode Analysis, export animated multi-MODEL PDB & optional porcupine PNG.
    Anm {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, default_value_t = 1)]
        mode: usize,
        #[arg(long, default_value_t = 2.2)]
        amp: f64,
        #[arg(long, default_value_t = 20)]
        frames: usize,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_traj_pdb: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
    },
    /// Compute pure-Rust Still GB/SA Solvation, 3D Electrostatic Dipole Moment & interface ΔG_bind.
    Gbsa {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Run Biomolecular Molecular Dynamics (`--engine rust` instant Langevin or `--engine openmm` live GPU CUDA MD).
    Md {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, default_value = "rust", help = "rust | openmm")]
        engine: String,
        #[arg(long, default_value_t = 120)]
        steps: usize,
        #[arg(long, default_value_t = 10.0)]
        production_ps: f64,
        #[arg(
            long,
            default_value = "implicit_gbn2",
            help = "implicit_gbn2 | explicit_tip3p"
        )]
        solvent: String,
        #[arg(long, default_value = "auto", help = "auto | CUDA | OpenCL | CPU")]
        platform: String,
        #[arg(long, default_value = "biomol_md_out")]
        out_dir: PathBuf,
        #[arg(long)]
        out_svg: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Run full Enzyme & Protein Engineering audit (catalytic triads, metals, 3D substrate tunnels, disulfides).
    Enzyme {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Probe local Tier-2 GPU Physics engines (OpenMM CUDA/OpenCL/CPU & GFN2-xTB).
    PhysicsEnv,
    /// Cryo-EM MRC2014 / OpenDX Volumetric Maps: synthesize `molmap`, open `.mrc`/`.dx`, `fitmap`, and render 3D density contour PNG.
    Cryoem {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        map: Option<PathBuf>,
        #[arg(long, default_value_t = 4.5)]
        molmap_res: f64,
        #[arg(long, default_value_t = 1.5)]
        sigma: f64,
        #[arg(long)]
        fit: bool,
        #[arg(long)]
        out_mrc: Option<PathBuf>,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
    },
    /// Stage a zero-weight De Novo & Co-Fold (`RFdiffusion` + `ProteinMPNN` + `ESM2` + `Boltz-2/Chai-1`) binder campaign from 3D hotspots.
    Denovo {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, default_value_t = 'C')]
        chain: char,
        #[arg(long, help = "Comma-separated hotspot residues, e.g. 'C558,C560,C573'")]
        hotspots: Option<String>,
        #[arg(long)]
        out_dir: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
    },
    /// Export a single-file offline-Mol* 3D interactive HTML biomolecular dossier.
    ExportHtml {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(
            long,
            help = "Optional module to run before export: antibodies|ncrna|enzymes|adc|gbsa"
        )]
        module: Option<String>,
        #[arg(long)]
        out_html: PathBuf,
        #[arg(long, default_value_t = true)]
        offline: bool,
    },
    /// Export an ArtCraft publication bundle (1600x1200 PNG + GridCraft TSV + WordCraft/DeckCraft Markdown).
    ExportArtcraft {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        module: Option<String>,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Inspect a `.wasm` plugin safely (never executes code) or list installed `$CRAFT_DATA_DIR/plugins`.
    PluginInspect {
        #[arg(long)]
        wasm: Option<PathBuf>,
        #[arg(long)]
        list: bool,
    },
    /// Discover running CraftBio/ArtCraft apps via Craft Link v1 (`craft-ipc`) and optionally push a PDB/DCD/DSL command.
    Link {
        #[arg(long, default_value = "biomolcraft")]
        app: String,
        #[arg(long)]
        push_pdb: Option<PathBuf>,
        #[arg(long)]
        push_dcd: Option<PathBuf>,
        #[arg(long)]
        dsl: Option<String>,
    },
    /// Start the hardened BiomolCraft MCP server over stdio (`storytold/craft-mcp-server`).
    Mcp {
        #[arg(long)]
        read_root: Option<PathBuf>,
        #[arg(long)]
        write_root: Option<PathBuf>,
    },
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    if let Some(p) = path.parent()
        && !p.as_os_str().is_empty()
    {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, text).map_err(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { pdb, json } => {
            let raw = std::fs::read_to_string(&pdb).map_err(|e| e.to_string())?;
            let id = pdb.file_stem().and_then(|s| s.to_str()).unwrap_or("mol");
            let mol = parse_structure_auto(id, &raw).map_err(|e| e.to_string())?;
            let audit = audit_structure_prep(&mol);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&audit).map_err(|e| e.to_string())?
                );
            } else {
                println!("# BiomolCraft Structure Inspection: {}\n", mol.title);
                println!(
                    "- **Atoms:** `{}` | **Chains:** `{}` (Protein: `{:?}`, Nucleic: `{:?}`)",
                    audit.total_atoms,
                    mol.chains.len(),
                    audit.protein_chains,
                    audit.nucleic_chains
                );
                println!(
                    "- **Ligands:** `{}` | **Ions:** `{}` | **Waters:** `{}` | **Disulfides:** `{}` | **3D RNA/DNA Base Pairs:** `{}`",
                    audit.ligand_residues,
                    audit.ion_count,
                    audit.water_count,
                    audit.disulfide_count,
                    audit.rna_base_pair_count
                );
                for n in &audit.notes {
                    println!("- {n}");
                }
            }
        }
        Commands::Render {
            pdb,
            style,
            color,
            module,
            surface,
            width,
            height,
            transparent,
            out_png,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let _ = sess
                .execute_dsl(&format!("style {style}; color {color}"))
                .map_err(|e| e.to_string())?;
            if surface {
                let _ = sess.execute_dsl("surface on").map_err(|e| e.to_string())?;
            }
            if let Some(m) = module {
                let _ = sess
                    .execute_dsl(&format!("module {m}"))
                    .map_err(|e| e.to_string())?;
            }
            let scene = sess.build_viewport_scene();
            let bytes = write_scene_png(&scene, &sess.camera, width, height, transparent, &out_png)
                .map_err(|e| e.to_string())?;
            println!(
                "Rendered {}x{} PNG to {} ({bytes} bytes, spheres={}, capsules={})",
                width,
                height,
                out_png.display(),
                scene.spheres.len(),
                scene.capsules.len()
            );
        }
        Commands::Align {
            mobile,
            mobile_chain,
            target,
            target_chain,
            out_pdb,
            json,
        } => {
            let m_raw = std::fs::read_to_string(&mobile).map_err(|e| e.to_string())?;
            let t_raw = std::fs::read_to_string(&target).map_err(|e| e.to_string())?;
            let mut m_mol = parse_structure_auto("mobile", &m_raw).map_err(|e| e.to_string())?;
            let t_mol = parse_structure_auto("target", &t_raw).map_err(|e| e.to_string())?;
            let tf = matchmaker_align(&mut m_mol, mobile_chain, &t_mol, target_chain)
                .map_err(|e| e.to_string())?;
            if let Some(p) = out_pdb {
                write_text(&p, &write_pdb(&m_mol))?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&tf).map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "Aligned chain {mobile_chain} -> {target_chain}: RMSD = {:.4} Å ({} matched guide atoms).",
                    tf.rmsd_a, tf.aligned_pairs
                );
            }
        }
        Commands::Mutate {
            pdb,
            chain,
            res,
            aa,
            rotamer,
            minimize_steps,
            out_pdb,
            json,
        } => {
            let raw = std::fs::read_to_string(&pdb).map_err(|e| e.to_string())?;
            let mut mol = parse_structure_auto("mutant", &raw).map_err(|e| e.to_string())?;
            let rep = if let Some(ri) = rotamer {
                mutate_residue_3d(&mut mol, chain, res, aa, ri).map_err(|e| e.to_string())?
            } else {
                mutate_and_repack_best_rotamer(&mut mol, chain, res, aa)
                    .map_err(|e| e.to_string())?
            };
            let min_rep = minimize_steps.map(|steps| {
                // Local 6.5 A shell around mutated residue
                let center = mol
                    .chain(chain)
                    .and_then(|c| c.residues.iter().find(|r| r.res_seq == res))
                    .and_then(|r| r.guide_idx)
                    .map(|i| mol.atoms[i].coords);
                let mask: Vec<bool> = if let Some(c) = center {
                    mol.atoms
                        .iter()
                        .map(|a| craftbio_struct::dist3(a.coords, c) <= 6.5)
                        .collect()
                } else {
                    vec![true; mol.atoms.len()]
                };
                minimize_local_mm(&mut mol, &mask, steps)
            });
            if let Some(p) = out_pdb {
                write_text(&p, &write_pdb(&mol))?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json!({ "mutation": rep, "minimization": min_rep })
                    )
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "Mutated {chain}:{}{res}{} (rotamer {}/{}, chi1={:.0}°): {} clashes, {} new H-bonds.",
                    rep.wild_type,
                    rep.mutant,
                    rep.rotamer_index + 1,
                    rep.total_rotamers,
                    rep.chi1_deg,
                    rep.clash_count,
                    rep.new_hbond_count
                );
                if let Some(m) = min_rep {
                    println!(
                        "Minimized {} local atoms ({} steps): E {:.2} -> {:.2} kcal/mol (max disp {:.3} Å).",
                        m.atoms_relaxed,
                        m.steps,
                        m.energy_before_kcal,
                        m.energy_after_kcal,
                        m.max_displacement_a
                    );
                }
            }
        }
        Commands::Module {
            pdb,
            name,
            out_md,
            out_scene,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let mol = sess.molecule.as_ref().ok_or("load failed")?;
            let vis = match name.to_ascii_lowercase().as_str() {
                "antibodies" | "ab" | "vhh" => run_antibody_module(mol, None, None)?.visual,
                "ncrna" | "rna" | "mrna" => run_ncrna_module(mol)?.0,
                "enzymes" | "enzyme" => run_enzyme_module(mol),
                "adc" => run_adc_module(mol, "ggfg", "dxd", 8.0, true)?.0,
                other => return Err(format!("unknown module '{other}'")),
            };
            if let Some(p) = out_md {
                write_text(&p, &vis.summary_markdown)?;
            }
            if let Some(p) = out_scene {
                sess.render_opts.residue_highlights = vis.residue_highlights;
                sess.render_opts.sphere_overlays = vis.sphere_overlays;
                sess.render_opts.line_overlays = vis.line_overlays;
                let doc = sess.to_craftscene_document();
                write_text(
                    &p,
                    &serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?,
                )?;
            }
            print!("{}", vis.summary_markdown);
        }
        Commands::RnaFold {
            preset,
            seq,
            pdb,
            list_presets,
            out_md,
            out_svg,
            json,
        } => {
            if list_presets {
                for p in rna_presets() {
                    println!(
                        "`{}` — {} ({}) [{} nt]",
                        p.id,
                        p.name,
                        p.class,
                        p.sequence.len()
                    );
                }
                return Ok(());
            }
            let rep = if let Some(pp) = pdb {
                let raw = std::fs::read_to_string(&pp).map_err(|e| e.to_string())?;
                let id = pp.file_stem().and_then(|s| s.to_str()).unwrap_or("rna3d");
                let mol = parse_structure_auto(id, &raw).map_err(|e| e.to_string())?;
                let (_, mut reps) = run_ncrna_module(&mol)?;
                reps.pop().ok_or("no RNA chain in PDB")?
            } else if let Some(pid) = preset {
                let p =
                    find_rna_preset(&pid).ok_or_else(|| format!("unknown RNA preset '{pid}'"))?;
                analyze_ncrna_mrna(p.name, p.class, p.sequence, None).map_err(|e| e.to_string())?
            } else if let Some(s) = seq {
                analyze_ncrna_mrna("Custom_RNA", "ncRNA/mRNA", &s, None)
                    .map_err(|e| e.to_string())?
            } else {
                return Err("provide --preset, --seq, --pdb, or --list-presets".to_string());
            };
            let md = render_rna_markdown(&rep);
            if let Some(p) = out_md {
                write_text(&p, &md)?;
            }
            if let Some(p) = out_svg {
                write_text(&p, &render_rna_arc_svg(&rep.name, &rep.fold))?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())?
                );
            } else {
                print!("{md}");
            }
        }
        Commands::Script { pdb, dsl, file } => {
            let mut sess = BiomolSession::default();
            if let Some(p) = pdb {
                println!(
                    "{}",
                    sess.load_structure_file(&p).map_err(|e| e.to_string())?
                );
            }
            let cmds = if let Some(f) = file {
                std::fs::read_to_string(&f).map_err(|e| e.to_string())?
            } else {
                dsl.unwrap_or_default()
            };
            for line in cmds.lines() {
                let l = line.trim();
                if l.is_empty() || l.starts_with('#') {
                    continue;
                }
                println!("{}", sess.execute_dsl(l).map_err(|e| e.to_string())?);
            }
        }
        Commands::Anm {
            pdb,
            mode,
            amp,
            frames,
            out_md,
            out_traj_pdb,
            out_png,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let mol = sess.molecule.as_mut().ok_or("no molecule")?;
            let (vis, _) = run_physics_anm_module(mol, mode, amp, frames)?;
            if let Some(p) = out_md {
                write_text(&p, &vis.summary_markdown)?;
            }
            if let Some(p) = out_traj_pdb {
                write_text(&p, &write_multi_model_pdb(mol))?;
            }
            sess.render_opts.residue_highlights = vis.residue_highlights;
            sess.render_opts.line_overlays = vis.line_overlays;
            sess.dirty_geometry = true;
            if let Some(p) = out_png {
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered ANM mode {mode} PNG to {} ({bytes} bytes).",
                    p.display()
                );
            }
            print!("{}", vis.summary_markdown);
        }
        Commands::Gbsa { pdb, out_md, json } => {
            let raw = std::fs::read_to_string(&pdb).map_err(|e| e.to_string())?;
            let id = pdb.file_stem().and_then(|s| s.to_str()).unwrap_or("mol");
            let mol = parse_structure_auto(id, &raw).map_err(|e| e.to_string())?;
            let rep = compute_gbsa_energy(&mol, None, None);
            let md = render_gbsa_markdown(&rep);
            if let Some(p) = out_md {
                write_text(&p, &md)?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())?
                );
            } else {
                print!("{md}");
            }
        }
        Commands::Md {
            pdb,
            engine,
            steps,
            production_ps,
            solvent,
            platform,
            out_dir,
            out_svg,
            json,
        } => {
            if engine.eq_ignore_ascii_case("openmm") {
                let cfg = OpenMmBiomolConfig {
                    input_pdb: pdb,
                    out_dir,
                    solvent_model: solvent,
                    production_ps,
                    platform,
                    ..OpenMmBiomolConfig::default()
                };
                let res = run_openmm_biomol_md(&cfg)?;
                // Verify pure-Rust binary DCD loading right back into craftbio-struct!
                let topo_raw =
                    std::fs::read_to_string(&res.topology_pdb_path).map_err(|e| e.to_string())?;
                let mut mol =
                    parse_structure_auto("openmm_topo", &topo_raw).map_err(|e| e.to_string())?;
                let dcd_bytes =
                    std::fs::read(&res.trajectory_dcd_path).map_err(|e| e.to_string())?;
                let verified_frames =
                    load_dcd_frames(&mut mol, &dcd_bytes).map_err(|e| e.to_string())?;
                if let Some(p) = out_svg {
                    let svg = render_rmsd_trace_svg(
                        &format!("OpenMM {} MD Backbone RMSD", res.platform_used),
                        &res.rmsd_trace_a,
                    );
                    write_text(&p, &svg)?;
                }
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&res).map_err(|e| e.to_string())?
                    );
                } else {
                    println!(
                        "OpenMM {} ({}) MD complete in {:.2}s: {} atoms, {} frames (DCD verified by pure-Rust loader: {} frames), E_min={:.1} kJ/mol, mean RMSD={:.3} Å, final RMSD={:.3} Å, mean Rg={:.2} Å.",
                        res.openmm_version,
                        res.platform_used,
                        res.elapsed_seconds,
                        res.atom_count,
                        res.frames_saved,
                        verified_frames,
                        res.minimized_energy_kj,
                        res.mean_backbone_rmsd_a,
                        res.final_backbone_rmsd_a,
                        res.mean_rg_a
                    );
                    println!("Topology PDB: {}", res.topology_pdb_path);
                    println!("Trajectory DCD: {}", res.trajectory_dcd_path);
                }
            } else {
                let raw = std::fs::read_to_string(&pdb).map_err(|e| e.to_string())?;
                let id = pdb.file_stem().and_then(|s| s.to_str()).unwrap_or("mol");
                let mut mol = parse_structure_auto(id, &raw).map_err(|e| e.to_string())?;
                let rep = run_local_langevin_md(&mut mol, &[], steps, 6, 300.0, 1.5, 2026)?;
                std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
                let traj_path = out_dir.join(format!("{id}_rust_md.pdb"));
                write_text(&traj_path, &write_multi_model_pdb(&mol))?;
                if let Some(p) = out_svg {
                    let vals: Vec<f64> = rep.trace.iter().map(|t| t.rmsd_a).collect();
                    write_text(
                        &p,
                        &render_rmsd_trace_svg("Pure-Rust Local Langevin MD RMSD", &vals),
                    )?;
                }
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())?
                    );
                } else {
                    println!(
                        "Pure-Rust Langevin MD ({} mobile atoms, {} steps -> {} frames): final RMSD {:.3} Å, mean E {:.1} kcal/mol -> {}",
                        rep.mobile_atoms,
                        rep.total_steps,
                        rep.frames_recorded,
                        rep.final_rmsd_a,
                        rep.mean_potential_kcal,
                        traj_path.display()
                    );
                }
            }
        }
        Commands::Enzyme {
            pdb,
            out_md,
            out_png,
            json,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let mol = sess.molecule.as_ref().ok_or("no molecule")?;
            let (vis, rep) = run_enzyme_engineering_module(mol);
            if let Some(p) = out_md {
                write_text(&p, &vis.summary_markdown)?;
            }
            if let Some(p) = out_png {
                sess.render_opts.residue_highlights = vis.residue_highlights;
                sess.render_opts.sphere_overlays = vis.sphere_overlays;
                sess.render_opts.line_overlays = vis.line_overlays;
                sess.dirty_geometry = true;
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered Enzyme/Tunnels PNG to {} ({bytes} bytes).",
                    p.display()
                );
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())?
                );
            } else {
                let _ = (
                    analyze_enzyme_engineering,
                    render_enzyme_markdown,
                    run_physics_gbsa_module,
                );
                print!("{}", vis.summary_markdown);
            }
        }
        Commands::PhysicsEnv => {
            let omm = check_openmm_biomol_env();
            let xtb = check_xtb_env();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "openmm": omm,
                    "gfn2_xtb_available": xtb
                }))
                .map_err(|e| e.to_string())?
            );
        }
        Commands::Cryoem {
            pdb,
            map,
            molmap_res,
            sigma,
            fit,
            out_mrc,
            out_md,
            out_png,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let (vis, grid) = if let Some(mp) = map {
                let g = if mp
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("dx"))
                {
                    let txt = std::fs::read_to_string(&mp).map_err(|e| e.to_string())?;
                    parse_dx_map(&mp.display().to_string(), &txt).map_err(|e| e.to_string())?
                } else {
                    let bytes = std::fs::read(&mp).map_err(|e| e.to_string())?;
                    parse_mrc_map(&mp.display().to_string(), &bytes).map_err(|e| e.to_string())?
                };
                let mol = sess.molecule.as_mut().ok_or("no molecule")?;
                let (v, _) = run_cryoem_volume_module(mol, &g, sigma, fit);
                (v, g)
            } else {
                let mol = sess.molecule.as_mut().ok_or("no molecule")?;
                let (v, g, _) = run_cryoem_molmap_module(mol, molmap_res, sigma)?;
                (v, g)
            };
            if let Some(p) = out_mrc {
                if let Some(parent) = p.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                std::fs::write(&p, write_mrc_map(&grid)).map_err(|e| e.to_string())?;
            }
            if let Some(p) = out_md {
                write_text(&p, &vis.summary_markdown)?;
            }
            if let Some(p) = out_png {
                sess.render_opts.sphere_overlays = vis.sphere_overlays;
                sess.dirty_geometry = true;
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered Cryo-EM volumetric PNG to {} ({bytes} bytes).",
                    p.display()
                );
            }
            print!("{}", vis.summary_markdown);
        }
        Commands::Denovo {
            pdb,
            chain,
            hotspots,
            out_dir,
            out_png,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let hs: Vec<String> = hotspots
                .unwrap_or_default()
                .split([',', ' ', ';'])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            let mol = sess.molecule.as_ref().ok_or("no molecule")?;
            let (vis, _) = run_denovo_cofold_module(mol, chain, &hs, out_dir.as_deref())?;
            if let Some(p) = out_png {
                sess.render_opts.residue_highlights = vis.residue_highlights;
                sess.render_opts.sphere_overlays = vis.sphere_overlays;
                sess.dirty_geometry = true;
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered De Novo hotspots PNG to {} ({bytes} bytes).",
                    p.display()
                );
            }
            print!("{}", vis.summary_markdown);
        }
        Commands::ExportHtml {
            pdb,
            module,
            out_html,
            offline,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let mut extra_svg: Option<String> = None;
            if let Some(m) = module {
                if m.eq_ignore_ascii_case("gbsa") {
                    let _ = sess.execute_dsl("gbsa").map_err(|e| e.to_string())?;
                } else {
                    let _ = sess
                        .execute_dsl(&format!("module {m}"))
                        .map_err(|e| e.to_string())?;
                    if m.eq_ignore_ascii_case("ncrna")
                        && let Some(ref mol) = sess.molecule
                        && let Ok((_, reps)) = run_ncrna_module(mol)
                        && let Some(first) = reps.first()
                    {
                        extra_svg = Some(render_rna_arc_svg(&first.name, &first.fold));
                    }
                }
            }
            let mol = sess.molecule.as_ref().ok_or("no molecule")?;
            let mut svgs = Vec::new();
            if let Some(ref s) = extra_svg {
                svgs.push(("ncRNA Base-Pair Arc Diagram", s.as_str()));
            }
            let sum = write_biomol_html_dossier(
                &mol.title,
                &write_pdb(mol),
                &sess.last_module_markdown,
                &svgs,
                offline,
                &out_html,
            )?;
            println!(
                "Wrote BiomolCraft interactive 3D HTML dossier: {} ({} bytes, molstar_inline_offline={})",
                sum.out_path, sum.bytes, sum.molstar_inline_offline
            );
        }
        Commands::ExportArtcraft {
            pdb,
            module,
            out_dir,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            if let Some(m) = module {
                let _ = sess
                    .execute_dsl(&format!("module {m}"))
                    .map_err(|e| e.to_string())?;
            }
            let title = sess
                .molecule
                .as_ref()
                .map_or_else(|| "BiomolCraft".to_string(), |m| m.title.clone());
            let atoms = sess.molecule.as_ref().map_or(0, |m| m.atoms.len());
            let chains = sess.molecule.as_ref().map_or(0, |m| m.chains.len());
            let sc = sess.build_viewport_scene();
            let png = render_scene_png_bytes(&sc, &sess.camera, 1600, 1200, false)
                .map_err(|e| e.to_string())?;
            let rows = vec![
                ("Structure", title.clone()),
                ("Atoms", atoms.to_string()),
                ("Chains", chains.to_string()),
                ("Provenance Entries", sess.provenance.len().to_string()),
            ];
            let sum =
                export_artcraft_bundle(&title, &png, &rows, &sess.last_module_markdown, &out_dir)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&sum).map_err(|e| e.to_string())?
            );
        }
        Commands::PluginInspect { wasm, list } => {
            if let Some(wp) = wasm {
                let bytes = std::fs::read(&wp).map_err(|e| e.to_string())?;
                let rep = inspect_wasm_plugin(&bytes).map_err(|e| e.to_string())?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())?
                );
            } else if list {
                let plugs = discover_installed_plugins();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&plugs).map_err(|e| e.to_string())?
                );
            } else {
                println!("Pass --wasm <plugin.wasm> or --list.");
            }
        }
        Commands::Link {
            app,
            push_pdb,
            push_dcd,
            dsl,
        } => {
            let instances =
                discover_craft_instances(None, Some(&app)).map_err(|e| e.to_string())?;
            if instances.is_empty() {
                println!("No live Craft Link v1 instances found for app '{app}'.");
                return Ok(());
            }
            println!(
                "Found {} live Craft Link instance(s) of '{app}'.",
                instances.len()
            );
            if let (Some(p), Some(d)) = (&push_pdb, &push_dcd) {
                let rep = send_trajectory_via_craft_link(
                    &instances[0],
                    "biomolcraft-cli",
                    &p.display().to_string(),
                    &d.display().to_string(),
                )
                .map_err(|e| e.to_string())?;
                println!("Pushed trajectory via Craft Link: {rep}");
            } else if let Some(p) = push_pdb {
                let doc = CraftSceneDocument {
                    title: p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Scene")
                        .to_string(),
                    source_app: "biomolcraft-cli".to_string(),
                    pdb_path: Some(p.display().to_string()),
                    ..Default::default()
                };
                let rep =
                    push_scene_via_craft_link(&instances[0], &doc).map_err(|e| e.to_string())?;
                println!("Pushed scene via Craft Link: {rep}");
            }
            if let Some(cmd) = dsl {
                let rep = send_dsl_via_craft_link(&instances[0], "biomolcraft-cli", &cmd)
                    .map_err(|e| e.to_string())?;
                println!("Sent DSL via Craft Link: {rep}");
            }
        }
        Commands::Mcp {
            read_root,
            write_root,
        } => {
            let sbx = PathSandbox::new(read_root, write_root, DEFAULT_MAX_FILE_BYTES)
                .map_err(|e| e.to_string())?;
            mcp::run_mcp_stdio(sbx)?;
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("biomolcraft-cli error: {msg}");
            std::process::ExitCode::FAILURE
        }
    }
}
