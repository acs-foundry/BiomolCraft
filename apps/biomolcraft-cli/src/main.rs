#![forbid(unsafe_code)]

//! `biomolcraft-cli` — Headless Biomolecular 3D Studio CLI, Craft Link v1 Client,
//! and `storytold/craft-libs::craft-mcp-server` MCP Server.

mod mcp;

use std::path::{Path, PathBuf};

use biomol_model::{
    audit_structure_prep, minimize_local_mm, mutate_and_repack_best_rotamer, mutate_residue_3d,
};
use biomol_modules::{run_adc_module, run_antibody_module, run_enzyme_module, run_ncrna_module};
use biomol_rna::{
    analyze_ncrna_mrna, find_rna_preset, render_rna_arc_svg, render_rna_markdown, rna_presets,
};
use biomol_script::BiomolSession;
use clap::{Parser, Subcommand};
use craftbio_sandbox::{DEFAULT_MAX_FILE_BYTES, PathSandbox};
use craftbio_scene::{
    CraftSceneDocument, discover_craft_instances, push_scene_via_craft_link,
    send_dsl_via_craft_link,
};
use craftbio_struct::{matchmaker_align, parse_structure_auto, write_pdb};
use craftbio_viewport::write_scene_png;

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
    /// Discover running CraftBio/ArtCraft apps via Craft Link v1 (`craft-ipc`) and optionally push a PDB/DSL command.
    Link {
        #[arg(long, default_value = "biomolcraft")]
        app: String,
        #[arg(long)]
        push_pdb: Option<PathBuf>,
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
        Commands::Link { app, push_pdb, dsl } => {
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
            if let Some(p) = push_pdb {
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
