#![forbid(unsafe_code)]

//! `biomolcraft-cli` — Headless Biomolecular 3D Studio CLI, Craft Link v1 Client,
//! and `storytold/craft-libs::craft-mcp-server` MCP Server.

mod mcp;

use std::path::{Path, PathBuf};

use biomol_enzyme::{analyze_enzyme_engineering, render_enzyme_markdown};
use biomol_model::{
    audit_structure_prep, build_de_novo_peptide_3d, build_de_novo_rna_helix_3d, minimize_local_mm,
    mutate_and_repack_best_rotamer, mutate_residue_3d, swapna_nucleotide_3d,
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
use biomol_script::{
    BiomolSession, COMMAND_REGISTRY, fetch_structure_spec, filter_command_palette,
    inspect_session_zip, load_session_zip, render_registry_help, save_session_zip,
};
use clap::{Parser, Subcommand};
use craftbio_report::{export_artcraft_bundle, write_biomol_html_dossier};
use craftbio_sandbox::{DEFAULT_MAX_FILE_BYTES, PathSandbox, inspect_wasm_plugin};
use craftbio_scene::{
    ArtCraftTarget, CraftSceneDocument, DockCraftSceneBundle, discover_craft_instances,
    push_dockcraft_bundle_via_craft_link, push_scene_via_craft_link, send_dsl_via_craft_link,
    send_trajectory_via_craft_link,
};
use craftbio_struct::{
    cealign_superpose, detect_binding_pockets, detect_clashes, detect_contacts, detect_hbonds,
    detect_snfg_glycans, estimate_pae_proxy_from_structure, generate_biological_assembly,
    generate_crystal_mates, load_dcd_frames, matchmaker_align, morph_biomolecules, parse_dx_map,
    parse_mrc_map, parse_pae_json, parse_structure_auto, write_mrc_map, write_multi_model_pdb,
    write_pdb, write_xtc_frames,
};
use craftbio_viewport::{
    render_scene_png_bytes, render_scene_rgba, write_movie_file, write_scene_png,
};

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
    /// Render a publication-grade 3D biomolecular PNG headlessly (triangulated Cartoon/SES + SSAO + up to 4K).
    Render {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(
            long,
            default_value = "cartoon",
            help = "cartoon | putty | nucleic | sticks | spheres | surface | lines"
        )]
        style: String,
        #[arg(
            long,
            default_value = "chain",
            help = "chain | ss | cpk | nucleotide | plddt | sap | esp | mlp"
        )]
        color: String,
        #[arg(
            long,
            help = "Optional module overlay: antibodies | ncrna | enzymes | adc"
        )]
        module: Option<String>,
        #[arg(
            long,
            help = "Optional publication preset: pub_white | pub_dark | chimerax_soft | pymol_ray"
        )]
        preset: Option<String>,
        #[arg(
            long,
            help = "Optional Color Key legend: auto | esp | mlp | sap | plddt"
        )]
        key: Option<String>,
        #[arg(long, help = "Optional calibrated Ångström scale bar length (e.g. 20)")]
        scalebar: Option<f32>,
        #[arg(
            long,
            help = "Optional 2D title banner label burned into top-left of PNG"
        )]
        label_2d: Option<String>,
        #[arg(long)]
        surface: bool,
        #[arg(long)]
        wireframe: bool,
        #[arg(long, default_value_t = 0.60)]
        ssao: f32,
        #[arg(
            long,
            default_value_t = 0.55,
            help = "Directional PCF 3x3 soft shadow map strength (0.0 = off, 0.55 = default, 1.0 = deep)"
        )]
        shadows: f32,
        #[arg(long)]
        clip_near: Option<f32>,
        #[arg(long)]
        clip_far: Option<f32>,
        #[arg(long, default_value_t = 1600)]
        width: u32,
        #[arg(long, default_value_t = 1200)]
        height: u32,
        #[arg(long)]
        transparent: bool,
        #[arg(long)]
        out_png: PathBuf,
        #[arg(long)]
        out_glb: Option<PathBuf>,
        #[arg(long)]
        out_obj: Option<PathBuf>,
    },
    /// Fetch a biomolecular structure or map online (RCSB PDB, af:UNIPROT, emdb:ID, pubchem:CID) with local cache.
    Fetch {
        #[arg(long, help = "e.g. 1N8Z, pdb:1CRN, af:P04626, emdb:1234, pubchem:2244")]
        id: String,
        #[arg(long, help = "Optional destination path to copy the fetched file")]
        out: Option<PathBuf>,
    },
    /// Export triangulated 3D Cartoon / SES / Nucleic meshes to binary glTF 2.0 (.glb) or Wavefront (.obj).
    ExportMesh {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, default_value = "cartoon")]
        style: String,
        #[arg(long, default_value = "chain")]
        color: String,
        #[arg(long)]
        surface: bool,
        #[arg(long)]
        out: PathBuf,
    },
    /// Render or inspect a rich DockCraft ViewDockX bundle (receptor + pose + HYDE coronas + torsion lights + ΔMIF + GridBox).
    Viewdock {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, default_value_t = 0)]
        pose: usize,
        #[arg(long)]
        preset: Option<String>,
        #[arg(long, default_value_t = 1600)]
        width: u32,
        #[arg(long, default_value_t = 1200)]
        height: u32,
        #[arg(long)]
        out_png: Option<PathBuf>,
        #[arg(long)]
        push_live: bool,
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
    /// Chemical interaction & geometry audit (`hbonds`, `clashes`, `contacts`) with optional annotated PNG.
    Measure {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        hbonds: bool,
        #[arg(long)]
        clashes: bool,
        #[arg(long)]
        contacts: bool,
        #[arg(long, default_value_t = 3.5)]
        max_dist: f64,
        #[arg(long, default_value_t = 0.60)]
        clash_overlap: f64,
        #[arg(long)]
        putty: bool,
        #[arg(long)]
        key: Option<String>,
        #[arg(long)]
        scalebar: Option<f32>,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Sequence-independent 3D Combinatorial Extension alignment (`cealign` / `super`, 0% seq identity needed).
    Cealign {
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
    /// Conformational Morphing between two structures (`morph`, Kabsch-aligned + SHAKE bond restoration).
    Morph {
        #[arg(long)]
        from: PathBuf,
        #[arg(long)]
        to: PathBuf,
        #[arg(long, default_value_t = 20)]
        frames: usize,
        #[arg(long)]
        out_pdb: Option<PathBuf>,
        #[arg(long)]
        out_movie: Option<PathBuf>,
        #[arg(long, default_value_t = 640)]
        width: u32,
        #[arg(long, default_value_t = 480)]
        height: u32,
        #[arg(long, default_value_t = 12)]
        fps: u16,
    },
    /// Parse AlphaFold/Boltz-2/Chai-1 PAE JSON (or estimate structural PAE domain packing proxy).
    Pae {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        pae_json: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Export an animated biomolecular movie (pure-Rust `.gif` or ffmpeg H.264 `.mp4`/`.webm`).
    Movie {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, default_value = "turn", help = "turn | rock | fly | morph")]
        mode: String,
        #[arg(long, help = "Optional second conformation PDB when --mode morph")]
        to_pdb: Option<PathBuf>,
        #[arg(long, default_value = "cartoon")]
        style: String,
        #[arg(long, default_value = "chain")]
        color: String,
        #[arg(long)]
        preset: Option<String>,
        #[arg(long)]
        key: Option<String>,
        #[arg(long)]
        scalebar: Option<f32>,
        #[arg(long, default_value_t = 24)]
        frames: usize,
        #[arg(long, default_value_t = 640)]
        width: u32,
        #[arg(long, default_value_t = 480)]
        height: u32,
        #[arg(long, default_value_t = 12)]
        fps: u16,
        #[arg(long)]
        out: PathBuf,
    },
    /// Mutate an RNA/DNA nucleotide in 3D (`swapna`, preserving ribose-phosphate backbone & base plane).
    Swapna {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        chain: char,
        #[arg(long)]
        res: i32,
        #[arg(long)]
        base: char,
        #[arg(long)]
        out_pdb: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Build a de novo 3D polypeptide (`helix|sheet|extended`) or A-form RNA duplex (`rna`).
    Build {
        #[arg(long, default_value = "helix", help = "helix | sheet | extended | rna")]
        kind: String,
        #[arg(long)]
        seq: String,
        #[arg(long, default_value_t = 'A')]
        chain: char,
        #[arg(long)]
        out_pdb: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
    },
    /// Biological Assembly (`sym`) & Crystallographic Symmetry Mates (`symexp` + `CRYST1` Unit Cell).
    Sym {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(
            long,
            default_value = "symexp",
            help = "bioassembly | symexp | contacts"
        )]
        mode: String,
        #[arg(long, default_value_t = 5.0)]
        cutoff: f64,
        #[arg(long)]
        unitcell: bool,
        #[arg(long)]
        out_pdb: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
    },
    /// Detect & render 3D SNFG Glycans (GlcNAc cubes, Man/Gal spheres, Fuc cones, Neu5Ac diamonds).
    Glycans {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Volumetric Map Operations (`vop`: gaussian | sharpen | zone | hide_dust | orthoplanes).
    Vop {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long)]
        map: Option<PathBuf>,
        #[arg(
            long,
            default_value = "sharpen",
            help = "gaussian | sharpen | zone | hide_dust | orthoplanes"
        )]
        op: String,
        #[arg(long, default_value_t = 1.5)]
        param: f64,
        #[arg(long, default_value_t = 1.5)]
        sigma: f64,
        #[arg(long)]
        out_mrc: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
    },
    /// Save, inspect, or render a CRC-32 verified portable `.biomol.zip` session (PyMOL `.pse` / ChimeraX `.cxs`).
    Session {
        #[arg(long, default_value = "inspect", help = "save | inspect | render")]
        action: String,
        #[arg(long, help = "Path to .biomol.zip session archive")]
        session: PathBuf,
        #[arg(long, help = "Primary PDB/mmCIF when --action save")]
        pdb: Option<PathBuf>,
        #[arg(
            long,
            help = "Optional second PDB/mmCIF to add as #2 when --action save"
        )]
        add_pdb: Option<PathBuf>,
        #[arg(
            long,
            help = "Optional DSL commands to execute before saving or rendering"
        )]
        dsl: Option<String>,
        #[arg(long, help = "Optional PNG output when --action render")]
        out_png: Option<PathBuf>,
        #[arg(long, default_value_t = 1600)]
        width: u32,
        #[arg(long, default_value_t = 1200)]
        height: u32,
        #[arg(long)]
        json: bool,
    },
    /// Export full PNG + TSV + Markdown bundle & live-push over Craft Link v1 to WordCraft, DeckCraft, or GridCraft.
    PushArtcraft {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(
            long,
            default_value = "wordcraft",
            help = "wordcraft | deckcraft | gridcraft"
        )]
        target: String,
        #[arg(
            long,
            help = "Optional module overlay: antibodies | ncrna | enzymes | adc"
        )]
        module: Option<String>,
        #[arg(long, default_value = "/tmp/biomolcraft_artcraft_push")]
        out_dir: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Introspect the unified typed BiomolCraft Command Registry (`COMMAND_REGISTRY`).
    #[command(name = "commands", alias = "registry")]
    Registry {
        #[arg(long, help = "Optional keyword filter")]
        query: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Detect 3D druggable binding cavities (fpocket/CASTp-style), compute DockCraft GridBox & render cavity PNG.
    Pockets {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, default_value_t = 6)]
        max_pockets: usize,
        #[arg(long)]
        out_json: Option<PathBuf>,
        #[arg(long)]
        out_md: Option<PathBuf>,
        #[arg(long)]
        out_png: Option<PathBuf>,
        #[arg(long, default_value_t = 1600)]
        width: u32,
        #[arg(long, default_value_t = 1200)]
        height: u32,
    },
    /// Load, validate, or generate a pure-Rust GROMACS `.xtc` compressed trajectory (`3dfcoord`) and render frames/GIF.
    Xtc {
        #[arg(long)]
        pdb: PathBuf,
        #[arg(long, help = "Existing GROMACS .xtc trajectory to load")]
        xtc: Option<PathBuf>,
        #[arg(
            long,
            help = "Generate an ANM collective-mode .xtc trajectory, save to this path, and verify reload roundtrip"
        )]
        write_demo_xtc: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        frame: usize,
        #[arg(long)]
        out_png: Option<PathBuf>,
        #[arg(long)]
        out_gif: Option<PathBuf>,
        #[arg(long)]
        json: bool,
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
            preset,
            key,
            scalebar,
            label_2d,
            surface,
            wireframe,
            ssao,
            shadows,
            clip_near,
            clip_far,
            width,
            height,
            transparent,
            out_png,
            out_glb,
            out_obj,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let _ = sess
                .execute_dsl(&format!(
                    "style {style}; color {color}; ssao {ssao:.2}; shadows {shadows:.2}"
                ))
                .map_err(|e| e.to_string())?;
            if let Some(p) = preset {
                let _ = sess
                    .execute_dsl(&format!("preset {p}"))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(k) = key {
                let _ = sess
                    .execute_dsl(&format!("key {k}"))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(sb) = scalebar {
                let _ = sess
                    .execute_dsl(&format!("scalebar {sb:.1}"))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(lbl) = label_2d {
                let _ = sess
                    .execute_dsl(&format!("2dlabel 0.03 0.03 {lbl}"))
                    .map_err(|e| e.to_string())?;
            }
            if wireframe {
                let _ = sess
                    .execute_dsl("surface wireframe")
                    .map_err(|e| e.to_string())?;
            } else if surface {
                let _ = sess.execute_dsl("surface on").map_err(|e| e.to_string())?;
            }
            if let Some(m) = module {
                let _ = sess
                    .execute_dsl(&format!("module {m}"))
                    .map_err(|e| e.to_string())?;
            }
            if clip_near.is_some() || clip_far.is_some() {
                let cn = clip_near.unwrap_or(20.0);
                let cf = clip_far.unwrap_or(20.0);
                let _ = sess
                    .execute_dsl(&format!("clip {cn:.2} {cf:.2}"))
                    .map_err(|e| e.to_string())?;
            }
            let scene = sess.build_viewport_scene();
            let title = sess
                .active_molecule()
                .map_or_else(|| "BiomolCraft".to_string(), |m| m.title.clone());
            if let Some(glb_p) = out_glb {
                let glb = scene.export_glb(&title);
                if let Some(p) = glb_p.parent()
                    && !p.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
                }
                std::fs::write(&glb_p, &glb).map_err(|e| e.to_string())?;
                println!(
                    "Exported glTF 2.0 (.glb) -> {} ({} bytes)",
                    glb_p.display(),
                    glb.len()
                );
            }
            if let Some(obj_p) = out_obj {
                let obj = scene.export_obj(&title);
                write_text(&obj_p, &obj)?;
                println!(
                    "Exported Wavefront OBJ -> {} ({} bytes)",
                    obj_p.display(),
                    obj.len()
                );
            }
            let bytes = write_scene_png(&scene, &sess.camera, width, height, transparent, &out_png)
                .map_err(|e| e.to_string())?;
            println!(
                "Rendered {}x{} PNG to {} ({bytes} bytes, op_meshes={}, tr_meshes={}, spheres={}, capsules={})",
                width,
                height,
                out_png.display(),
                scene.opaque_meshes.len(),
                scene.transparent_meshes.len(),
                scene.spheres.len(),
                scene.capsules.len()
            );
        }
        Commands::Fetch { id, out } => {
            let asset = fetch_structure_spec(&id)?;
            if let Some(dest) = out {
                if let Some(p) = dest.parent()
                    && !p.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
                }
                std::fs::copy(&asset.local_path, &dest).map_err(|e| e.to_string())?;
                println!(
                    "Fetched {} {} ({}) -> {} (copied to {})",
                    asset.provider,
                    asset.canonical_id,
                    if asset.cached_hit {
                        "cache hit"
                    } else {
                        "downloaded"
                    },
                    asset.local_path.display(),
                    dest.display()
                );
            } else {
                println!(
                    "Fetched {} {} ({}) -> {}",
                    asset.provider,
                    asset.canonical_id,
                    if asset.cached_hit {
                        "cache hit"
                    } else {
                        "downloaded"
                    },
                    asset.local_path.display()
                );
            }
        }
        Commands::ExportMesh {
            pdb,
            style,
            color,
            surface,
            out,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let _ = sess
                .execute_dsl(&format!("style {style}; color {color}"))
                .map_err(|e| e.to_string())?;
            if surface {
                let _ = sess.execute_dsl("surface on").map_err(|e| e.to_string())?;
            }
            let msg = sess
                .execute_dsl(&format!("export_mesh {}", out.display()))
                .map_err(|e| e.to_string())?;
            println!("{msg}");
        }
        Commands::Viewdock {
            bundle,
            pose,
            preset,
            width,
            height,
            out_png,
            push_live,
        } => {
            let raw = std::fs::read_to_string(&bundle).map_err(|e| e.to_string())?;
            let parsed: DockCraftSceneBundle =
                serde_json::from_str(&raw).map_err(|e| format!("bad dock bundle JSON: {e}"))?;
            if push_live {
                let instances = discover_craft_instances(None, Some("biomolcraft"))
                    .map_err(|e| e.to_string())?;
                let target = instances
                    .first()
                    .ok_or("no running BiomolCraft instance on Craft Link v1")?;
                let rep = push_dockcraft_bundle_via_craft_link(target, &parsed)
                    .map_err(|e| e.to_string())?;
                println!("Pushed ViewDockX bundle over Craft Link v1: {rep}");
            }
            let mut sess = BiomolSession::default();
            let msg = sess
                .load_dockcraft_bundle(parsed)
                .map_err(|e| e.to_string())?;
            println!("{msg}");
            if pose > 0 {
                println!(
                    "{}",
                    sess.select_dock_pose(pose).map_err(|e| e.to_string())?
                );
            }
            if let Some(p) = preset {
                let _ = sess
                    .execute_dsl(&format!("preset {p}"))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(png_p) = out_png {
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, width, height, false, &png_p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered ViewDockX {}x{} PNG to {} ({bytes} bytes).",
                    width,
                    height,
                    png_p.display()
                );
            }
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
            let mol = sess.active_molecule().ok_or("load failed")?.clone();
            let vis = match name.to_ascii_lowercase().as_str() {
                "antibodies" | "ab" | "vhh" => run_antibody_module(&mol, None, None)?.visual,
                "ncrna" | "rna" | "mrna" => run_ncrna_module(&mol)?.0,
                "enzymes" | "enzyme" => run_enzyme_module(&mol),
                "adc" => run_adc_module(&mol, "ggfg", "dxd", 8.0, true)?.0,
                other => return Err(format!("unknown module '{other}'")),
            };
            if let Some(p) = out_md {
                write_text(&p, &vis.summary_markdown)?;
            }
            if let Some(p) = out_scene {
                sess.render_opts.residue_highlights = vis.residue_highlights;
                sess.render_opts.sphere_overlays = vis.sphere_overlays;
                sess.render_opts.line_overlays = vis.line_overlays;
                sess.sync_active_render_opts();
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
            let (vis, traj_txt) = {
                let mol = sess.active_molecule_mut().ok_or("no molecule")?;
                let (v, _) = run_physics_anm_module(mol, mode, amp, frames)?;
                let t = out_traj_pdb.as_ref().map(|_| write_multi_model_pdb(mol));
                (v, t)
            };
            if let Some(p) = out_md {
                write_text(&p, &vis.summary_markdown)?;
            }
            if let (Some(p), Some(t)) = (out_traj_pdb, traj_txt) {
                write_text(&p, &t)?;
            }
            sess.render_opts.residue_highlights = vis.residue_highlights;
            sess.render_opts.line_overlays = vis.line_overlays;
            sess.sync_active_render_opts();
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
            let mol = sess.active_molecule().ok_or("no molecule")?.clone();
            let (vis, rep) = run_enzyme_engineering_module(&mol);
            if let Some(p) = out_md {
                write_text(&p, &vis.summary_markdown)?;
            }
            if let Some(p) = out_png {
                sess.render_opts.residue_highlights = vis.residue_highlights;
                sess.render_opts.sphere_overlays = vis.sphere_overlays;
                sess.render_opts.line_overlays = vis.line_overlays;
                sess.sync_active_render_opts();
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
                let mol = sess.active_molecule_mut().ok_or("no molecule")?;
                let (v, _) = run_cryoem_volume_module(mol, &g, sigma, fit);
                (v, g)
            } else {
                let mol = sess.active_molecule_mut().ok_or("no molecule")?;
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
                sess.active_volume = Some(grid);
                sess.active_volume_sigma = sigma;
                sess.dirty_geometry = true;
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered Cryo-EM Marching-Tetrahedra volumetric PNG to {} ({bytes} bytes).",
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
            let mol = sess.active_molecule().ok_or("no molecule")?.clone();
            let (vis, _) = run_denovo_cofold_module(&mol, chain, &hs, out_dir.as_deref())?;
            if let Some(p) = out_png {
                sess.render_opts.residue_highlights = vis.residue_highlights;
                sess.render_opts.sphere_overlays = vis.sphere_overlays;
                sess.sync_active_render_opts();
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
                        && let Some(mol) = sess.active_molecule()
                        && let Ok((_, reps)) = run_ncrna_module(mol)
                        && let Some(first) = reps.first()
                    {
                        extra_svg = Some(render_rna_arc_svg(&first.name, &first.fold));
                    }
                }
            }
            let mol = sess.active_molecule().ok_or("no molecule")?;
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
                .active_molecule()
                .map_or_else(|| "BiomolCraft".to_string(), |m| m.title.clone());
            let atoms = sess.active_molecule().map_or(0, |m| m.atoms.len());
            let chains = sess.active_molecule().map_or(0, |m| m.chains.len());
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
        Commands::Measure {
            pdb,
            hbonds,
            clashes,
            contacts,
            max_dist,
            clash_overlap,
            putty,
            key,
            scalebar,
            out_md,
            out_png,
            json,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let mol = sess.active_molecule().ok_or("no molecule")?.clone();
            let do_all = !hbonds && !clashes && !contacts;
            let hb_vec = if do_all || hbonds {
                let _ = sess
                    .execute_dsl(&format!("hbonds {max_dist:.2}"))
                    .map_err(|e| e.to_string())?;
                detect_hbonds(&mol, None, None, max_dist, 95.0)
            } else {
                Vec::new()
            };
            let cl_vec = if do_all || clashes {
                let _ = sess
                    .execute_dsl(&format!("clashes {clash_overlap:.2}"))
                    .map_err(|e| e.to_string())?;
                detect_clashes(&mol, None, None, clash_overlap)
            } else {
                Vec::new()
            };
            let ct_vec = if do_all || contacts {
                let _ = sess
                    .execute_dsl(&format!("contacts {:.2}", max_dist.max(4.0)))
                    .map_err(|e| e.to_string())?;
                detect_contacts(&mol, None, None, max_dist.max(4.0), 200)
            } else {
                Vec::new()
            };
            if putty {
                let _ = sess.execute_dsl("putty on").map_err(|e| e.to_string())?;
            }
            if let Some(k) = key {
                let _ = sess
                    .execute_dsl(&format!("key {k}"))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(sb) = scalebar {
                let _ = sess
                    .execute_dsl(&format!("scalebar {sb:.1}"))
                    .map_err(|e| e.to_string())?;
            }
            let summary_md = format!(
                "# BiomolCraft Chemical Measurement Audit: {}\n\n- **H-Bonds (<= {:.2} Å):** `{}`\n- **Steric Clashes (overlap >= {:.2} Å):** `{}`\n- **Inter-chain Contacts (<= {:.2} Å):** `{}`\n",
                mol.title,
                max_dist,
                hb_vec.len(),
                clash_overlap,
                cl_vec.len(),
                max_dist.max(4.0),
                ct_vec.len()
            );
            if let Some(p) = out_md {
                write_text(&p, &summary_md)?;
            }
            if let Some(p) = out_png {
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered annotated measurement PNG -> {} ({bytes} bytes).",
                    p.display()
                );
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "structure": mol.title,
                        "hbonds_count": hb_vec.len(),
                        "clashes_count": cl_vec.len(),
                        "contacts_count": ct_vec.len(),
                        "top_hbonds": hb_vec.iter().take(15).collect::<Vec<_>>(),
                        "top_clashes": cl_vec.iter().take(15).collect::<Vec<_>>(),
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                print!("{summary_md}");
            }
        }
        Commands::Cealign {
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
            let tf = cealign_superpose(&mut m_mol, mobile_chain, &t_mol, target_chain)
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
                    "CEAlign (sequence-independent) {mobile_chain} -> {target_chain}: RMSD = {:.4} Å ({} core guide atoms).",
                    tf.rmsd_a, tf.aligned_pairs
                );
            }
        }
        Commands::Morph {
            from,
            to,
            frames,
            out_pdb,
            out_movie,
            width,
            height,
            fps,
        } => {
            let raw_a = std::fs::read_to_string(&from).map_err(|e| e.to_string())?;
            let raw_b = std::fs::read_to_string(&to).map_err(|e| e.to_string())?;
            let mol_a = parse_structure_auto("from", &raw_a).map_err(|e| e.to_string())?;
            let mol_b = parse_structure_auto("to", &raw_b).map_err(|e| e.to_string())?;
            let morphed = morph_biomolecules(&mol_a, &mol_b, frames).map_err(|e| e.to_string())?;
            if let Some(p) = out_pdb {
                write_text(&p, &write_multi_model_pdb(&morphed))?;
                println!(
                    "Wrote {}-frame conformational morph PDB -> {}",
                    morphed.frames.len(),
                    p.display()
                );
            }
            if let Some(mp) = out_movie {
                let mut sess = BiomolSession::default();
                sess.load_structure_file(&from).map_err(|e| e.to_string())?;
                sess.add_structure_file(&to).map_err(|e| e.to_string())?;
                let _ = sess
                    .execute_dsl(&format!("morph #1 #2 {frames}"))
                    .map_err(|e| e.to_string())?;
                let msg = sess
                    .execute_dsl(&format!(
                        "movie morph {} {frames} {width} {height} {fps}",
                        mp.display()
                    ))
                    .map_err(|e| e.to_string())?;
                println!("{msg}");
            } else {
                println!(
                    "Generated {} SHAKE-relaxed conformational morph frames ({} atoms).",
                    morphed.frames.len(),
                    morphed.atoms.len()
                );
            }
        }
        Commands::Pae {
            pdb,
            pae_json,
            json,
        } => {
            let raw = std::fs::read_to_string(&pdb).map_err(|e| e.to_string())?;
            let id = pdb.file_stem().and_then(|s| s.to_str()).unwrap_or("mol");
            let mol = parse_structure_auto(id, &raw).map_err(|e| e.to_string())?;
            let pae = if let Some(jp) = pae_json {
                let jtxt = std::fs::read_to_string(&jp).map_err(|e| e.to_string())?;
                parse_pae_json(&jp.display().to_string(), &jtxt).map_err(|e| e.to_string())?
            } else {
                estimate_pae_proxy_from_structure(&mol)
            };
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "label": pae.label,
                        "length": pae.length,
                        "mean_pae": pae.mean_pae,
                        "max_pae": pae.max_pae,
                        "is_experimental_json": pae.is_experimental_json,
                        "domain_blocks": pae.domain_blocks,
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "PAE '{}': {}x{}, mean={:.2} Å, max={:.2} Å, experimental_json={}, rigid domains={:?}",
                    pae.label,
                    pae.length,
                    pae.length,
                    pae.mean_pae,
                    pae.max_pae,
                    pae.is_experimental_json,
                    pae.domain_blocks
                );
            }
        }
        Commands::Movie {
            pdb,
            mode,
            to_pdb,
            style,
            color,
            preset,
            key,
            scalebar,
            frames,
            width,
            height,
            fps,
            out,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            if let Some(tp) = to_pdb {
                sess.add_structure_file(&tp).map_err(|e| e.to_string())?;
                if mode.eq_ignore_ascii_case("morph") {
                    let _ = sess
                        .execute_dsl(&format!("morph #1 #2 {frames}"))
                        .map_err(|e| e.to_string())?;
                }
            }
            let _ = sess
                .execute_dsl(&format!("style {style}; color {color}"))
                .map_err(|e| e.to_string())?;
            if let Some(p) = preset {
                let _ = sess
                    .execute_dsl(&format!("preset {p}"))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(k) = key {
                let _ = sess
                    .execute_dsl(&format!("key {k}"))
                    .map_err(|e| e.to_string())?;
            }
            if let Some(sb) = scalebar {
                let _ = sess
                    .execute_dsl(&format!("scalebar {sb:.1}"))
                    .map_err(|e| e.to_string())?;
            }
            let msg = sess
                .execute_dsl(&format!(
                    "movie {mode} {} {frames} {width} {height} {fps}",
                    out.display()
                ))
                .map_err(|e| e.to_string())?;
            println!("{msg}");
        }
        Commands::Swapna {
            pdb,
            chain,
            res,
            base,
            out_pdb,
            json,
        } => {
            let raw = std::fs::read_to_string(&pdb).map_err(|e| e.to_string())?;
            let mut mol = parse_structure_auto("rna", &raw).map_err(|e| e.to_string())?;
            let rep =
                swapna_nucleotide_3d(&mut mol, chain, res, base).map_err(|e| e.to_string())?;
            if let Some(p) = out_pdb {
                write_text(&p, &write_pdb(&mol))?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "swapna {chain}:{}{res}{} ({}): {} clashes, {} H-bonds, {} 3D base pairs.",
                    rep.wild_type,
                    rep.mutant,
                    if rep.is_dna { "DNA" } else { "RNA" },
                    rep.clash_count,
                    rep.hbond_count,
                    rep.base_pairs_after
                );
            }
        }
        Commands::Build {
            kind,
            seq,
            chain,
            out_pdb,
            out_png,
        } => {
            let mol = if kind.eq_ignore_ascii_case("rna") || kind.eq_ignore_ascii_case("duplex") {
                build_de_novo_rna_helix_3d(&seq, chain)
            } else {
                build_de_novo_peptide_3d(&seq, &kind, chain)
            }
            .map_err(|e| e.to_string())?;
            let pdb_txt = write_pdb(&mol);
            if let Some(p) = out_pdb {
                write_text(&p, &pdb_txt)?;
                println!("Wrote built 3D '{kind}' PDB -> {}", p.display());
            }
            if let Some(p) = out_png {
                let mut sess = BiomolSession::default();
                let _ = sess.load_structure_text(&mol.id, &pdb_txt, None);
                let _ = sess.execute_dsl("preset pub_white; scalebar 15");
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered built 3D structure PNG -> {} ({bytes} bytes).",
                    p.display()
                );
            } else {
                println!(
                    "Built 3D '{}' ({} atoms, {} chains, {} 3D RNA base pairs).",
                    mol.title,
                    mol.atoms.len(),
                    mol.chains.len(),
                    mol.base_pairs.len()
                );
            }
        }
        Commands::Sym {
            pdb,
            mode,
            cutoff,
            unitcell,
            out_pdb,
            out_png,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let mol = sess.active_molecule().ok_or("no molecule")?.clone();
            match mode.to_ascii_lowercase().as_str() {
                "bioassembly" | "sym" => {
                    let bio = generate_biological_assembly(&mol).map_err(|e| e.to_string())?;
                    println!(
                        "Biological Assembly: {} atoms across {} chains ({} BIOMT ops).",
                        bio.atoms.len(),
                        bio.chains.len(),
                        mol.biomt_ops.len()
                    );
                    if let Some(p) = out_pdb {
                        write_text(&p, &write_pdb(&bio))?;
                    }
                    let _ = sess.execute_dsl("sym");
                }
                _ => {
                    let (mates, contacts) =
                        generate_crystal_mates(&mol, cutoff).map_err(|e| e.to_string())?;
                    println!(
                        "Crystal Symmetry ({:?}): {} mate(s) within {cutoff:.1} Å ({contacts} sampled contact pairs <= 4.0 Å).",
                        mol.unit_cell.as_ref().map(|u| u.space_group.as_str()),
                        mates.len()
                    );
                    if let Some(p) = out_pdb
                        && let Some((_, first_mate)) = mates.first()
                    {
                        write_text(&p, &write_pdb(first_mate))?;
                    }
                    let _ = sess.execute_dsl(&format!("symexp {cutoff:.1}"));
                }
            }
            if unitcell {
                let _ = sess.execute_dsl("unitcell on");
            }
            if let Some(p) = out_png {
                let _ = sess.execute_dsl("preset pub_dark; scalebar 25");
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!("Rendered Symmetry PNG -> {} ({bytes} bytes).", p.display());
            }
        }
        Commands::Glycans {
            pdb,
            out_md,
            out_png,
            json,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let mol = sess.active_molecule().ok_or("no molecule")?.clone();
            let rep = detect_snfg_glycans(&mol);
            let _ = sess.execute_dsl("preset pub_white; snfg on; scalebar 25");
            if let Some(p) = out_md {
                write_text(&p, &sess.last_module_markdown)?;
            }
            if let Some(p) = out_png {
                let _ = sess.execute_dsl("2dlabel 0.03 0.03 SNFG 3D Glycans");
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered SNFG 3D Glycans PNG -> {} ({bytes} bytes).",
                    p.display()
                );
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", sess.last_module_markdown);
            }
        }
        Commands::Vop {
            pdb,
            map,
            op,
            param,
            sigma,
            out_mrc,
            out_png,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            if let Some(mp) = map {
                let _ = sess
                    .execute_dsl(&format!("open_map {} {sigma:.2}", mp.display()))
                    .map_err(|e| e.to_string())?;
            } else {
                let _ = sess
                    .execute_dsl(&format!("molmap 4.5 {sigma:.2}"))
                    .map_err(|e| e.to_string())?;
            }
            let vop_cmd = if op.eq_ignore_ascii_case("orthoplanes") {
                "vop orthoplanes on".to_string()
            } else {
                format!("vop {op} {param:.2}")
            };
            let msg = sess.execute_dsl(&vop_cmd).map_err(|e| e.to_string())?;
            println!("{msg}");
            if let Some(p) = out_mrc
                && let Some(ref g) = sess.active_volume
            {
                if let Some(parent) = p.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                std::fs::write(&p, write_mrc_map(g)).map_err(|e| e.to_string())?;
                println!("Wrote filtered MRC2014 volume -> {}", p.display());
            }
            if let Some(p) = out_png {
                let _ = sess.execute_dsl("preset pub_dark; scalebar 15");
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1600, 1200, false, &p)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered Volumetric VOP PNG -> {} ({bytes} bytes).",
                    p.display()
                );
            }
        }
        Commands::Session {
            action,
            session,
            pdb,
            add_pdb,
            dsl,
            out_png,
            width,
            height,
            json,
        } => match action.to_ascii_lowercase().as_str() {
            "save" => {
                let mut sess = BiomolSession::default();
                if let Some(p) = pdb {
                    sess.load_structure_file(&p).map_err(|e| e.to_string())?;
                }
                if let Some(p2) = add_pdb {
                    sess.add_structure_file(&p2).map_err(|e| e.to_string())?;
                }
                if let Some(script) = dsl {
                    let _ = sess.execute_dsl(&script).map_err(|e| e.to_string())?;
                }
                sess.sync_active_render_opts();
                let sum = save_session_zip(&sess, &session).map_err(|e| e.to_string())?;
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&sum).map_err(|e| e.to_string())?
                    );
                } else {
                    println!(
                        "Saved verified portable session {} ({} bytes, {} models, {} atoms, {} scenes, volume={}, viewdock={}).",
                        session.display(),
                        sum.archive_bytes,
                        sum.model_count,
                        sum.total_atoms,
                        sum.scene_count,
                        sum.has_volume_mrc,
                        sum.has_viewdock
                    );
                }
            }
            "render" => {
                let (mut sess, sum) = load_session_zip(&session).map_err(|e| e.to_string())?;
                if let Some(script) = dsl {
                    let _ = sess.execute_dsl(&script).map_err(|e| e.to_string())?;
                }
                let png_path = out_png.ok_or("`session --action render` requires --out-png")?;
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, width, height, false, &png_path)
                    .map_err(|e| e.to_string())?;
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "session": sum,
                            "out_png": png_path.display().to_string(),
                            "png_bytes": bytes,
                        }))
                        .map_err(|e| e.to_string())?
                    );
                } else {
                    println!(
                        "Reloaded verified session {} ({} models, {} scenes) & rendered {}x{} PNG -> {} ({bytes} bytes).",
                        session.display(),
                        sum.model_count,
                        sum.scene_count,
                        width,
                        height,
                        png_path.display()
                    );
                }
            }
            _ => {
                let sum = inspect_session_zip(&session).map_err(|e| e.to_string())?;
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&sum).map_err(|e| e.to_string())?
                    );
                } else {
                    println!(
                        "# BiomolCraft Verified Portable Session: {}\n",
                        session.display()
                    );
                    println!(
                        "- **Schema:** `{}` (BiomolCraft `v{}`) | **Archive Size:** `{} bytes`",
                        sum.schema_version, sum.app_version, sum.archive_bytes
                    );
                    println!(
                        "- **Models ({}):** `{:?}` (`{}` total atoms)",
                        sum.model_count, sum.model_names, sum.total_atoms
                    );
                    println!(
                        "- **Saved Scenes:** `{}` | **3D/2D Labels:** `{}/{}` | **Color Key:** `{}` | **Scale Bar:** `{}`",
                        sum.scene_count,
                        sum.label_3d_count,
                        sum.label_2d_count,
                        sum.has_color_key,
                        sum.has_scale_bar
                    );
                    println!(
                        "- **Cryo-EM Volume (.mrc):** `{}` | **ViewDockX:** `{}` ({} poses) | **Provenance Entries:** `{}`",
                        sum.has_volume_mrc,
                        sum.has_viewdock,
                        sum.viewdock_poses,
                        sum.provenance_entries
                    );
                }
            }
        },
        Commands::PushArtcraft {
            pdb,
            target,
            module,
            out_dir,
            json,
        } => {
            let tgt = ArtCraftTarget::parse(&target).ok_or_else(|| {
                format!("unknown ArtCraft target '{target}' (use wordcraft|deckcraft|gridcraft)")
            })?;
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let _ = sess.execute_dsl("preset pub_white; scalebar 20");
            if let Some(m) = module {
                let _ = sess
                    .execute_dsl(&format!("module {m}"))
                    .map_err(|e| e.to_string())?;
            }
            let res = sess
                .push_artcraft_target(tgt, &out_dir)
                .map_err(|e| e.to_string())?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&res).map_err(|e| e.to_string())?
                );
            } else {
                println!("{}", res.message);
            }
        }
        Commands::Registry { query, json } => {
            let q = query.as_deref().unwrap_or("");
            if json {
                let specs = if q.is_empty() {
                    COMMAND_REGISTRY.iter().collect::<Vec<_>>()
                } else {
                    filter_command_palette(q)
                };
                println!(
                    "{}",
                    serde_json::to_string_pretty(&specs).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", render_registry_help(q));
            }
        }
        Commands::Pockets {
            pdb,
            max_pockets,
            out_json,
            out_md,
            out_png,
            width,
            height,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let msg = sess
                .detect_pockets(max_pockets)
                .map_err(|e| e.to_string())?;
            println!("{msg}");
            if let Some(jp) = out_json {
                let json_str = serde_json::to_string_pretty(&sess.binding_pockets)
                    .map_err(|e| e.to_string())?;
                write_text(&jp, &json_str)?;
                println!(
                    "Wrote {} binding pocket(s) JSON -> {}",
                    sess.binding_pockets.len(),
                    jp.display()
                );
            }
            if let Some(mp) = out_md {
                write_text(&mp, &sess.last_module_markdown)?;
                println!("Wrote binding pockets Markdown -> {}", mp.display());
            }
            if let Some(pp) = out_png {
                let _ = sess.execute_dsl("preset pub_dark; shadows 0.75; scalebar 15");
                if !sess.binding_pockets.is_empty() {
                    let _ = sess.select_pocket_gridbox(1);
                    // Pull camera gently back so both the receptor cartoon and full golden GridBox fit cleanly
                    sess.camera.distance = (sess.camera.distance * 1.45).max(46.0);
                    let top = &sess.binding_pockets[0];
                    let _ = sess.execute_dsl(&format!(
                        "2dlabel 0.03 0.03 Pocket #1: Drug={:.2} | Vol={:.0} A3 | Enc={:.0}%",
                        top.druggability_score,
                        top.volume_a3,
                        top.enclosure * 100.0
                    ));
                }
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, width, height, false, &pp)
                    .map_err(|e| e.to_string())?;
                println!(
                    "Rendered 3D Binding Pockets + GridBox {}x{} PNG -> {} ({bytes} bytes).",
                    width,
                    height,
                    pp.display()
                );
            } else {
                let _ = detect_binding_pockets; // keep explicit import used
            }
        }
        Commands::Xtc {
            pdb,
            xtc,
            write_demo_xtc,
            frame,
            out_png,
            out_gif,
            json,
        } => {
            let mut sess = BiomolSession::default();
            sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
            let xtc_path_loaded = if let Some(demo_out) = write_demo_xtc {
                let _ = sess.execute_dsl("anm 1 2.4").map_err(|e| e.to_string())?;
                let mol = sess.active_molecule().ok_or("no active molecule")?;
                let xtc_bytes = write_xtc_frames(mol, 1000.0).map_err(|e| e.to_string())?;
                if let Some(parent) = demo_out.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                std::fs::write(&demo_out, &xtc_bytes).map_err(|e| e.to_string())?;
                // Reload freshly encoded XTC onto clean structure to verify lossless roundtrip
                sess.load_structure_file(&pdb).map_err(|e| e.to_string())?;
                let raw_rt = std::fs::read(&demo_out).map_err(|e| e.to_string())?;
                let msg = sess
                    .load_xtc_bytes(&raw_rt, &demo_out.display().to_string())
                    .map_err(|e| e.to_string())?;
                if !json {
                    println!(
                        "Wrote pure-Rust GROMACS .xtc ({} bytes) -> {}\n{msg}",
                        xtc_bytes.len(),
                        demo_out.display()
                    );
                }
                demo_out
            } else if let Some(xtc_p) = xtc {
                let raw = std::fs::read(&xtc_p).map_err(|e| e.to_string())?;
                let msg = sess
                    .load_xtc_bytes(&raw, &xtc_p.display().to_string())
                    .map_err(|e| e.to_string())?;
                if !json {
                    println!("{msg}");
                }
                xtc_p
            } else {
                return Err(
                    "provide either --xtc <in.xtc> or --write-demo-xtc <out.xtc>".to_string(),
                );
            };

            let n_frames = sess.active_molecule().map_or(0, |m| m.frames.len());
            let natoms = sess.active_molecule().map_or(0, |m| m.atoms.len());
            let t_end = sess
                .active_molecule()
                .and_then(|m| m.frames.last())
                .map_or(0.0, |f| f.time_ps);
            if frame < n_frames
                && let Some(mol) = sess.active_molecule_mut()
            {
                mol.set_active_frame(frame);
            }
            if let Some(pp) = out_png {
                let _ = sess.execute_dsl("preset pub_dark; shadows 0.65; scalebar 15");
                let sc = sess.build_viewport_scene();
                let bytes = write_scene_png(&sc, &sess.camera, 1280, 960, false, &pp)
                    .map_err(|e| e.to_string())?;
                if !json {
                    println!(
                        "Rendered XTC frame {}/{} -> {} ({bytes} bytes).",
                        frame + 1,
                        n_frames,
                        pp.display()
                    );
                }
            }
            if let Some(gp) = out_gif {
                let _ = sess.execute_dsl("preset pub_dark; shadows 0.60");
                let mut frames_rgba = Vec::with_capacity(n_frames);
                for fi in 0..n_frames {
                    if let Some(mol) = sess.active_molecule_mut() {
                        mol.set_active_frame(fi);
                    }
                    let sc = sess.build_viewport_scene();
                    frames_rgba.push(
                        render_scene_rgba(&sc, &sess.camera, 480, 360, false)
                            .map_err(|e| e.to_string())?,
                    );
                }
                let bytes =
                    write_movie_file(&gp, 480, 360, 12, &frames_rgba).map_err(|e| e.to_string())?;
                if !json {
                    println!(
                        "Rendered XTC animated GIF ({n_frames} frames) -> {} ({bytes} bytes).",
                        gp.display()
                    );
                }
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "xtc_path": xtc_path_loaded.display().to_string(),
                        "atoms": natoms,
                        "frames": n_frames,
                        "duration_ps": t_end,
                    }))
                    .map_err(|e| e.to_string())?
                );
            }
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
