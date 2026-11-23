#![forbid(unsafe_code)]

//! Unified Typed Command Registry (`storytold/craft-anything` Single Source of Truth).
//!
//! Drives the interactive DSL console (`help`, `[TAB]` autocomplete, suggestion chips),
//! the GUI `Ctrl+K` Command Palette, CLI `commands` introspection, and MCP documentation
//! from one declarative table (`COMMAND_REGISTRY`).

use serde::Serialize;

/// Declarative specification of a BiomolCraft DSL command.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub category: &'static str,
    pub syntax: &'static str,
    pub description: &'static str,
    pub subcommands: &'static [&'static str],
}

/// Autocomplete candidate returned for a partial DSL console input.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionItem {
    pub insert_text: String,
    pub display_label: String,
    pub category: &'static str,
    pub description: &'static str,
}

/// Complete declarative BiomolCraft v1.0.0 DSL command table.
pub const COMMAND_REGISTRY: &[CommandSpec] = &[
    // --- File, Models & Portable Sessions ---
    CommandSpec {
        name: "open",
        aliases: &["load"],
        category: "File & Sessions",
        syntax: "open <path.pdb|cif|sdf|biomol.zip> [add|replace]",
        description: "Open a structure file (or portable .biomol.zip session) into the multi-model workspace.",
        subcommands: &["add", "replace"],
    },
    CommandSpec {
        name: "fetch",
        aliases: &[],
        category: "File & Sessions",
        syntax: "fetch <1N8Z|af:UNIPROT|emdb:ID|pubchem:CID>",
        description: "Fetch PDB, AlphaFoldDB, EMDB volume, or PubChem compound with local disk caching.",
        subcommands: &["1N8Z", "5JDS", "1Y26", "1CRN", "af:", "emdb:", "pubchem:"],
    },
    CommandSpec {
        name: "save_session",
        aliases: &["pse", "cxs"],
        category: "File & Sessions",
        syntax: "save_session <out.biomol.zip>",
        description: "Save complete verified portable session (.biomol.zip: all models, scenes, labels, volume & ViewDockX).",
        subcommands: &[],
    },
    CommandSpec {
        name: "load_session",
        aliases: &["open_session"],
        category: "File & Sessions",
        syntax: "load_session <in.biomol.zip>",
        description: "Load and CRC-32 verify a portable .biomol.zip session with bomb & traversal guards.",
        subcommands: &[],
    },
    CommandSpec {
        name: "save",
        aliases: &[],
        category: "File & Sessions",
        syntax: "save <out.pdb|out.cif|out.craftscene.json|out.biomol.zip>",
        description: "Export active structure (.pdb/.cif), CraftScene JSON, or portable .biomol.zip session.",
        subcommands: &[],
    },
    CommandSpec {
        name: "models",
        aliases: &[],
        category: "File & Sessions",
        syntax: "models",
        description: "List all loaded models (#1..#N) with visibility, style, and atom counts.",
        subcommands: &[],
    },
    CommandSpec {
        name: "active",
        aliases: &[],
        category: "File & Sessions",
        syntax: "active <#id>",
        description: "Switch the active target model in the multi-model workspace.",
        subcommands: &["#1", "#2", "#3"],
    },
    CommandSpec {
        name: "show",
        aliases: &["style"],
        category: "Visual & Rendering",
        syntax: "show <#id | cartoon|putty|nucleic|sticks|spheres|surface|lines>",
        description: "Show a hidden model or switch 3D molecular representation.",
        subcommands: &[
            "cartoon", "putty", "nucleic", "sticks", "spheres", "surface", "lines",
        ],
    },
    CommandSpec {
        name: "hide",
        aliases: &[],
        category: "File & Sessions",
        syntax: "hide <#id>",
        description: "Hide model #id without unloading it from memory.",
        subcommands: &["#1", "#2", "#3"],
    },
    CommandSpec {
        name: "close",
        aliases: &[],
        category: "File & Sessions",
        syntax: "close <#id|all>",
        description: "Unload a model (or `all` models) from the workspace.",
        subcommands: &["all", "#1", "#2"],
    },
    // --- Visual, Lighting, Scenes & Overlays ---
    CommandSpec {
        name: "preset",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "preset <pub_white|pub_dark|chimerax_soft|pymol_ray>",
        description: "Apply a 1-click publication lighting, SSAO, silhouette & background preset.",
        subcommands: &["pub_white", "pub_dark", "chimerax_soft", "pymol_ray"],
    },
    CommandSpec {
        name: "color",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "color <chain|ss|cpk|nucleotide|plddt|sap|esp|mlp>",
        description: "Color the active model by chain, DSSP, CPK, nucleotide, pLDDT, SAP, ESP, or MLP.",
        subcommands: &[
            "chain",
            "ss",
            "cpk",
            "nucleotide",
            "plddt",
            "sap",
            "esp",
            "mlp",
        ],
    },
    CommandSpec {
        name: "surface",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "surface <on|off|wireframe>",
        description: "Toggle triangulated Marching-Tetrahedra solvent-excluded molecular surface (SES).",
        subcommands: &["on", "off", "wireframe"],
    },
    CommandSpec {
        name: "putty",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "putty <on|off>",
        description: "Toggle variable-radius B-factor / pLDDT Putty sausage ribbon.",
        subcommands: &["on", "off"],
    },
    CommandSpec {
        name: "lighting",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "lighting <soft|flat|full|studio|pub_white>",
        description: "Set viewport directional/ambient lighting model.",
        subcommands: &["soft", "flat", "full", "studio", "pub_white"],
    },
    CommandSpec {
        name: "ssao",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "ssao <0.0..1.0>",
        description: "Adjust Screen-Space Ambient Occlusion depth shading intensity.",
        subcommands: &["0.0", "0.55", "0.75"],
    },
    CommandSpec {
        name: "bg",
        aliases: &["background"],
        category: "Visual & Rendering",
        syntax: "bg <dark|white|black>",
        description: "Set 3D viewport background color.",
        subcommands: &["dark", "white", "black"],
    },
    CommandSpec {
        name: "silhouettes",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "silhouettes <on|off>",
        description: "Toggle crisp publication outline silhouettes.",
        subcommands: &["on", "off"],
    },
    CommandSpec {
        name: "waters",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "waters <on|off>",
        description: "Toggle crystallographic water molecules.",
        subcommands: &["on", "off"],
    },
    CommandSpec {
        name: "clip",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "clip <near_a> <far_a> | clip reset",
        description: "Set front/back camera Z-slab clipping planes (or `clip reset`).",
        subcommands: &["reset", "off", "10 10"],
    },
    CommandSpec {
        name: "scene",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "scene <save|recall|list|delete> [name|f1..f12]",
        description: "Save, recall, list, or delete F1-F12 multi-model camera & style snapshots.",
        subcommands: &["save", "recall", "list", "delete"],
    },
    CommandSpec {
        name: "key",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "key <auto|esp|mlp|sap|plddt|off>",
        description: "Display a calibrated publication Color Key gradient legend (GUI + 4K PNG).",
        subcommands: &["auto", "esp", "mlp", "sap", "plddt", "off"],
    },
    CommandSpec {
        name: "scalebar",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "scalebar <length_a | off>",
        description: "Show a camera-calibrated Ångström scale bar (GUI + 4K PNG).",
        subcommands: &["10", "20", "25", "off"],
    },
    CommandSpec {
        name: "label",
        aliases: &[],
        category: "Visual & Rendering",
        syntax: "label <C:RES[:ATOM]> <text> | label clear",
        description: "Attach a projected 3D callout label to a residue/atom (or `label clear`).",
        subcommands: &["clear"],
    },
    CommandSpec {
        name: "2dlabel",
        aliases: &["2dlabels"],
        category: "Visual & Rendering",
        syntax: "2dlabel <norm_x> <norm_y> <text> | 2dlabel clear",
        description: "Place a 2D screen-space title banner burned into GUI & headless 4K PNGs.",
        subcommands: &["clear", "0.03 0.03"],
    },
    CommandSpec {
        name: "reset",
        aliases: &["center"],
        category: "Visual & Rendering",
        syntax: "reset",
        description: "Reset clipping slab, center camera on visible models, and clear selection.",
        subcommands: &[],
    },
    // --- Selection, Analysis & Measurements ---
    CommandSpec {
        name: "select",
        aliases: &["sele"],
        category: "Selection & Analysis",
        syntax: "select [name,] <selection_expr>",
        description: "Evaluate PyMOL/ChimeraX selection algebra (chain, resi, protein, rna, ligand, within, b > 80).",
        subcommands: &["protein", "rna", "ligand", "water", "chain"],
    },
    CommandSpec {
        name: "hbonds",
        aliases: &[],
        category: "Selection & Analysis",
        syntax: "hbonds [max_dist_a]",
        description: "Detect geometric hydrogen bonds and display cyan dashed overlays.",
        subcommands: &["3.5", "3.2"],
    },
    CommandSpec {
        name: "clashes",
        aliases: &[],
        category: "Selection & Analysis",
        syntax: "clashes [vdw_overlap_a]",
        description: "Detect steric VDW overlaps and paint crimson clash markers.",
        subcommands: &["0.60", "0.40"],
    },
    CommandSpec {
        name: "contacts",
        aliases: &[],
        category: "Selection & Analysis",
        syntax: "contacts [cutoff_a]",
        description: "List and visualize non-covalent atomic contacts within cutoff.",
        subcommands: &["4.0", "4.5"],
    },
    CommandSpec {
        name: "distance",
        aliases: &["dist"],
        category: "Selection & Analysis",
        syntax: "distance <C:R:A1> <C:R:A2>",
        description: "Measure Euclidean 3D distance in Ångströms with dashed measurement bar.",
        subcommands: &[],
    },
    CommandSpec {
        name: "angle",
        aliases: &[],
        category: "Selection & Analysis",
        syntax: "angle <A1> <A2> <A3>",
        description: "Measure 3-atom bond or vector angle in degrees.",
        subcommands: &[],
    },
    CommandSpec {
        name: "torsion",
        aliases: &["dihedral"],
        category: "Selection & Analysis",
        syntax: "torsion <A1> <A2> <A3> <A4>",
        description: "Measure signed 4-atom dihedral torsion angle (-180°..+180°).",
        subcommands: &[],
    },
    CommandSpec {
        name: "axis",
        aliases: &["define_axis"],
        category: "Selection & Analysis",
        syntax: "axis [label]",
        description: "Fit a 3D PCA principal axis through the active selection or helix.",
        subcommands: &[],
    },
    CommandSpec {
        name: "plane",
        aliases: &["define_plane"],
        category: "Selection & Analysis",
        syntax: "plane [label]",
        description: "Fit a 3D PCA best-fit plane through the active selection or ring.",
        subcommands: &[],
    },
    CommandSpec {
        name: "measure",
        aliases: &[],
        category: "Selection & Analysis",
        syntax: "measure clear",
        description: "Clear measurement lines, axes, planes, and measurement log.",
        subcommands: &["clear"],
    },
    CommandSpec {
        name: "msa",
        aliases: &["matchalign"],
        category: "Selection & Analysis",
        syntax: "msa",
        description: "Build a synchronized multi-model sequence alignment (Match->Align).",
        subcommands: &[],
    },
    CommandSpec {
        name: "pae",
        aliases: &[],
        category: "Selection & Analysis",
        syntax: "pae <auto | path/to/pae.json>",
        description: "Load AlphaFold/Boltz-2/Chai-1 PAE JSON or estimate structural PAE domain packing.",
        subcommands: &["auto"],
    },
    // --- Alignment, Morphing & Movies ---
    CommandSpec {
        name: "align",
        aliases: &[],
        category: "Alignment & Movies",
        syntax: "align <#2:CH|#mob> <#1:CH|#ref>",
        description: "Sequence-guided MatchMaker / Kabsch 3D superposition.",
        subcommands: &["#2:A #1:A"],
    },
    CommandSpec {
        name: "cealign",
        aliases: &[],
        category: "Alignment & Movies",
        syntax: "cealign <#2:CH> <#1:CH>",
        description: "Sequence-independent Combinatorial Extension 3D superposition (works at 0% identity).",
        subcommands: &["#2:A #1:A"],
    },
    CommandSpec {
        name: "morph",
        aliases: &[],
        category: "Alignment & Movies",
        syntax: "morph <#from> <#to> [frames]",
        description: "Generate a SHAKE-relaxed conformational morph trajectory between two models.",
        subcommands: &["#1 #2 20"],
    },
    CommandSpec {
        name: "movie",
        aliases: &[],
        category: "Alignment & Movies",
        syntax: "movie <turn|rock|fly|morph> <out.gif|out.mp4> [frames] [W] [H] [fps]",
        description: "Export an animated GIF89a (pure-Rust) or H.264 MP4 biomolecular movie.",
        subcommands: &["turn", "rock", "fly", "morph"],
    },
    // --- Interactive Modeling, Sculpting & Builder ---
    CommandSpec {
        name: "mutate",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "mutate <CHAIN:RESSEQ> <AA1> [rotamer_idx]",
        description: "Mutate a protein residue in 3D using Dunbrack rotamers with clash scoring.",
        subcommands: &[],
    },
    CommandSpec {
        name: "swapna",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "swapna <CHAIN:RESSEQ> <A|U|G|C|T>",
        description: "Mutate an RNA/DNA nucleotide preserving backbone & base plane geometry.",
        subcommands: &[],
    },
    CommandSpec {
        name: "bond_rot",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "bond_rot <C:R:A1> <C:R:A2> <delta_deg>",
        description: "Interactively rotate a covalent torsion bond with real-time clash/H-bond feedback.",
        subcommands: &[],
    },
    CommandSpec {
        name: "tug",
        aliases: &["sculpt"],
        category: "Modeling & Editing",
        syntax: "tug <C:R:A> <dx> <dy> <dz> [radius_a]",
        description: "ISOLDE-lite harmonic atom pulling with local MM sculpting relaxation.",
        subcommands: &[],
    },
    CommandSpec {
        name: "minimize",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "minimize [steps]",
        description: "Relax steric clashes with local MM steepest-descent minimization.",
        subcommands: &["80", "150"],
    },
    CommandSpec {
        name: "build",
        aliases: &["fab"],
        category: "Modeling & Editing",
        syntax: "build <helix|sheet|extended|rna> <SEQUENCE> [chain]",
        description: "Build a de novo 3D peptide (helix/sheet/extended) or A-form RNA duplex.",
        subcommands: &["helix", "sheet", "extended", "rna"],
    },
    CommandSpec {
        name: "addh",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "addh [pH]",
        description: "Add pH-aware polar hydrogens (default pH 7.4).",
        subcommands: &["7.4", "6.0", "5.0"],
    },
    CommandSpec {
        name: "altlocs",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "altlocs",
        description: "Audit alternate atomic conformations (AltLoc occupancy records).",
        subcommands: &[],
    },
    CommandSpec {
        name: "renumber",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "renumber <chain> <start_res_seq>",
        description: "Renumber all residues in a chain sequentially starting from start_res_seq.",
        subcommands: &[],
    },
    CommandSpec {
        name: "changechains",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "changechains <old_chain> <new_chain>",
        description: "Rename a chain ID cleanly across all atoms, residues, and metadata.",
        subcommands: &[],
    },
    CommandSpec {
        name: "combine",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "combine [new_name]",
        description: "Merge all currently visible models into a single combined Biomolecule.",
        subcommands: &[],
    },
    CommandSpec {
        name: "delete",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "delete <selection_expr>",
        description: "Delete atoms matching a selection expression (transactional Undo supported).",
        subcommands: &["water", "ion", "ligand"],
    },
    CommandSpec {
        name: "undo",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "undo",
        description: "Undo the last 3D structural modeling or sculpting edit.",
        subcommands: &[],
    },
    CommandSpec {
        name: "redo",
        aliases: &[],
        category: "Modeling & Editing",
        syntax: "redo",
        description: "Redo a previously undone 3D structural edit.",
        subcommands: &[],
    },
    // --- Symmetry, Glycans & Cryo-EM Volumes ---
    CommandSpec {
        name: "sym",
        aliases: &[],
        category: "Symmetry & Volumes",
        syntax: "sym",
        description: "Generate full Biological Assembly from PDB REMARK 350 BIOMT matrices.",
        subcommands: &[],
    },
    CommandSpec {
        name: "symexp",
        aliases: &["crystalcontacts"],
        category: "Symmetry & Volumes",
        syntax: "symexp [cutoff_a]",
        description: "Generate crystallographic symmetry mates from CRYST1 space group.",
        subcommands: &["5.0", "8.0"],
    },
    CommandSpec {
        name: "unitcell",
        aliases: &[],
        category: "Symmetry & Volumes",
        syntax: "unitcell <on|off>",
        description: "Toggle CRYST1 crystallographic unit-cell 3D wireframe box.",
        subcommands: &["on", "off"],
    },
    CommandSpec {
        name: "snfg",
        aliases: &["glycans"],
        category: "Symmetry & Volumes",
        syntax: "snfg <on|off>",
        description: "Detect and display 3D SNFG Glycans (GlcNAc cubes, Man spheres, Fuc cones).",
        subcommands: &["on", "off"],
    },
    CommandSpec {
        name: "molmap",
        aliases: &[],
        category: "Symmetry & Volumes",
        syntax: "molmap [resolution_a] [sigma]",
        description: "Synthesize a resolution-filtered 3D Cryo-EM density map around the active model.",
        subcommands: &["4.5 1.5", "3.5 1.5"],
    },
    CommandSpec {
        name: "open_map",
        aliases: &[],
        category: "Symmetry & Volumes",
        syntax: "open_map <map.mrc|map.dx> [sigma]",
        description: "Open an MRC2014 Cryo-EM map or OpenDX APBS electrostatic grid.",
        subcommands: &[],
    },
    CommandSpec {
        name: "fitmap",
        aliases: &[],
        category: "Symmetry & Volumes",
        syntax: "fitmap",
        description: "Rigid-body fit the active atomic model into the loaded Cryo-EM density map.",
        subcommands: &[],
    },
    CommandSpec {
        name: "vop",
        aliases: &["volume"],
        category: "Symmetry & Volumes",
        syntax: "vop <gaussian|sharpen|zone|hide_dust|orthoplanes> [param]",
        description: "Apply volumetric map filters, site zoning, dust removal, or XY/XZ/YZ Orthoplanes.",
        subcommands: &["gaussian", "sharpen", "zone", "hide_dust", "orthoplanes"],
    },
    // --- ViewDockX, Physics, Biologics Modules & Suite Export ---
    CommandSpec {
        name: "load_dock",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "load_dock <bundle.dockbundle.json>",
        description: "Load a rich DockCraft ViewDockX session (poses, HYDE coronas, torsion lights, ΔMIF).",
        subcommands: &[],
    },
    CommandSpec {
        name: "pose",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "pose <0-based_index>",
        description: "Switch active docked ligand pose inside ViewDockX.",
        subcommands: &["0", "1", "2"],
    },
    CommandSpec {
        name: "gridbox",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "gridbox <from_selection [pad] | cx cy cz sx sy sz | off>",
        description: "Define, adjust, or hide the 3D DockCraft GridBox synchronized over Craft Link v1.",
        subcommands: &["from_selection", "off"],
    },
    CommandSpec {
        name: "module",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "module <antibodies|ncrna|enzymes|adc>",
        description: "Run a CraftBio structural design module and apply its 3D overlays.",
        subcommands: &["antibodies", "ncrna", "enzymes", "adc"],
    },
    CommandSpec {
        name: "rnafold",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "rnafold <preset_id | RNA_SEQUENCE>",
        description: "Fold an ncRNA/mRNA sequence via pure-Rust Zuker/Turner-2004 MFE.",
        subcommands: &[
            "guanine_riboswitch",
            "crispr_sgrna",
            "bnt162b2_5utr",
            "patisiran_sirna",
        ],
    },
    CommandSpec {
        name: "anm",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "anm [mode_1..6] [amplitude_a]",
        description: "Compute Anisotropic Network Model collective modes & porcupine vectors.",
        subcommands: &["1 2.2", "2 2.2"],
    },
    CommandSpec {
        name: "gbsa",
        aliases: &["dipole"],
        category: "ViewDockX & Modules",
        syntax: "gbsa",
        description: "Compute pure-Rust Still GB/SA solvation energy, 3D dipole, and interface ΔG_bind.",
        subcommands: &[],
    },
    CommandSpec {
        name: "md",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "md [steps]",
        description: "Run pure-Rust local Langevin molecular dynamics trajectory.",
        subcommands: &["120", "250"],
    },
    CommandSpec {
        name: "frame",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "frame <0-based_index>",
        description: "Jump to a trajectory or morph frame on the active model.",
        subcommands: &["0"],
    },
    CommandSpec {
        name: "load_dcd",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "load_dcd <trajectory.dcd>",
        description: "Attach a binary CHARMM/OpenMM .dcd trajectory to the active model.",
        subcommands: &[],
    },
    CommandSpec {
        name: "denovo",
        aliases: &[],
        category: "ViewDockX & Modules",
        syntax: "denovo <chain> [hotspots...]",
        description: "Stage a zero-weight RFdiffusion/ProteinMPNN/Boltz-2 de novo binder campaign.",
        subcommands: &[],
    },
    CommandSpec {
        name: "png",
        aliases: &["ray"],
        category: "Export & Suite Push",
        syntax: "png <out.png> [width] [height] [transparent]",
        description: "Render a deterministic headless publication PNG (up to 4K with HUD overlays).",
        subcommands: &[],
    },
    CommandSpec {
        name: "export_mesh",
        aliases: &[],
        category: "Export & Suite Push",
        syntax: "export_mesh <out.glb|out.obj>",
        description: "Export triangulated 3D Cartoon/SES/Cryo-EM meshes to glTF 2.0 (.glb) or .obj.",
        subcommands: &[],
    },
    CommandSpec {
        name: "export_html",
        aliases: &[],
        category: "Export & Suite Push",
        syntax: "export_html <out.html>",
        description: "Export a self-contained single-file offline interactive 3D HTML dossier.",
        subcommands: &[],
    },
    CommandSpec {
        name: "export_artcraft",
        aliases: &[],
        category: "Export & Suite Push",
        syntax: "export_artcraft <out_dir>",
        description: "Export a complete ArtCraft bundle (PNG + GridCraft TSV + WordCraft/DeckCraft MD).",
        subcommands: &[],
    },
    CommandSpec {
        name: "push_artcraft",
        aliases: &["push"],
        category: "Export & Suite Push",
        syntax: "push_artcraft <wordcraft|deckcraft|gridcraft> [out_dir]",
        description: "Export bundle & live-push figure/slide/table to ArtCraft Suite over Craft Link v1.",
        subcommands: &["wordcraft", "deckcraft", "gridcraft"],
    },
    CommandSpec {
        name: "help",
        aliases: &[],
        category: "File & Sessions",
        syntax: "help [query]",
        description: "Display the unified typed BiomolCraft command reference (optionally filtered).",
        subcommands: &[],
    },
];

/// Compute `[TAB]` / inline autocomplete completions for a live DSL console string.
#[must_use]
pub fn autocomplete_dsl(input: &str) -> Vec<CompletionItem> {
    // Support semicolon-chained statements by completing the trailing clause
    let (prefix_chain, current_clause) = match input.rfind(';') {
        Some(idx) => (&input[..=idx], &input[idx + 1..]),
        None => ("", input),
    };
    let trimmed = current_clause.trim_start();
    let chain_pad = if prefix_chain.is_empty() {
        String::new()
    } else {
        format!("{prefix_chain} ")
    };

    // Case 1: Typing the second token (subcommand completion)
    if let Some((cmd_tok, rest_raw)) = trimmed.split_once(char::is_whitespace) {
        let cmd_low = cmd_tok.to_ascii_lowercase();
        let sub_prefix = rest_raw.trim_start().to_ascii_lowercase();
        if let Some(spec) = COMMAND_REGISTRY
            .iter()
            .find(|c| c.name == cmd_low || c.aliases.contains(&cmd_low.as_str()))
        {
            let mut out = Vec::new();
            for &sub in spec.subcommands {
                if sub_prefix.is_empty() || sub.to_ascii_lowercase().starts_with(&sub_prefix) {
                    out.push(CompletionItem {
                        insert_text: format!("{chain_pad}{} {sub}", spec.name),
                        display_label: format!("{} {sub}", spec.name),
                        category: spec.category,
                        description: spec.description,
                    });
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
        return Vec::new();
    }

    // Case 2: Typing the first token (command or alias prefix)
    let low = trimmed.to_ascii_lowercase();
    let mut matches = Vec::new();
    for spec in COMMAND_REGISTRY {
        let name_hit = low.is_empty() || spec.name.starts_with(&low);
        let alias_hit = !low.is_empty() && spec.aliases.iter().any(|a| a.starts_with(&low));
        if name_hit || alias_hit {
            let suffix = if spec.subcommands.is_empty() && !spec.syntax.contains('<') {
                ""
            } else {
                " "
            };
            matches.push(CompletionItem {
                insert_text: format!("{chain_pad}{}{suffix}", spec.name),
                display_label: spec.syntax.to_string(),
                category: spec.category,
                description: spec.description,
            });
        }
    }
    matches
}

/// Filter `COMMAND_REGISTRY` for the GUI `Ctrl+K` Command Palette by substring across name,
/// aliases, category, syntax, and description.
#[must_use]
pub fn filter_command_palette(query: &str) -> Vec<&'static CommandSpec> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return COMMAND_REGISTRY.iter().collect();
    }
    let tokens: Vec<&str> = q.split_whitespace().collect();
    COMMAND_REGISTRY
        .iter()
        .filter(|spec| {
            let hay = format!(
                "{} {} {} {} {}",
                spec.name,
                spec.aliases.join(" "),
                spec.category,
                spec.syntax,
                spec.description
            )
            .to_ascii_lowercase();
            tokens.iter().all(|t| hay.contains(t))
        })
        .collect()
}

/// Render unified help text from `COMMAND_REGISTRY` (optionally filtered by a search keyword).
#[must_use]
pub fn render_registry_help(filter: &str) -> String {
    let specs = filter_command_palette(filter);
    if specs.is_empty() {
        return format!("No commands match '{filter}'. Type `help` to list all commands.");
    }
    let mut out = format!(
        "BiomolCraft v1.0.0 Unified Command Registry ({} commands; press [TAB] to autocomplete or Ctrl+K for Command Palette):\n",
        specs.len()
    );
    let mut last_cat = "";
    for s in specs {
        if s.category != last_cat {
            out.push_str(&format!("\n[{}]\n", s.category));
            last_cat = s.category;
        }
        out.push_str(&format!("  {:<46} — {}\n", s.syntax, s.description));
    }
    out
}

// ---------------------------------------------------------------------------
// Capability classification of DSL verbs (sandbox confinement)
// ---------------------------------------------------------------------------
//
// The desktop app runs the DSL with the user's own file permissions, which is correct:
// the user picked the paths. A remote MCP client did not, so the MCP server may only
// expose verbs that cannot reach the filesystem or the network on their own.
//
// These two tables are the single source of truth for that split. `COMPUTE_VERBS` is a
// whitelist: a verb is only callable by an untrusted client if it is listed here by name
// or alias. Anything unlisted — including a verb added to the interpreter tomorrow — is
// refused, so the failure mode of forgetting to classify a new command is "too strict".

/// DSL verbs (canonical names and aliases) that read or write files, or hit the network,
/// using paths taken straight from the command text.
pub const FILESYSTEM_VERBS: &[&str] = &[
    "open",
    "load",
    "fetch",
    "save",
    "save_session",
    "pse",
    "cxs",
    "load_session",
    "open_session",
    "load_dock",
    "load_dcd",
    "open_map",
    "pae",
    "export_mesh",
    "export_html",
    "export_artcraft",
    "push_artcraft",
    "push",
    "png",
    "ray",
    "movie",
];

/// DSL verbs (canonical names and aliases) that only touch in-memory session state:
/// geometry, selections, colouring, camera, analysis and scientific modules.
pub const COMPUTE_VERBS: &[&str] = &[
    "help",
    "models",
    "active",
    "hide",
    "close",
    "scene",
    "clip",
    "lighting",
    "ssao",
    "bg",
    "background",
    "preset",
    "style",
    "show",
    "color",
    "surface",
    "silhouettes",
    "waters",
    "pose",
    "gridbox",
    "select",
    "sele",
    "mutate",
    "minimize",
    "align",
    "module",
    "rnafold",
    "anm",
    "gbsa",
    "dipole",
    "md",
    "frame",
    "tunnels",
    "molmap",
    "fitmap",
    "denovo",
    "putty",
    "hbonds",
    "clashes",
    "contacts",
    "distance",
    "dist",
    "angle",
    "torsion",
    "dihedral",
    "measure",
    "axis",
    // Registry-only aliases: advertised by `help`/autocomplete but not yet implemented in
    // the interpreter's command match. Classified as compute so the capability tables stay
    // exhaustive; see REGISTRY_ONLY_ALIASES in the tests.
    "define_axis",
    "define_plane",
    "matchalign",
    "plane",
    "label",
    "2dlabel",
    "2dlabels",
    "key",
    "scalebar",
    "cealign",
    "super",
    "msa",
    "morph",
    "swapna",
    "bond_rot",
    "tug",
    "sculpt",
    "build",
    "fab",
    "addh",
    "altlocs",
    "renumber",
    "changechains",
    "combine",
    "delete",
    "sym",
    "symexp",
    "crystalcontacts",
    "unitcell",
    "snfg",
    "glycans",
    "vop",
    "volume",
    "undo",
    "redo",
    "reset",
    "center",
];

/// Does this DSL verb reach the filesystem or the network by itself?
///
/// Case-insensitive, and accepts aliases (`pse` as well as `save_session`).
#[must_use]
pub fn verb_touches_filesystem(verb: &str) -> bool {
    let low = verb.trim().to_ascii_lowercase();
    FILESYSTEM_VERBS.contains(&low.as_str())
}

/// Is this DSL verb safe to expose to a caller confined to a `PathSandbox`?
///
/// Returns `false` for filesystem verbs **and** for anything unknown, so a newly added
/// command is refused until somebody classifies it.
#[must_use]
pub fn verb_is_sandbox_safe(verb: &str) -> bool {
    let low = verb.trim().to_ascii_lowercase();
    !FILESYSTEM_VERBS.contains(&low.as_str()) && COMPUTE_VERBS.contains(&low.as_str())
}

#[cfg(test)]
mod capability_tests {
    use super::*;

    /// Every verb in the registry must be classified exactly once. Adding a command to
    /// `COMMAND_REGISTRY` without saying whether it touches the filesystem fails here.
    #[test]
    fn every_registry_command_and_alias_is_classified_exactly_once() {
        for spec in COMMAND_REGISTRY {
            for verb in std::iter::once(spec.name).chain(spec.aliases.iter().copied()) {
                let fs = FILESYSTEM_VERBS.contains(&verb);
                let compute = COMPUTE_VERBS.contains(&verb);
                assert!(
                    fs ^ compute,
                    "DSL verb '{verb}' must be in exactly one of FILESYSTEM_VERBS / COMPUTE_VERBS (fs={fs}, compute={compute})"
                );
            }
        }
    }

    #[test]
    fn the_two_capability_tables_are_lowercase_and_disjoint() {
        for v in FILESYSTEM_VERBS.iter().chain(COMPUTE_VERBS.iter()) {
            assert_eq!(*v, v.to_ascii_lowercase(), "verb '{v}' must be lowercase");
        }
        for v in FILESYSTEM_VERBS {
            assert!(!COMPUTE_VERBS.contains(v), "verb '{v}' is in both tables");
        }
    }

    #[test]
    fn filesystem_verbs_are_never_sandbox_safe_whatever_the_casing() {
        for v in FILESYSTEM_VERBS {
            assert!(verb_touches_filesystem(v), "{v}");
            assert!(verb_touches_filesystem(&v.to_ascii_uppercase()), "{v}");
            assert!(!verb_is_sandbox_safe(v), "{v}");
            assert!(!verb_is_sandbox_safe(&v.to_ascii_uppercase()), "{v}");
        }
    }

    #[test]
    fn unknown_verbs_are_refused_rather_than_assumed_harmless() {
        for v in ["", "   ", "exfiltrate", "save_everything", "opened"] {
            assert!(!verb_is_sandbox_safe(v), "unknown verb '{v}' was allowed");
        }
    }
}
