#![forbid(unsafe_code)]

//! `biomol-model` — Interactive 3D Biomolecular Modeling, PyMOL-Style Mutagenesis Wizard
//! (Dunbrack Rotamers + Live Steric Clash Discs + Local Repacking), Local MM Energy Minimizer
//! (ISOLDE-lite Steepest-Descent), and 3D Structure Preparation Audit.

pub mod aggregation;
pub mod edit3d;
pub mod inspect;
pub mod ramachandran;

pub use aggregation::{
    Aggregation3DReport, AggregationPatch, AggregationVisuals, ResidueAggregationScore,
    SolubilityRescueReport, aggregation_visual_palette, aggrescan_a3v, analyze_aggregation_3d,
    render_aggregation_markdown, simulate_solubility_rescue,
};
pub use edit3d::{
    AddHydrogensReport, BondRotReport, SwapNaReport, TugSculptReport, add_polar_hydrogens_ph,
    build_de_novo_peptide_3d, build_de_novo_rna_helix_3d, combine_biomolecules,
    delete_atoms_by_mask, evaluate_local_binding_proxy, rename_chain, renumber_chain,
    rotate_bond_torsion, swapna_nucleotide_3d, tug_atom_with_sculpting,
};
pub use inspect::{
    ChainInfoSummary, MissingGap, StructureInfoReport, TruncatedSidechain,
    expected_sidechain_heavy_atoms, inspect_structure_info, measure_residue_chi1_chi2,
    render_info_markdown,
};
pub use ramachandran::{
    RamaPoint, RamaRegion, RamaResidueKind, RamachandranReport, analyze_ramachandran,
    classify_ramachandran, ramachandran_density, render_ramachandran_markdown,
};

use std::fmt::Write as _;

use craftbio_scene::SphereOverlay;
use craftbio_struct::{
    Atom, BioStructError, Biomolecule, ResidueClass, aa1_to_aa3, approx_atom_charge, dist3,
    finalize_biomolecule, measure_dihedral_deg, vdw_radius,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("structure error: {0}")]
    Struct(#[from] BioStructError),
    #[error("residue {0}:{1} not found or lacks N/CA/C backbone frame")]
    ResidueNotFound(char, i32),
    #[error("unsupported target amino acid '{0}'")]
    InvalidAminoAcid(char),
}

/// Backbone conformational bin used to order backbone-dependent Dunbrack rotamers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackboneBin {
    Alpha,
    Beta,
    Coil,
}

impl BackboneBin {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Alpha => "alpha-helix",
            Self::Beta => "beta-sheet",
            Self::Coil => "coil/other",
        }
    }
}

/// A backbone-dependent Dunbrack sidechain rotamer entry (`χ1–χ4` + prevalence `%`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DunbrackRotamer {
    pub chi1: f64,
    pub chi2: Option<f64>,
    pub chi3: Option<f64>,
    pub chi4: Option<f64>,
    pub probability_pct: f64,
}

/// Result of a 3D point mutation with rotamer placement and clash diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutation3DReport {
    pub chain: char,
    pub res_seq: i32,
    pub wild_type: char,
    pub mutant: char,
    pub rotamer_index: usize,
    pub total_rotamers: usize,
    pub phi_deg: Option<f64>,
    pub psi_deg: Option<f64>,
    pub backbone_bin: BackboneBin,
    pub chi1_deg: f64,
    pub chi2_deg: Option<f64>,
    pub rotamer_probability_pct: f64,
    pub clash_count: usize,
    pub clashes_ligand: usize,
    pub clashes_protein: usize,
    pub max_overlap_a: f64,
    pub new_hbond_count: usize,
    pub clash_overlays: Vec<SphereOverlay>,
}

/// One row in a full Dunbrack rotamer enumeration table (ChimeraX `swapaa` / PRAC3 Part 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotamerCandidate {
    pub rotamer_index: usize,
    pub chi1_deg: f64,
    pub chi2_deg: Option<f64>,
    pub probability_pct: f64,
    pub clashes_ligand: usize,
    pub clashes_protein: usize,
    pub total_clashes: usize,
    pub max_overlap_a: f64,
    pub new_hbonds: usize,
}

/// Complete enumerated Dunbrack rotamer table for a residue position & target amino acid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotamerTableReport {
    pub chain: char,
    pub res_seq: i32,
    pub wild_type: char,
    pub mutant: char,
    pub phi_deg: Option<f64>,
    pub psi_deg: Option<f64>,
    pub backbone_bin: BackboneBin,
    pub candidates: Vec<RotamerCandidate>,
    pub best_rotamer_index: usize,
}

/// Parse either a 1-letter (`I`) or 3-letter (`ILE`/`ile`) standard amino acid code.
#[must_use]
pub fn parse_aa_code(raw: &str) -> Option<char> {
    let s = raw.trim().to_ascii_uppercase();
    if s.len() == 1 {
        let c = s.chars().next()?;
        return "ACDEFGHIKLMNPQRSTVWY".contains(c).then_some(c);
    }
    match s.as_str() {
        "ALA" => Some('A'),
        "CYS" => Some('C'),
        "ASP" => Some('D'),
        "GLU" => Some('E'),
        "PHE" => Some('F'),
        "GLY" => Some('G'),
        "HIS" | "HID" | "HIE" | "HIP" => Some('H'),
        "ILE" => Some('I'),
        "LYS" => Some('K'),
        "LEU" => Some('L'),
        "MET" => Some('M'),
        "ASN" => Some('N'),
        "PRO" => Some('P'),
        "GLN" => Some('Q'),
        "ARG" => Some('R'),
        "SER" => Some('S'),
        "THR" => Some('T'),
        "VAL" => Some('V'),
        "TRP" => Some('W'),
        "TYR" => Some('Y'),
        _ => None,
    }
}

/// Summary of a local MM energy minimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimizationReport {
    pub atoms_relaxed: usize,
    pub steps: usize,
    pub energy_before_kcal: f64,
    pub energy_after_kcal: f64,
    pub max_displacement_a: f64,
}

/// Structural quality & preparation audit summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructurePrepAudit {
    pub total_atoms: usize,
    pub protein_chains: Vec<char>,
    pub nucleic_chains: Vec<char>,
    pub ligand_residues: usize,
    pub ion_count: usize,
    pub water_count: usize,
    pub disulfide_count: usize,
    pub rna_base_pair_count: usize,
    pub steric_clash_pairs: usize,
    pub notes: Vec<String>,
}

pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 1e-9 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

/// Curated top Dunbrack backbone-independent `chi1` rotamer angles (degrees) per amino acid.
#[must_use]
pub fn dunbrack_chi1_rotamers(aa: char) -> &'static [f64] {
    match aa.to_ascii_uppercase() {
        'G' | 'A' | 'P' => &[0.0],
        'V' | 'I' | 'T' => &[180.0, -60.0, 60.0],
        'S' | 'C' => &[60.0, -60.0, 180.0],
        _ => &[-60.0, 180.0, 60.0],
    }
}

/// Classify local backbone `(φ, ψ)` angles into Dunbrack secondary-structure bins.
#[must_use]
pub fn classify_backbone_bin(phi: Option<f64>, psi: Option<f64>) -> BackboneBin {
    if let (Some(p), Some(q)) = (phi, psi) {
        if (-110.0..=-30.0).contains(&p) && (-75.0..=10.0).contains(&q) {
            return BackboneBin::Alpha;
        }
        if (-180.0..=-60.0).contains(&p) && (80.0..=180.0).contains(&q) {
            return BackboneBin::Beta;
        }
    }
    BackboneBin::Coil
}

/// Curated backbone-dependent Dunbrack 2010 rotamer table (`χ1–χ4` + prevalence `%`)
/// ordered from most probable to least probable for the local `(φ, ψ)` bin.
#[must_use]
pub fn dunbrack_rotamers(aa: char, bin: BackboneBin) -> Vec<DunbrackRotamer> {
    let r1 = |c1: f64, p: f64| DunbrackRotamer {
        chi1: c1,
        chi2: None,
        chi3: None,
        chi4: None,
        probability_pct: p,
    };
    let r2 = |c1: f64, c2: f64, p: f64| DunbrackRotamer {
        chi1: c1,
        chi2: Some(c2),
        chi3: None,
        chi4: None,
        probability_pct: p,
    };
    let r3 = |c1: f64, c2: f64, c3: f64, p: f64| DunbrackRotamer {
        chi1: c1,
        chi2: Some(c2),
        chi3: Some(c3),
        chi4: None,
        probability_pct: p,
    };
    let r4 = |c1: f64, c2: f64, c3: f64, c4: f64, p: f64| DunbrackRotamer {
        chi1: c1,
        chi2: Some(c2),
        chi3: Some(c3),
        chi4: Some(c4),
        probability_pct: p,
    };

    match aa.to_ascii_uppercase() {
        'G' | 'A' => vec![r1(0.0, 100.0)],
        'P' => vec![r2(28.0, -32.0, 52.0), r2(-28.0, 32.0, 48.0)],
        'S' => match bin {
            BackboneBin::Alpha => vec![r1(-64.0, 52.0), r1(64.0, 33.0), r1(179.0, 15.0)],
            BackboneBin::Beta => vec![r1(63.0, 44.0), r1(-63.0, 31.0), r1(178.0, 25.0)],
            BackboneBin::Coil => vec![r1(63.0, 48.0), r1(-63.0, 29.0), r1(179.0, 23.0)],
        },
        'C' => match bin {
            BackboneBin::Alpha => vec![r1(-65.0, 68.0), r1(-177.0, 23.0), r1(64.0, 9.0)],
            _ => vec![r1(-64.0, 56.0), r1(-176.0, 27.0), r1(63.0, 17.0)],
        },
        'T' => match bin {
            BackboneBin::Alpha => vec![r1(-61.0, 64.0), r1(60.0, 29.0), r1(-174.0, 7.0)],
            BackboneBin::Beta => vec![r1(61.0, 49.0), r1(-61.0, 43.0), r1(-174.0, 8.0)],
            BackboneBin::Coil => vec![r1(61.0, 50.0), r1(-61.0, 41.0), r1(-174.0, 9.0)],
        },
        'V' => match bin {
            BackboneBin::Alpha => vec![r1(174.0, 78.0), r1(-63.0, 17.0), r1(64.0, 5.0)],
            BackboneBin::Beta => vec![r1(175.0, 73.0), r1(-62.0, 20.0), r1(64.0, 7.0)],
            BackboneBin::Coil => vec![r1(174.0, 72.0), r1(-63.0, 21.0), r1(64.0, 7.0)],
        },
        'I' => match bin {
            BackboneBin::Alpha => vec![
                r2(-64.0, 169.0, 66.0),
                r2(-62.0, -61.0, 17.0),
                r2(-170.0, 168.0, 11.0),
                r2(62.0, 169.0, 3.0),
                r2(-171.0, 66.0, 1.5),
                r2(-63.0, 66.0, 1.0),
                r2(62.0, -60.0, 0.5),
            ],
            BackboneBin::Beta => vec![
                r2(-63.0, 168.0, 58.0),
                r2(-62.0, -60.0, 18.0),
                r2(62.0, 169.0, 12.0),
                r2(-170.0, 168.0, 8.0),
                r2(-171.0, 66.0, 2.0),
                r2(-63.0, 66.0, 1.2),
                r2(62.0, -60.0, 0.8),
            ],
            BackboneBin::Coil => vec![
                r2(-63.0, 169.0, 60.0),
                r2(-62.0, -60.0, 17.0),
                r2(63.0, 169.0, 11.0),
                r2(-170.0, 168.0, 8.0),
                r2(-171.0, 66.0, 2.0),
                r2(-63.0, 66.0, 1.2),
                r2(63.0, -60.0, 0.8),
            ],
        },
        'L' => match bin {
            BackboneBin::Alpha => vec![
                r2(-65.0, 174.0, 65.0),
                r2(-176.0, 64.0, 28.0),
                r2(-65.0, -64.0, 3.0),
                r2(-175.0, 175.0, 2.5),
                r2(63.0, 80.0, 1.5),
            ],
            _ => vec![
                r2(-65.0, 175.0, 58.0),
                r2(-177.0, 64.0, 32.0),
                r2(-175.0, 174.0, 4.0),
                r2(-65.0, -64.0, 3.5),
                r2(63.0, 80.0, 2.5),
            ],
        },
        'D' => vec![
            r2(-69.0, -15.0, 48.0),
            r2(-172.0, 0.0, 31.0),
            r2(64.0, 0.0, 14.0),
            r2(-69.0, 40.0, 7.0),
        ],
        'N' => vec![
            r2(-68.0, -25.0, 44.0),
            r2(-173.0, 10.0, 27.0),
            r2(63.0, -10.0, 15.0),
            r2(-68.0, 125.0, 14.0),
        ],
        'E' => vec![
            r3(-67.0, 180.0, -10.0, 36.0),
            r3(-177.0, 178.0, 0.0, 26.0),
            r3(-67.0, -65.0, -25.0, 16.0),
            r3(-177.0, 65.0, 10.0, 12.0),
            r3(64.0, 180.0, 0.0, 10.0),
        ],
        'Q' => vec![
            r3(-67.0, 180.0, -15.0, 35.0),
            r3(-177.0, 178.0, 5.0, 25.0),
            r3(-67.0, -65.0, -30.0, 17.0),
            r3(-177.0, 65.0, 15.0, 13.0),
            r3(64.0, 180.0, 0.0, 10.0),
        ],
        'M' => vec![
            r3(-67.0, 179.0, -70.0, 34.0),
            r3(-67.0, -64.0, -70.0, 24.0),
            r3(-176.0, 178.0, 75.0, 22.0),
            r3(-67.0, 179.0, 70.0, 12.0),
            r3(64.0, 180.0, -70.0, 8.0),
        ],
        'K' => vec![
            r4(-67.0, 180.0, 180.0, 180.0, 34.0),
            r4(-177.0, 179.0, 180.0, 180.0, 25.0),
            r4(-67.0, -66.0, 180.0, 180.0, 16.0),
            r4(-67.0, 180.0, -68.0, 180.0, 13.0),
            r4(64.0, 180.0, 180.0, 180.0, 12.0),
        ],
        'R' => vec![
            r4(-67.0, 180.0, 180.0, 180.0, 32.0),
            r4(-177.0, 179.0, 180.0, 180.0, 24.0),
            r4(-67.0, -66.0, 180.0, -175.0, 17.0),
            r4(-67.0, 180.0, 68.0, 180.0, 14.0),
            r4(64.0, 180.0, 180.0, 180.0, 13.0),
        ],
        'H' => vec![
            r2(-64.0, -75.0, 33.0),
            r2(-64.0, 82.0, 22.0),
            r2(-176.0, 72.0, 21.0),
            r2(-176.0, -82.0, 13.0),
            r2(64.0, -80.0, 11.0),
        ],
        'F' | 'Y' => match bin {
            BackboneBin::Alpha => vec![
                r2(-65.0, 95.0, 48.0),
                r2(-177.0, 76.0, 37.0),
                r2(64.0, 90.0, 11.0),
                r2(-65.0, -15.0, 4.0),
            ],
            _ => vec![
                r2(-65.0, 96.0, 45.0),
                r2(-177.0, 78.0, 35.0),
                r2(64.0, 90.0, 15.0),
                r2(-65.0, -15.0, 5.0),
            ],
        },
        'W' => vec![
            r2(-66.0, 98.0, 35.0),
            r2(-66.0, -90.0, 20.0),
            r2(-178.0, 68.0, 19.0),
            r2(-178.0, -104.0, 14.0),
            r2(63.0, -88.0, 12.0),
        ],
        _ => vec![r1(-60.0, 100.0)],
    }
}

/// Natural Extension Reference Frame (NeRF): place atom `D` at exact distance `bond_len` from `c`,
/// exact bond angle `∠(b-c-d) = bond_angle_deg`, and exact IUPAC dihedral `τ(a-b-c-d) = dihedral_deg`.
pub(crate) fn place_nerf(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    bond_len: f64,
    bond_angle_deg: f64,
    dihedral_deg: f64,
) -> [f64; 3] {
    let bc = norm3(sub3(c, b));
    let ab = sub3(b, a);
    let n_raw = cross(ab, bc);
    let n_len = (n_raw[0] * n_raw[0] + n_raw[1] * n_raw[1] + n_raw[2] * n_raw[2]).sqrt();
    let n = if n_len > 1e-8 {
        [n_raw[0] / n_len, n_raw[1] / n_len, n_raw[2] / n_len]
    } else {
        let alt = if bc[0].abs() < 0.8 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        norm3(cross(alt, bc))
    };
    let m = cross(n, bc);
    let th = bond_angle_deg.to_radians();
    let tau = dihedral_deg.to_radians();
    let sin_th = th.sin();
    let dx = -th.cos();
    let dy = sin_th * tau.cos();
    let dz = sin_th * tau.sin();
    [
        c[0] + bond_len * (dx * bc[0] + dy * m[0] + dz * n[0]),
        c[1] + bond_len * (dx * bc[1] + dy * m[1] + dz * n[1]),
        c[2] + bond_len * (dx * bc[2] + dy * m[2] + dz * n[2]),
    ]
}

/// Compute local `(φ, ψ)` backbone torsion angles for `chain:res_seq` when flanking residues exist.
#[must_use]
pub fn residue_backbone_phi_psi(
    mol: &Biomolecule,
    chain: char,
    res_seq: i32,
) -> (Option<f64>, Option<f64>) {
    let Some(ch) = mol.chain(chain) else {
        return (None, None);
    };
    let prot: Vec<_> = ch
        .residues
        .iter()
        .filter(|r| r.class == ResidueClass::Protein)
        .collect();
    let Some(pos) = prot.iter().position(|r| r.res_seq == res_seq) else {
        return (None, None);
    };
    let atom = |r_idx: usize, name: &str| -> Option<[f64; 3]> {
        prot[r_idx]
            .atom_indices
            .iter()
            .copied()
            .find(|&i| mol.atoms[i].name == name)
            .map(|i| mol.atoms[i].coords)
    };
    let (Some(n), Some(ca), Some(c)) = (atom(pos, "N"), atom(pos, "CA"), atom(pos, "C")) else {
        return (None, None);
    };
    let phi = if pos > 0 {
        atom(pos - 1, "C").and_then(|c_prev| {
            (dist3(c_prev, n) <= 2.1).then(|| (measure_dihedral_deg(c_prev, n, ca, c) * 10.0).round() / 10.0)
        })
    } else {
        None
    };
    let psi = if pos + 1 < prot.len() {
        atom(pos + 1, "N").and_then(|n_next| {
            (dist3(c, n_next) <= 2.1).then(|| (measure_dihedral_deg(n, ca, c, n_next) * 10.0).round() / 10.0)
        })
    } else {
        None
    };
    (phi, psi)
}

/// Construct idealized heavy sidechain atom coordinates in the local `(N, CA, C)` frame for `aa`
/// using exact NeRF internal-coordinate geometry at `rot` (`χ1`, `χ2`, `χ3`, `χ4`).
pub(crate) fn build_sidechain_atoms_rotamer(
    n_pos: [f64; 3],
    ca_pos: [f64; 3],
    c_pos: [f64; 3],
    aa: char,
    rot: DunbrackRotamer,
) -> Vec<(&'static str, &'static str, [f64; 3])> {
    if aa == 'G' {
        return Vec::new();
    }
    // Ideal L-amino-acid CB placed via NeRF from (C, N, CA) at 1.53 Å, 110.5°, dihedral -122.7°
    let cb = place_nerf(c_pos, n_pos, ca_pos, 1.53, 110.5, -122.7);
    if aa == 'A' {
        return vec![("CB", "C", cb)];
    }

    let mut out = vec![("CB", "C", cb)];
    let c1 = rot.chi1;
    let c2 = rot.chi2.unwrap_or(180.0);
    let c3 = rot.chi3.unwrap_or(180.0);
    let c4 = rot.chi4.unwrap_or(180.0);

    match aa {
        'S' => {
            let og = place_nerf(n_pos, ca_pos, cb, 1.42, 110.8, c1);
            out.push(("OG", "O", og));
        }
        'C' => {
            let sg = place_nerf(n_pos, ca_pos, cb, 1.81, 114.0, c1);
            out.push(("SG", "S", sg));
        }
        'T' => {
            let og1 = place_nerf(n_pos, ca_pos, cb, 1.43, 109.5, c1);
            let cg2 = place_nerf(n_pos, ca_pos, cb, 1.52, 110.5, c1 - 122.0);
            out.push(("OG1", "O", og1));
            out.push(("CG2", "C", cg2));
        }
        'V' => {
            let cg1 = place_nerf(n_pos, ca_pos, cb, 1.52, 111.0, c1);
            let cg2 = place_nerf(n_pos, ca_pos, cb, 1.52, 111.0, c1 - 122.5);
            out.push(("CG1", "C", cg1));
            out.push(("CG2", "C", cg2));
        }
        'I' => {
            let cg1 = place_nerf(n_pos, ca_pos, cb, 1.53, 111.5, c1);
            let cg2 = place_nerf(n_pos, ca_pos, cb, 1.52, 110.5, c1 - 122.5);
            let cd1 = place_nerf(ca_pos, cb, cg1, 1.52, 113.8, c2);
            out.push(("CG1", "C", cg1));
            out.push(("CG2", "C", cg2));
            out.push(("CD1", "C", cd1));
        }
        'L' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.53, 116.0, c1);
            let cd1 = place_nerf(ca_pos, cb, cg, 1.52, 110.7, c2);
            let cd2 = place_nerf(ca_pos, cb, cg, 1.52, 110.7, c2 + 123.0);
            out.push(("CG", "C", cg));
            out.push(("CD1", "C", cd1));
            out.push(("CD2", "C", cd2));
        }
        'D' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.52, 113.0, c1);
            let od1 = place_nerf(ca_pos, cb, cg, 1.25, 118.5, c2);
            let od2 = place_nerf(ca_pos, cb, cg, 1.25, 118.5, c2 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("OD1", "O", od1));
            out.push(("OD2", "O", od2));
        }
        'N' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.52, 113.0, c1);
            let od1 = place_nerf(ca_pos, cb, cg, 1.23, 121.0, c2);
            let nd2 = place_nerf(ca_pos, cb, cg, 1.33, 116.5, c2 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("OD1", "O", od1));
            out.push(("ND2", "N", nd2));
        }
        'E' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.52, 114.0, c1);
            let cd = place_nerf(ca_pos, cb, cg, 1.52, 113.0, c2);
            let oe1 = place_nerf(cb, cg, cd, 1.25, 118.5, c3);
            let oe2 = place_nerf(cb, cg, cd, 1.25, 118.5, c3 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("OE1", "O", oe1));
            out.push(("OE2", "O", oe2));
        }
        'Q' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.52, 114.0, c1);
            let cd = place_nerf(ca_pos, cb, cg, 1.52, 113.0, c2);
            let oe1 = place_nerf(cb, cg, cd, 1.23, 121.0, c3);
            let ne2 = place_nerf(cb, cg, cd, 1.33, 116.5, c3 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("OE1", "O", oe1));
            out.push(("NE2", "N", ne2));
        }
        'M' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.52, 113.5, c1);
            let sd = place_nerf(ca_pos, cb, cg, 1.81, 112.5, c2);
            let ce = place_nerf(cb, cg, sd, 1.79, 100.5, c3);
            out.push(("CG", "C", cg));
            out.push(("SD", "S", sd));
            out.push(("CE", "C", ce));
        }
        'K' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.52, 114.0, c1);
            let cd = place_nerf(ca_pos, cb, cg, 1.52, 112.0, c2);
            let ce = place_nerf(cb, cg, cd, 1.52, 112.0, c3);
            let nz = place_nerf(cg, cd, ce, 1.49, 112.0, c4);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("CE", "C", ce));
            out.push(("NZ", "N", nz));
        }
        'R' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.52, 114.0, c1);
            let cd = place_nerf(ca_pos, cb, cg, 1.52, 112.0, c2);
            let ne = place_nerf(cb, cg, cd, 1.46, 112.0, c3);
            let cz = place_nerf(cg, cd, ne, 1.33, 124.0, c4);
            let nh1 = place_nerf(cd, ne, cz, 1.33, 120.0, 0.0);
            let nh2 = place_nerf(cd, ne, cz, 1.33, 120.0, 180.0);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("NE", "N", ne));
            out.push(("CZ", "C", cz));
            out.push(("NH1", "N", nh1));
            out.push(("NH2", "N", nh2));
        }
        'H' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.50, 113.8, c1);
            let nd1 = place_nerf(ca_pos, cb, cg, 1.38, 122.7, c2);
            let cd2 = place_nerf(ca_pos, cb, cg, 1.36, 131.0, c2 + 180.0);
            let ce1 = place_nerf(cb, cg, nd1, 1.33, 107.0, 180.0);
            let ne2 = place_nerf(cb, cg, cd2, 1.37, 107.2, 180.0);
            out.push(("CG", "C", cg));
            out.push(("ND1", "N", nd1));
            out.push(("CD2", "C", cd2));
            out.push(("CE1", "C", ce1));
            out.push(("NE2", "N", ne2));
        }
        'F' | 'Y' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.51, 113.8, c1);
            let cd1 = place_nerf(ca_pos, cb, cg, 1.39, 120.0, c2);
            let cd2 = place_nerf(ca_pos, cb, cg, 1.39, 120.0, c2 + 180.0);
            let ce1 = place_nerf(cb, cg, cd1, 1.39, 120.0, 180.0);
            let ce2 = place_nerf(cb, cg, cd2, 1.39, 120.0, 180.0);
            let cz = place_nerf(cg, cd1, ce1, 1.39, 120.0, 0.0);
            out.push(("CG", "C", cg));
            out.push(("CD1", "C", cd1));
            out.push(("CD2", "C", cd2));
            out.push(("CE1", "C", ce1));
            out.push(("CE2", "C", ce2));
            out.push(("CZ", "C", cz));
            if aa == 'Y' {
                let oh = place_nerf(cd1, ce1, cz, 1.37, 120.0, 180.0);
                out.push(("OH", "O", oh));
            }
        }
        'W' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.50, 113.8, c1);
            let cd1 = place_nerf(ca_pos, cb, cg, 1.37, 126.9, c2);
            let cd2 = place_nerf(ca_pos, cb, cg, 1.43, 126.8, c2 + 180.0);
            let ne1 = place_nerf(cb, cg, cd1, 1.38, 108.5, 180.0);
            let ce2 = place_nerf(cb, cg, cd2, 1.41, 107.0, 180.0);
            let ce3 = place_nerf(cb, cg, cd2, 1.40, 134.0, 0.0);
            let cz2 = place_nerf(cg, cd2, ce2, 1.40, 122.0, 180.0);
            let cz3 = place_nerf(cg, cd2, ce3, 1.39, 118.5, 180.0);
            let ch2 = place_nerf(cd2, ce2, cz2, 1.40, 117.5, 0.0);
            out.push(("CG", "C", cg));
            out.push(("CD1", "C", cd1));
            out.push(("CD2", "C", cd2));
            out.push(("NE1", "N", ne1));
            out.push(("CE2", "C", ce2));
            out.push(("CE3", "C", ce3));
            out.push(("CZ2", "C", cz2));
            out.push(("CZ3", "C", cz3));
            out.push(("CH2", "C", ch2));
        }
        'P' => {
            let cg = place_nerf(n_pos, ca_pos, cb, 1.50, 104.5, c1);
            let cd = place_nerf(ca_pos, cb, cg, 1.50, 104.5, c2);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
        }
        _ => {}
    }
    out
}

/// Convenience wrapper for `edit3d` peptide builder using a single `chi1_deg`.
pub(crate) fn build_sidechain_atoms(
    n_pos: [f64; 3],
    ca_pos: [f64; 3],
    c_pos: [f64; 3],
    aa: char,
    chi1_deg: f64,
) -> Vec<(&'static str, &'static str, [f64; 3])> {
    let mut rot = dunbrack_rotamers(aa, BackboneBin::Coil)
        .into_iter()
        .next()
        .unwrap_or(DunbrackRotamer {
            chi1: chi1_deg,
            chi2: None,
            chi3: None,
            chi4: None,
            probability_pct: 100.0,
        });
    rot.chi1 = chi1_deg;
    build_sidechain_atoms_rotamer(n_pos, ca_pos, c_pos, aa, rot)
}

/// Mutate residue `chain:res_seq` in place to `target_aa` using backbone-dependent Dunbrack
/// rotamer `rotamer_idx`, returning live steric clash overlays (ChimeraX `0.60 Å` overlap cutoff),
/// ligand vs protein clash breakdown, and H-bond diagnostics.
pub fn mutate_residue_3d(
    mol: &mut Biomolecule,
    chain: char,
    res_seq: i32,
    target_aa: char,
    rotamer_idx: usize,
) -> Result<Mutation3DReport, ModelError> {
    let aa = target_aa.to_ascii_uppercase();
    if !"ACDEFGHIKLMNPQRSTVWY".contains(aa) {
        return Err(ModelError::InvalidAminoAcid(target_aa));
    }
    let (phi_deg, psi_deg) = residue_backbone_phi_psi(mol, chain, res_seq);
    let backbone_bin = classify_backbone_bin(phi_deg, psi_deg);

    let ch = mol
        .chain(chain)
        .ok_or(ModelError::ResidueNotFound(chain, res_seq))?;
    let res = ch
        .residues
        .iter()
        .find(|r| r.res_seq == res_seq && r.class == ResidueClass::Protein)
        .ok_or(ModelError::ResidueNotFound(chain, res_seq))?;

    let wt = res.code;
    let n_pos = res
        .atom_indices
        .iter()
        .find(|&&i| mol.atoms[i].name == "N")
        .map(|&i| mol.atoms[i].coords)
        .ok_or(ModelError::ResidueNotFound(chain, res_seq))?;
    let ca_pos = res
        .atom_indices
        .iter()
        .find(|&&i| mol.atoms[i].name == "CA")
        .map(|&i| mol.atoms[i].coords)
        .ok_or(ModelError::ResidueNotFound(chain, res_seq))?;
    let c_pos = res
        .atom_indices
        .iter()
        .find(|&&i| mol.atoms[i].name == "C")
        .map(|&i| mol.atoms[i].coords)
        .ok_or(ModelError::ResidueNotFound(chain, res_seq))?;

    let rots = dunbrack_rotamers(aa, backbone_bin);
    let r_idx = rotamer_idx % rots.len().max(1);
    let rot = rots[r_idx];
    let new_sc = build_sidechain_atoms_rotamer(n_pos, ca_pos, c_pos, aa, rot);
    let new_rname = aa1_to_aa3(aa).to_string();

    // Rebuild atom list replacing target residue sidechain atoms while preserving backbone
    let mut new_atoms = Vec::with_capacity(mol.atoms.len() + 12);
    let mut inserted_sc = false;
    let mut inserted_indices = Vec::new();
    let icode = res.icode;
    let bf = res.mean_bfactor;

    for a in &mol.atoms {
        if a.chain == chain && a.res_seq == res_seq && a.icode == icode {
            if matches!(a.name.as_str(), "N" | "CA" | "C" | "O" | "OXT") {
                let mut bb = a.clone();
                bb.res_name = new_rname.clone();
                bb.res_code = aa;
                bb.formal_charge = approx_atom_charge(ResidueClass::Protein, aa, &bb.name);
                new_atoms.push(bb);
                if a.name == "O" && !inserted_sc {
                    for (aname, elem, coords) in &new_sc {
                        inserted_indices.push(new_atoms.len());
                        new_atoms.push(Atom {
                            serial: 0,
                            name: (*aname).to_string(),
                            element: (*elem).to_string(),
                            res_name: new_rname.clone(),
                            res_code: aa,
                            res_class: ResidueClass::Protein,
                            chain,
                            res_seq,
                            icode,
                            coords: *coords,
                            occupancy: 1.0,
                            b_factor: bf,
                            formal_charge: approx_atom_charge(ResidueClass::Protein, aa, aname),
                            is_hetatm: false,
                        });
                    }
                    inserted_sc = true;
                }
            }
        } else {
            new_atoms.push(a.clone());
        }
    }
    if !inserted_sc {
        for (aname, elem, coords) in &new_sc {
            inserted_indices.push(new_atoms.len());
            new_atoms.push(Atom {
                serial: 0,
                name: (*aname).to_string(),
                element: (*elem).to_string(),
                res_name: new_rname.clone(),
                res_code: aa,
                res_class: ResidueClass::Protein,
                chain,
                res_seq,
                icode,
                coords: *coords,
                occupancy: 1.0,
                b_factor: bf,
                formal_charge: approx_atom_charge(ResidueClass::Protein, aa, aname),
                is_hetatm: false,
            });
        }
    }
    for (i, a) in new_atoms.iter_mut().enumerate() {
        a.serial = i + 1;
    }

    // Evaluate steric clashes (ChimeraX 0.60 Å VDW overlap cutoff) & polar H-bonds
    let mut clash_count = 0usize;
    let mut clashes_ligand = 0usize;
    let mut clashes_protein = 0usize;
    let mut max_overlap = 0.0_f64;
    let mut hbond_count = 0usize;
    let mut clash_overlays = Vec::new();
    for &si in &inserted_indices {
        let sa = &new_atoms[si];
        let rsa = vdw_radius(&sa.element);
        for (oi, oa) in new_atoms.iter().enumerate() {
            if oi == si
                || oa.res_class == ResidueClass::Water
                || oa.element == "H"
                || (oa.chain == chain && oa.res_seq == res_seq)
            {
                continue;
            }
            // Skip immediate peptide-bond neighbor backbone clashes with CB
            if sa.name == "CB"
                && oa.chain == chain
                && (oa.res_seq - res_seq).abs() == 1
                && matches!(oa.name.as_str(), "N" | "C" | "O")
            {
                continue;
            }
            let d = dist3(sa.coords, oa.coords);
            let overlap = (rsa + vdw_radius(&oa.element)) - d;
            let polar_pair =
                matches!(sa.element.as_str(), "N" | "O") && matches!(oa.element.as_str(), "N" | "O");
            if overlap >= 0.60 && !(polar_pair && d >= 2.35) {
                clash_count += 1;
                max_overlap = max_overlap.max(overlap);
                if oa.res_class == ResidueClass::Ligand {
                    clashes_ligand += 1;
                } else {
                    clashes_protein += 1;
                }
                if clash_overlays.len() < 16 {
                    clash_overlays.push(SphereOverlay {
                        label: format!(
                            "Clash {}{}:{}..{}{} ({:.2} A)",
                            sa.res_code, sa.res_seq, sa.name, oa.res_name, oa.res_seq, overlap
                        ),
                        category: "Clash".to_string(),
                        coords: [
                            0.5 * (sa.coords[0] + oa.coords[0]),
                            0.5 * (sa.coords[1] + oa.coords[1]),
                            0.5 * (sa.coords[2] + oa.coords[2]),
                        ],
                        radius_a: (0.45 + 0.30 * overlap).clamp(0.45, 1.25),
                        rgba: [0.96, 0.18, 0.22, 0.78],
                    });
                }
            } else if d <= 3.45 && polar_pair {
                hbond_count += 1;
            }
        }
    }

    let rebuilt = finalize_biomolecule(
        mol.id.clone(),
        mol.title.clone(),
        new_atoms,
        Vec::new(),
        mol.is_alphafold_plddt,
    )?;
    *mol = rebuilt;

    Ok(Mutation3DReport {
        chain,
        res_seq,
        wild_type: wt,
        mutant: aa,
        rotamer_index: r_idx,
        total_rotamers: rots.len(),
        phi_deg,
        psi_deg,
        backbone_bin,
        chi1_deg: rot.chi1,
        chi2_deg: rot.chi2,
        rotamer_probability_pct: rot.probability_pct,
        clash_count,
        clashes_ligand,
        clashes_protein,
        max_overlap_a: (max_overlap * 100.0).round() / 100.0,
        new_hbond_count: hbond_count,
        clash_overlays,
    })
}

/// Evaluate ALL backbone-dependent Dunbrack rotamers at `(chain, res_seq) -> target_aa`
/// without modifying `mol`, returning a complete comparison table (PRAC3 Part 3).
pub fn enumerate_rotamers_3d(
    mol: &Biomolecule,
    chain: char,
    res_seq: i32,
    target_aa: char,
) -> Result<RotamerTableReport, ModelError> {
    let aa = target_aa.to_ascii_uppercase();
    if !"ACDEFGHIKLMNPQRSTVWY".contains(aa) {
        return Err(ModelError::InvalidAminoAcid(target_aa));
    }
    let (phi_deg, psi_deg) = residue_backbone_phi_psi(mol, chain, res_seq);
    let backbone_bin = classify_backbone_bin(phi_deg, psi_deg);
    let rots = dunbrack_rotamers(aa, backbone_bin);

    let mut wt = 'X';
    let mut candidates = Vec::with_capacity(rots.len());
    let mut best_idx = 0usize;
    let mut best_score = f64::INFINITY;

    for (r_idx, rot) in rots.iter().enumerate() {
        let mut probe = mol.clone();
        let rep = mutate_residue_3d(&mut probe, chain, res_seq, aa, r_idx)?;
        wt = rep.wild_type;
        let sc = (rep.clash_count as f64) * 10.0 + rep.max_overlap_a * 4.0
            - (rep.new_hbond_count as f64) * 2.5
            - (rot.probability_pct * 0.02);
        if sc < best_score {
            best_score = sc;
            best_idx = r_idx;
        }
        candidates.push(RotamerCandidate {
            rotamer_index: r_idx,
            chi1_deg: rot.chi1,
            chi2_deg: rot.chi2,
            probability_pct: rot.probability_pct,
            clashes_ligand: rep.clashes_ligand,
            clashes_protein: rep.clashes_protein,
            total_clashes: rep.clash_count,
            max_overlap_a: rep.max_overlap_a,
            new_hbonds: rep.new_hbond_count,
        });
    }

    Ok(RotamerTableReport {
        chain,
        res_seq,
        wild_type: wt,
        mutant: aa,
        phi_deg,
        psi_deg,
        backbone_bin,
        candidates,
        best_rotamer_index: best_idx,
    })
}

/// Render a Markdown Dunbrack rotamer table (exact ChimeraX `swapaa` / PRAC3 Part 3 format).
#[must_use]
pub fn render_rotamers_markdown(rep: &RotamerTableReport) -> String {
    let mut md = String::new();
    let phi_s = rep
        .phi_deg
        .map_or_else(|| "—".to_string(), |v| format!("{v:.1}°"));
    let psi_s = rep
        .psi_deg
        .map_or_else(|| "—".to_string(), |v| format!("{v:.1}°"));
    let _ = writeln!(
        md,
        "## Dunbrack Rotamer Table: `{}:{}{}{}`\n",
        rep.chain, rep.wild_type, rep.res_seq, rep.mutant
    );
    let _ = writeln!(
        md,
        "- **Backbone:** `φ = {phi_s}`, `ψ = {psi_s}` (`{}`) | **Library Rotamers:** `{}` | **Best Auto:** `#{}`\n",
        rep.backbone_bin.label(),
        rep.candidates.len(),
        rep.best_rotamer_index + 1
    );
    md.push_str("| Rot # | Prob (%) | χ1 (°) | χ2 (°) | Clashes (Lig) | Clashes (Prot) | Max Overlap (Å) | New H-Bonds |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for c in &rep.candidates {
        let c2 = c
            .chi2_deg
            .map_or_else(|| "—".to_string(), |v| format!("{v:.0}"));
        let mark = if c.rotamer_index == rep.best_rotamer_index {
            " ★"
        } else {
            ""
        };
        let _ = writeln!(
            md,
            "| #{}{} | {:.1} | {:.0} | {} | {} | {} | {:.2} | {} |",
            c.rotamer_index + 1,
            mark,
            c.probability_pct,
            c.chi1_deg,
            c2,
            c.clashes_ligand,
            c.clashes_protein,
            c.max_overlap_a,
            c.new_hbonds
        );
    }
    md
}

/// Automatically evaluate all backbone-dependent Dunbrack rotamers and apply the clash-minimum rotamer.
pub fn mutate_and_repack_best_rotamer(
    mol: &mut Biomolecule,
    chain: char,
    res_seq: i32,
    target_aa: char,
) -> Result<Mutation3DReport, ModelError> {
    let table = enumerate_rotamers_3d(mol, chain, res_seq, target_aa)?;
    mutate_residue_3d(mol, chain, res_seq, target_aa, table.best_rotamer_index)
}

/// Local ISOLDE-lite steepest-descent Molecular Mechanics relaxation (soft-LJ clash relief +
/// covalent bond spring preservation + harmonic backbone/selection restraint).
pub fn minimize_local_mm(
    mol: &mut Biomolecule,
    active_mask: &[bool],
    steps: usize,
) -> MinimizationReport {
    let n = mol.atoms.len();
    let mobile: Vec<usize> = (0..n)
        .filter(|&i| active_mask.get(i).copied().unwrap_or(true))
        .collect();
    if mobile.is_empty() || steps == 0 {
        return MinimizationReport {
            atoms_relaxed: 0,
            steps: 0,
            energy_before_kcal: 0.0,
            energy_after_kcal: 0.0,
            max_displacement_a: 0.0,
        };
    }

    let orig_coords: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.coords).collect();
    let bond_eq: Vec<(usize, usize, f64)> = mol
        .bonds
        .iter()
        .map(|b| {
            (
                b.a,
                b.b,
                dist3(orig_coords[b.a], orig_coords[b.b]).clamp(1.1, 2.6),
            )
        })
        .collect();

    // Collect non-bonded neighbor pairs involving at least one mobile atom within 5.5 A
    let mut is_mob = vec![false; n];
    for &i in &mobile {
        is_mob[i] = true;
    }
    let mut bonded_pairs = std::collections::HashSet::with_capacity(bond_eq.len() * 2);
    for &(a, b, _) in &bond_eq {
        bonded_pairs.insert((a.min(b), a.max(b)));
    }
    let mut nb_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for &i in &mobile {
        let ri = vdw_radius(&mol.atoms[i].element);
        for j in 0..n {
            if j == i || (is_mob[j] && j < i) {
                continue;
            }
            if bonded_pairs.contains(&(i.min(j), i.max(j))) {
                continue;
            }
            let d = dist3(mol.atoms[i].coords, mol.atoms[j].coords);
            if d <= 5.0 {
                let rmin = (ri + vdw_radius(&mol.atoms[j].element)) * 0.85;
                nb_pairs.push((i, j, rmin));
            }
        }
    }

    let eval_energy_and_grad = |coords: &[[f64; 3]], grad: &mut [[f64; 3]]| -> f64 {
        for g in grad.iter_mut() {
            *g = [0.0; 3];
        }
        let mut e = 0.0_f64;
        // 1. Harmonic covalent bond restraints (k = 120 kcal/mol/A^2)
        for &(a, b, r0) in &bond_eq {
            if !is_mob[a] && !is_mob[b] {
                continue;
            }
            let dv = sub3(coords[a], coords[b]);
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
                .sqrt()
                .max(1e-4);
            let dr = d - r0;
            e += 60.0 * dr * dr;
            let f = 120.0 * dr / d;
            for k in 0..3 {
                if is_mob[a] {
                    grad[a][k] += f * dv[k];
                }
                if is_mob[b] {
                    grad[b][k] -= f * dv[k];
                }
            }
        }
        // 2. Soft steric clash repulsion for d < rmin (smooth quadratic barrier)
        for &(i, j, rmin) in &nb_pairs {
            let dv = sub3(coords[i], coords[j]);
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
                .sqrt()
                .max(0.25);
            if d < rmin {
                let pen = rmin - d;
                e += 25.0 * pen * pen;
                let f = -50.0 * pen / d;
                for k in 0..3 {
                    if is_mob[i] {
                        grad[i][k] += f * dv[k];
                    }
                    if is_mob[j] {
                        grad[j][k] -= f * dv[k];
                    }
                }
            }
        }
        // 3. Soft backbone tether (keeps CA/P anchored gently while sidechains relax)
        for &i in &mobile {
            let k_tether = if matches!(
                mol.atoms[i].name.as_str(),
                "N" | "CA" | "C" | "O" | "P" | "C4'"
            ) {
                12.0
            } else {
                0.5
            };
            let dv = sub3(coords[i], orig_coords[i]);
            e += 0.5 * k_tether * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
            for k in 0..3 {
                grad[i][k] += k_tether * dv[k];
            }
        }
        e
    };

    let mut coords = orig_coords.clone();
    let mut grad = vec![[0.0; 3]; n];
    let e_before = eval_energy_and_grad(&coords, &mut grad);
    let mut e_cur = e_before;
    let mut step_size = 0.004_f64;

    for _ in 0..steps.min(300) {
        let mut trial = coords.clone();
        for &i in &mobile {
            let gn = (grad[i][0] * grad[i][0] + grad[i][1] * grad[i][1] + grad[i][2] * grad[i][2])
                .sqrt();
            if gn > 1e-6 {
                let max_move = (step_size * gn).min(0.15);
                let scale = max_move / gn;
                for k in 0..3 {
                    trial[i][k] -= grad[i][k] * scale;
                }
            }
        }
        let mut next_grad = vec![[0.0; 3]; n];
        let e_next = eval_energy_and_grad(&trial, &mut next_grad);
        if e_next <= e_cur {
            coords = trial;
            grad = next_grad;
            e_cur = e_next;
            step_size = (step_size * 1.15).min(0.03);
        } else {
            step_size *= 0.5;
            if step_size < 1e-6 {
                break;
            }
        }
    }

    let mut max_disp = 0.0_f64;
    for &i in &mobile {
        let d = dist3(coords[i], orig_coords[i]);
        if d > max_disp {
            max_disp = d;
        }
        mol.atoms[i].coords = coords[i];
    }

    MinimizationReport {
        atoms_relaxed: mobile.len(),
        steps: steps.min(300),
        energy_before_kcal: (e_before * 100.0).round() / 100.0,
        energy_after_kcal: (e_cur * 100.0).round() / 100.0,
        max_displacement_a: (max_disp * 1000.0).round() / 1000.0,
    }
}

/// Run a comprehensive 3D biomolecular preparation & sanity audit.
#[must_use]
pub fn audit_structure_prep(mol: &Biomolecule) -> StructurePrepAudit {
    let mut prot = Vec::new();
    let mut nuc = Vec::new();
    let mut ligs = 0usize;
    let mut ions = 0usize;
    let mut waters = 0usize;
    for ch in &mol.chains {
        match ch.dominant_class {
            ResidueClass::Protein => prot.push(ch.id),
            ResidueClass::Rna | ResidueClass::Dna => nuc.push(ch.id),
            _ => {}
        }
        for r in &ch.residues {
            match r.class {
                ResidueClass::Ligand => ligs += 1,
                ResidueClass::Ion => ions += 1,
                ResidueClass::Water => waters += 1,
                _ => {}
            }
        }
    }
    // Sample severe inter-residue heavy-atom clashes (< 1.85 A)
    let mut clash_pairs = 0usize;
    for i in 0..mol.atoms.len() {
        let a = &mol.atoms[i];
        if a.res_class == ResidueClass::Water {
            continue;
        }
        for b in &mol.atoms[(i + 1)..] {
            if b.res_class == ResidueClass::Water
                || (a.chain == b.chain && (a.res_seq - b.res_seq).abs() <= 1)
            {
                continue;
            }
            if (a.coords[0] - b.coords[0]).abs() < 1.85 && dist3(a.coords, b.coords) < 1.85 {
                clash_pairs += 1;
            }
        }
    }
    let mut notes = Vec::new();
    if clash_pairs > 0 {
        notes.push(format!("{clash_pairs} inter-residue heavy-atom clash(es) < 1.85 Å detected — run `minimize` to relieve."));
    } else {
        notes.push("No severe heavy-atom steric clashes (< 1.85 Å) detected.".to_string());
    }
    if !nuc.is_empty() {
        notes.push(format!(
            "Nucleic acid chain(s) {:?} detected with {} 3D base pairs.",
            nuc,
            mol.base_pairs.len()
        ));
    }
    StructurePrepAudit {
        total_atoms: mol.atoms.len(),
        protein_chains: prot,
        nucleic_chains: nuc,
        ligand_residues: ligs,
        ion_count: ions,
        water_count: waters,
        disulfide_count: mol.disulfides.len(),
        rna_base_pair_count: mol.base_pairs.len(),
        steric_clash_pairs: clash_pairs,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use craftbio_struct::parse_pdb;

    #[test]
    fn mutates_residue_with_dunbrack_rotamer_and_minimizes_clash()
    -> Result<(), Box<dyn std::error::Error>> {
        let pdb = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.210  1.00 20.00           C
ATOM      6  CA  GLY B   1       3.200  -1.600  -2.200  1.00 20.00           C
END
";
        let mut mol = parse_pdb("mut_test", pdb)?;
        let rep = mutate_residue_3d(&mut mol, 'A', 1, 'Y', 0)?;
        assert_eq!(rep.wild_type, 'A');
        assert_eq!(rep.mutant, 'Y');
        assert!(
            mol.atoms
                .iter()
                .any(|a| a.name == "OH" && a.res_name == "TYR")
        );

        let mask = vec![true; mol.atoms.len()];
        let min_rep = minimize_local_mm(&mut mol, &mask, 40);
        assert!(min_rep.energy_after_kcal <= min_rep.energy_before_kcal + 1e-6);
        Ok(())
    }

    #[test]
    fn wave6_swapna_bond_rot_tug_builders_and_edit_ops() -> Result<(), Box<dyn std::error::Error>> {
        // 1. De novo 3D A-form RNA duplex + swapna G -> A
        let mut rna = build_de_novo_rna_helix_3d("GCGCUAGC", 'X')?;
        assert_eq!(rna.chains.len(), 2);
        let sw = swapna_nucleotide_3d(&mut rna, 'X', 1, 'A')?;
        assert_eq!(sw.wild_type, 'G');
        assert_eq!(sw.mutant, 'A');
        assert!(
            rna.atoms
                .iter()
                .any(|a| a.chain == 'X' && a.res_seq == 1 && a.name == "N6")
        );

        // 2. De novo 3D alpha-helix + addh + bond_rot + tug + renumber + rename + combine + delete
        let mut pep = build_de_novo_peptide_3d("AKLTYWDSHE", "helix", 'A')?;
        assert!(pep.atoms.len() > 60);
        let hr = add_polar_hydrogens_ph(&mut pep, 7.4);
        assert!(hr.hydrogens_added >= 8);

        // Rotate CA->CB sidechain bond of Tyr5 (acyclic rotatable bond!)
        let ca_i = pep
            .atoms
            .iter()
            .position(|a| a.res_seq == 5 && a.name == "CA")
            .ok_or("missing CA")?;
        let cb_i = pep
            .atoms
            .iter()
            .position(|a| a.res_seq == 5 && a.name == "CB")
            .ok_or("missing CB")?;
        let br = rotate_bond_torsion(&mut pep, ca_i, cb_i, 45.0)?;
        assert!(br.rotated_atoms >= 6);

        let tug = tug_atom_with_sculpting(&mut pep, cb_i, [0.8, 0.0, 0.0], 5.5, 25)?;
        assert!(tug.atoms_sculpted >= 5);

        let rcnt = renumber_chain(&mut pep, 'A', 101)?;
        assert_eq!(rcnt, 10);
        let moved = rename_chain(&mut pep, 'A', 'H')?;
        assert!(moved > 50);

        let combined = combine_biomolecules(&[&pep, &rna], "pep_rna_combo")?;
        assert_eq!(combined.chains.len(), 3);

        let mut trimmed = pep.clone();
        let del_mask: Vec<bool> = trimmed.atoms.iter().map(|a| a.res_seq == 110).collect();
        let del = delete_atoms_by_mask(&mut trimmed, &del_mask)?;
        assert!(del > 0);
        Ok(())
    }

    #[test]
    fn wave9_dunbrack_chi1_chi2_ramachandran_aggregation_and_structure_info()
    -> Result<(), Box<dyn std::error::Error>> {
        // 1. Exact NeRF chi1/chi2 fidelity on Isoleucine rotamers (PRAC3 T315I)
        let mut pep = build_de_novo_peptide_3d("AAKTIAA", "helix", 'A')?;
        let table = enumerate_rotamers_3d(&pep, 'A', 4, 'I')?;
        assert_eq!(table.candidates.len(), 7);
        assert_eq!(parse_aa_code("ile"), Some('I'));

        let rep = mutate_residue_3d(&mut pep, 'A', 4, 'I', 0)?;
        let (m_c1, m_c2) = measure_residue_chi1_chi2(&pep, 'A', 4);
        let c1 = m_c1.ok_or("missing chi1")?;
        let c2 = m_c2.ok_or("missing chi2")?;
        let target_c2 = rep.chi2_deg.ok_or("missing target chi2")?;
        assert!(
            (c1 - rep.chi1_deg).abs() < 0.5,
            "chi1 mismatch: measured {c1} vs target {}",
            rep.chi1_deg
        );
        assert!(
            (c2 - target_c2).abs() < 0.5,
            "chi2 mismatch: measured {c2} vs target {target_c2}"
        );

        // 2. Ramachandran plot (General vs Glycine vs Proline)
        let rama = analyze_ramachandran(&pep, None);
        assert!(rama.points.len() >= 5);
        assert_eq!(
            classify_ramachandran(RamaResidueKind::General, -63.0, -42.0).0,
            RamaRegion::Favored
        );
        assert_eq!(
            classify_ramachandran(RamaResidueKind::General, 68.0, -55.0).0,
            RamaRegion::Outlier
        );
        assert_ne!(
            classify_ramachandran(RamaResidueKind::Glycine, 68.0, -55.0).0,
            RamaRegion::Outlier
        );
        assert_eq!(
            classify_ramachandran(RamaResidueKind::Proline, 60.0, 40.0).0,
            RamaRegion::Outlier
        );

        // 3. General 3D Surface Aggregation & Solubility Rescue (TRAB4 Tool 2)
        let hydro_pep = build_de_novo_peptide_3d("AFIFVLFIA", "helix", 'A')?;
        let agg = analyze_aggregation_3d(&hydro_pep, false);
        assert!(!agg.residue_scores.is_empty());
        assert!(agg.max_score > 0.5);
        let rescue = simulate_solubility_rescue(&hydro_pep, 'A', 4, 'D', false)?;
        assert!(
            rescue.after_avg_score < rescue.before_avg_score,
            "expected F4D mutation to reduce global aggregation score ({} -> {})",
            rescue.before_avg_score,
            rescue.after_avg_score
        );

        // 4. Structure Info Audit (gaps & truncated sidechains, PRAC1 Part 5)
        let gap_pdb = "\
ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 18.00           N
ATOM      2  CA  ALA A  10       1.450   0.000   0.000  1.00 18.00           C
ATOM      3  C   ALA A  10       2.050   1.380   0.000  1.00 18.00           C
ATOM      4  O   ALA A  10       1.350   2.380   0.000  1.00 18.00           O
ATOM      5  CB  ALA A  10       1.980  -0.760  -1.210  1.00 18.00           C
ATOM      6  N   LYS A  15       8.000   0.000   0.000  1.00 65.00           N
ATOM      7  CA  LYS A  15       9.450   0.000   0.000  1.00 65.00           C
ATOM      8  C   LYS A  15      10.050   1.380   0.000  1.00 65.00           C
ATOM      9  O   LYS A  15       9.350   2.380   0.000  1.00 65.00           O
ATOM     10  CB  LYS A  15       9.980  -0.760  -1.210  1.00 65.00           C
END
";
        let gmol = parse_pdb("gap_test", gap_pdb)?;
        let info = inspect_structure_info(&gmol);
        assert_eq!(info.missing_gaps.len(), 1);
        assert_eq!(info.missing_gaps[0].missing_count, 4);
        assert_eq!(info.truncated_sidechains.len(), 1);
        assert_eq!(info.truncated_sidechains[0].res_name, "LYS");
        assert!(
            info.truncated_sidechains[0]
                .missing_atoms
                .iter()
                .any(|a| a == "NZ")
        );
        Ok(())
    }
}
