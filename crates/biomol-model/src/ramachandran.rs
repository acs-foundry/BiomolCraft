#![forbid(unsafe_code)]

//! Pure-Rust Ramachandran `(φ, ψ, ω)` Backbone Dihedral & Stereochemical Quality Engine.
//!
//! Computes exact IUPAC backbone torsion angles per residue (respecting chain breaks via
//! peptide `C–N <= 2.1 Å` continuity), classifies residues into Richardson/MolProbity-style
//! categories (`General`, `Glycine`, `Proline`, `PreProline`), grades each point as
//! `Favored`, `Allowed`, or `Outlier`, detects `cis`-peptide bonds (`|ω| < 30°`), and
//! evaluates teaching contour maps for interactive 2D visualization.

use std::fmt::Write as _;

use craftbio_struct::{Biomolecule, ResidueClass, SecondaryStructure, dist3, measure_dihedral_deg};
use serde::{Deserialize, Serialize};

/// Ramachandran steric residue class (Richardson / MolProbity convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RamaResidueKind {
    General,
    Glycine,
    Proline,
    PreProline,
}

impl RamaResidueKind {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Glycine => "Glycine",
            Self::Proline => "Proline",
            Self::PreProline => "Pre-Proline",
        }
    }
}

/// Ramachandran conformational region grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RamaRegion {
    Favored,
    Allowed,
    Outlier,
}

impl RamaRegion {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Favored => "Favored",
            Self::Allowed => "Allowed",
            Self::Outlier => "Outlier",
        }
    }
}

/// Per-residue backbone `(φ, ψ, ω)` measurement and stereochemical classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamaPoint {
    pub chain: char,
    pub res_seq: i32,
    pub res_name: String,
    pub res_code: char,
    pub kind: RamaResidueKind,
    pub ss: SecondaryStructure,
    pub phi_deg: f64,
    pub psi_deg: f64,
    pub omega_deg: Option<f64>,
    pub is_cis_peptide: bool,
    pub region: RamaRegion,
    pub density_score: f64,
}

/// Complete Ramachandran stereochemical audit report for a biomolecule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamachandranReport {
    pub title: String,
    pub points: Vec<RamaPoint>,
    pub favored_count: usize,
    pub allowed_count: usize,
    pub outlier_count: usize,
    pub favored_pct: f64,
    pub allowed_pct: f64,
    pub outlier_pct: f64,
    pub cis_peptide_count: usize,
    pub cis_proline_count: usize,
}

fn wrapped_angle_diff(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    } else if d < -180.0 {
        d += 360.0;
    }
    d
}

fn torus_gaussian(phi: f64, psi: f64, mu_phi: f64, mu_psi: f64, sig_phi: f64, sig_psi: f64) -> f64 {
    let dp = wrapped_angle_diff(phi, mu_phi) / sig_phi;
    let dq = wrapped_angle_diff(psi, mu_psi) / sig_psi;
    (-0.5 * (dp * dp + dq * dq)).exp()
}

/// Continuous smoothed Ramachandran density (`>= 0.0`) on the `[-180, 180]²` torus for `kind`.
///
/// Basins approximate the Top500 / MolProbity distributions (Lovell et al., *Proteins* 2003)
/// for teaching and fast stereochemical screening.
#[must_use]
pub fn ramachandran_density(kind: RamaResidueKind, phi: f64, psi: f64) -> f64 {
    let g = |mp: f64, mq: f64, sp: f64, sq: f64, w: f64| -> f64 {
        w * torus_gaussian(phi, psi, mp, mq, sp, sq)
    };
    match kind {
        RamaResidueKind::General => {
            let mut s = 0.0;
            // Right-handed alpha-helix & 3_10 helix
            s += g(-63.0, -42.0, 22.0, 22.0, 1.00);
            s += g(-88.0, -12.0, 26.0, 28.0, 0.52);
            // Antiparallel / parallel beta-sheet & PPII
            s += g(-122.0, 134.0, 32.0, 32.0, 0.96);
            s += g(-72.0, 145.0, 24.0, 26.0, 0.85);
            // Left-handed alpha-helix (small allowed/favored island)
            s += g(58.0, 38.0, 17.0, 22.0, 0.36);
            // Gamma-turn tiny bridge
            s += g(-82.0, 65.0, 18.0, 22.0, 0.10);
            s
        }
        RamaResidueKind::Glycine => {
            let mut s = 0.0;
            // Symmetric left/right alpha & beta/PPII basins (no C-beta steric barrier)
            s += g(-68.0, -38.0, 28.0, 30.0, 0.95);
            s += g(68.0, 38.0, 28.0, 30.0, 0.95);
            s += g(-105.0, 145.0, 42.0, 38.0, 0.90);
            s += g(105.0, -145.0, 42.0, 38.0, 0.90);
            s += g(86.0, 0.0, 26.0, 34.0, 0.55);
            s += g(-86.0, 0.0, 26.0, 34.0, 0.55);
            s += g(70.0, -55.0, 26.0, 26.0, 0.25);
            s += g(-70.0, 55.0, 26.0, 26.0, 0.25);
            s
        }
        RamaResidueKind::Proline => {
            let mut s = 0.0;
            // Pyrrolidine ring constrains phi strictly near -63 deg
            s += g(-63.0, -36.0, 13.0, 24.0, 1.00);
            s += g(-65.0, 146.0, 13.0, 28.0, 1.00);
            s += g(-64.0, 8.0, 11.0, 22.0, 0.22);
            s
        }
        RamaResidueKind::PreProline => {
            let mut s = 0.0;
            // Steric clash with following Pro C-delta suppresses right-half alpha and enlarges zeta
            s += g(-122.0, 138.0, 30.0, 30.0, 1.00);
            s += g(-74.0, 146.0, 22.0, 24.0, 0.88);
            s += g(-68.0, -38.0, 19.0, 19.0, 0.72);
            s += g(-128.0, 68.0, 22.0, 22.0, 0.44);
            s += g(56.0, 44.0, 14.0, 16.0, 0.14);
            s
        }
    }
}

/// Classify a single `(φ, ψ)` point for `kind` into `Favored`, `Allowed`, or `Outlier`.
#[must_use]
pub fn classify_ramachandran(kind: RamaResidueKind, phi: f64, psi: f64) -> (RamaRegion, f64) {
    let score = ramachandran_density(kind, phi, psi);
    let region = if score >= 0.22 {
        RamaRegion::Favored
    } else if score >= 0.045 {
        RamaRegion::Allowed
    } else {
        RamaRegion::Outlier
    };
    (region, (score * 1000.0).round() / 1000.0)
}

fn residue_atom_coords(mol: &Biomolecule, atom_indices: &[usize], name: &str) -> Option<[f64; 3]> {
    atom_indices
        .iter()
        .copied()
        .find(|&i| mol.atoms[i].name == name)
        .map(|i| mol.atoms[i].coords)
}

/// Compute exact backbone `(φ, ψ, ω)` Ramachandran audit for all covalently continuous
/// protein residues in `mol` (optionally restricted to `chain_filter`).
#[must_use]
pub fn analyze_ramachandran(
    mol: &Biomolecule,
    chain_filter: Option<char>,
) -> RamachandranReport {
    let mut points = Vec::new();
    let mut favored_count = 0usize;
    let mut allowed_count = 0usize;
    let mut outlier_count = 0usize;
    let mut cis_peptide_count = 0usize;
    let mut cis_proline_count = 0usize;

    for ch in &mol.chains {
        if let Some(cf) = chain_filter
            && ch.id != cf
        {
            continue;
        }
        let prot: Vec<_> = ch
            .residues
            .iter()
            .filter(|r| r.class == ResidueClass::Protein)
            .collect();
        if prot.len() < 3 {
            continue;
        }
        for i in 1..(prot.len() - 1) {
            let prev = prot[i - 1];
            let cur = prot[i];
            let next = prot[i + 1];

            let (Some(c_prev), Some(n_cur), Some(ca_cur), Some(c_cur), Some(n_next)) = (
                residue_atom_coords(mol, &prev.atom_indices, "C"),
                residue_atom_coords(mol, &cur.atom_indices, "N"),
                residue_atom_coords(mol, &cur.atom_indices, "CA"),
                residue_atom_coords(mol, &cur.atom_indices, "C"),
                residue_atom_coords(mol, &next.atom_indices, "N"),
            ) else {
                continue;
            };

            // Enforce peptide bond continuity (ignore gaps / missing loops!)
            if dist3(c_prev, n_cur) > 2.1 || dist3(c_cur, n_next) > 2.1 {
                continue;
            }

            let phi = measure_dihedral_deg(c_prev, n_cur, ca_cur, c_cur);
            let psi = measure_dihedral_deg(n_cur, ca_cur, c_cur, n_next);
            let omega = residue_atom_coords(mol, &prev.atom_indices, "CA")
                .map(|ca_prev| measure_dihedral_deg(ca_prev, c_prev, n_cur, ca_cur));
            let is_cis = omega.is_some_and(|w| w.abs() < 30.0);
            if is_cis {
                cis_peptide_count += 1;
                if cur.code == 'P' {
                    cis_proline_count += 1;
                }
            }

            let kind = match cur.code {
                'G' => RamaResidueKind::Glycine,
                'P' => RamaResidueKind::Proline,
                _ if next.code == 'P' => RamaResidueKind::PreProline,
                _ => RamaResidueKind::General,
            };
            let (region, density_score) = classify_ramachandran(kind, phi, psi);
            match region {
                RamaRegion::Favored => favored_count += 1,
                RamaRegion::Allowed => allowed_count += 1,
                RamaRegion::Outlier => outlier_count += 1,
            }

            let r1 = |v: f64| (v * 10.0).round() / 10.0;
            points.push(RamaPoint {
                chain: ch.id,
                res_seq: cur.res_seq,
                res_name: cur.res_name.clone(),
                res_code: cur.code,
                kind,
                ss: cur.ss,
                phi_deg: r1(phi),
                psi_deg: r1(psi),
                omega_deg: omega.map(r1),
                is_cis_peptide: is_cis,
                region,
                density_score,
            });
        }
    }

    let total = points.len().max(1) as f64;
    let pct = |c: usize| ((c as f64 / total) * 1000.0).round() / 10.0;
    RamachandranReport {
        title: mol.title.clone(),
        points,
        favored_count,
        allowed_count,
        outlier_count,
        favored_pct: pct(favored_count),
        allowed_pct: pct(allowed_count),
        outlier_pct: pct(outlier_count),
        cis_peptide_count,
        cis_proline_count,
    }
}

/// Render a clean Markdown Ramachandran quality table (ideal for Log, CLI & teaching dossiers).
#[must_use]
pub fn render_ramachandran_markdown(rep: &RamachandranReport) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "## Ramachandran Backbone Audit — {}\n", rep.title);
    let _ = writeln!(
        md,
        "- **Evaluated Residues:** `{}` | **Favored:** `{}` (`{:.1}%`) | **Allowed:** `{}` (`{:.1}%`) | **Outliers:** `{}` (`{:.1}%`)",
        rep.points.len(),
        rep.favored_count,
        rep.favored_pct,
        rep.allowed_count,
        rep.allowed_pct,
        rep.outlier_count,
        rep.outlier_pct
    );
    let _ = writeln!(
        md,
        "- **cis-Peptide Bonds (`|ω| < 30°`):** `{}` (`{}` cis-Prolines)\n",
        rep.cis_peptide_count, rep.cis_proline_count
    );

    let outliers: Vec<&RamaPoint> = rep
        .points
        .iter()
        .filter(|p| p.region == RamaRegion::Outlier)
        .collect();
    if outliers.is_empty() {
        md.push_str("No Ramachandran outliers detected.\n");
    } else {
        md.push_str("### Ramachandran Outliers\n\n| Residue | Class | φ (°) | ψ (°) | ω (°) |\n|---|---|---:|---:|---:|\n");
        for p in outliers.iter().take(30) {
            let om = p
                .omega_deg
                .map_or_else(|| "—".to_string(), |w| format!("{w:.1}"));
            let _ = writeln!(
                md,
                "| `{}:{}{}` | {} | {:.1} | {:.1} | {} |",
                p.chain,
                p.res_name,
                p.res_seq,
                p.kind.label(),
                p.phi_deg,
                p.psi_deg,
                om
            );
        }
    }
    md
}
