#![forbid(unsafe_code)]

//! L0 structural & sequence I/O: FASTA, PDB (`ATOM`/`SSBOND`/`REMARK`), mmCIF (`_atom_site`),
//! exact amino-acid biophysical constants, and Rayon-parallel 3D Shrake-Rupley SASA.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum atoms permitted in a single loaded biological structure for interactive/MCP safety.
pub const MAX_BIO_ATOMS: usize = 120_000;

#[derive(Debug, Error)]
pub enum BioError {
    #[error("empty biological input")]
    EmptyInput,
    #[error("invalid FASTA or sequence: {0}")]
    Sequence(String),
    #[error("invalid PDB/mmCIF structure: {0}")]
    Structure(String),
    #[error("structure exceeds safety limit of {MAX_BIO_ATOMS} atoms (got {0})")]
    TooManyAtoms(usize),
}

/// Single heavy atom inside a parsed 3D protein / antibody / antigen structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom3D {
    pub serial: usize,
    pub name: String,
    pub element: String,
    pub res_name: String,
    pub res_aa1: char,
    pub chain: char,
    pub res_seq: i32,
    pub icode: char,
    pub coords: [f64; 3],
    pub occupancy: f64,
    pub b_factor: f64,
    pub is_hetatm: bool,
}

/// Grouped 3D residue inside a chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residue3D {
    pub chain: char,
    pub res_seq: i32,
    pub icode: char,
    pub res_name: String,
    pub aa1: char,
    pub atom_indices: Vec<usize>,
    pub ca_idx: Option<usize>,
    pub cb_idx: Option<usize>,
    pub mean_bfactor: f64,
}

/// Disulfide bond detected from `SSBOND` header or `SG–SG <= 2.6 Å`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisulfideBond {
    pub chain_a: char,
    pub res_seq_a: i32,
    pub chain_b: char,
    pub res_seq_b: i32,
    pub distance_a: f64,
}

/// Chain summary with 1-letter sequence and ordered residues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain3D {
    pub id: char,
    pub sequence: String,
    pub residues: Vec<Residue3D>,
}

/// Complete parsed 3D biological structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinStructure {
    pub title: String,
    pub atoms: Vec<Atom3D>,
    pub chains: Vec<Chain3D>,
    pub disulfides: Vec<DisulfideBond>,
    pub is_alphafold_plddt: bool,
}

impl ProteinStructure {
    pub fn chain(&self, id: char) -> Option<&Chain3D> {
        self.chains.iter().find(|c| c.id == id)
    }

    pub fn chain_ids(&self) -> Vec<char> {
        self.chains.iter().map(|c| c.id).collect()
    }
}

/// Convert standard & Amber/PROPKA/modified 3-letter residue codes to 1-letter IUPAC amino acids.
#[must_use]
pub fn aa3_to_aa1(res: &str) -> Option<char> {
    match res.trim().to_ascii_uppercase().as_str() {
        "ALA" => Some('A'),
        "ARG" => Some('R'),
        "ASN" => Some('N'),
        "ASP" | "ASH" => Some('D'),
        "CYS" | "CYX" | "CYM" => Some('C'),
        "GLN" => Some('Q'),
        "GLU" | "GLH" => Some('E'),
        "GLY" => Some('G'),
        "HIS" | "HID" | "HIE" | "HIP" => Some('H'),
        "ILE" => Some('I'),
        "LEU" => Some('L'),
        "LYS" | "LYN" => Some('K'),
        "MET" | "MSE" => Some('M'),
        "PHE" => Some('F'),
        "PRO" => Some('P'),
        "SER" | "SEP" => Some('S'),
        "THR" | "TPO" => Some('T'),
        "TRP" => Some('W'),
        "TYR" | "PTR" => Some('Y'),
        "VAL" => Some('V'),
        "PCA" | "PYR" => Some('Q'),
        _ => None,
    }
}

/// Convert 1-letter IUPAC amino acid to 3-letter PDB code.
#[must_use]
pub fn aa1_to_aa3(c: char) -> &'static str {
    match c.to_ascii_uppercase() {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        _ => "UNK",
    }
}

/// Clean an amino-acid sequence string to uppercase 20 standard 1-letter residues.
pub fn sanitize_protein_sequence(raw: &str) -> Result<String, BioError> {
    let seq: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    if seq.is_empty() {
        return Err(BioError::EmptyInput);
    }
    for ch in seq.chars() {
        if !"ACDEFGHIKLMNPQRSTVWY".contains(ch) {
            return Err(BioError::Sequence(format!(
                "unsupported amino-acid symbol '{ch}' (expected 20 canonical IUPAC codes)"
            )));
        }
    }
    Ok(seq)
}

/// Parse single- or multi-entry FASTA text (or plain raw sequence) into `(name, sequence)` pairs.
pub fn parse_fasta(text: &str) -> Result<Vec<(String, String)>, BioError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(BioError::EmptyInput);
    }
    if !trimmed.contains('>') {
        let seq = sanitize_protein_sequence(trimmed)?;
        return Ok(vec![("Chain_A".to_string(), seq)]);
    }
    let mut entries = Vec::new();
    let mut cur_name: Option<String> = None;
    let mut cur_seq = String::new();
    for line in trimmed.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Some(hdr) = l.strip_prefix('>') {
            if let Some(name) = cur_name.take()
                && !cur_seq.is_empty()
            {
                entries.push((name, sanitize_protein_sequence(&cur_seq)?));
                cur_seq.clear();
            }
            cur_name = Some(hdr.trim().to_string());
        } else {
            cur_seq.push_str(l);
        }
    }
    if let Some(name) = cur_name
        && !cur_seq.is_empty()
    {
        entries.push((name, sanitize_protein_sequence(&cur_seq)?));
    }
    if entries.is_empty() {
        return Err(BioError::EmptyInput);
    }
    Ok(entries)
}

fn infer_element(atom_name: &str, raw_elem: &str) -> String {
    let e = raw_elem.trim();
    if !e.is_empty() && e.chars().all(|c| c.is_ascii_alphabetic()) {
        return e.to_ascii_uppercase();
    }
    let a = atom_name
        .trim()
        .trim_start_matches(|c: char| c.is_ascii_digit());
    if let Some(ch) = a.chars().next() {
        ch.to_ascii_uppercase().to_string()
    } else {
        "C".to_string()
    }
}

/// Build grouped `Chain3D` and `DisulfideBond` structures from parsed `Atom3D` list.
fn finalize_structure(
    title: String,
    atoms: Vec<Atom3D>,
    is_alphafold_plddt: bool,
) -> Result<ProteinStructure, BioError> {
    if atoms.is_empty() {
        return Err(BioError::Structure("no protein atoms found".to_string()));
    }
    type ResKey = (char, i32, char);
    type ResAccum = (String, char, Vec<usize>);
    let mut chain_order: Vec<char> = Vec::new();
    let mut res_map: BTreeMap<ResKey, ResAccum> = BTreeMap::new();
    let mut res_order: HashMap<char, Vec<(i32, char)>> = HashMap::new();

    for (idx, a) in atoms.iter().enumerate() {
        if !chain_order.contains(&a.chain) {
            chain_order.push(a.chain);
        }
        let key = (a.chain, a.res_seq, a.icode);
        let entry = res_map.entry(key).or_insert_with(|| {
            res_order
                .entry(a.chain)
                .or_default()
                .push((a.res_seq, a.icode));
            (a.res_name.clone(), a.res_aa1, Vec::new())
        });
        entry.2.push(idx);
    }

    let mut chains = Vec::with_capacity(chain_order.len());
    for cid in chain_order {
        let mut residues = Vec::new();
        let mut seq = String::new();
        if let Some(keys) = res_order.get(&cid) {
            for &(rseq, icode) in keys {
                if let Some((rname, aa1, a_idxs)) = res_map.get(&(cid, rseq, icode)) {
                    let ca_idx = a_idxs.iter().copied().find(|&i| atoms[i].name == "CA");
                    let cb_idx = a_idxs.iter().copied().find(|&i| atoms[i].name == "CB");
                    let mean_b = if a_idxs.is_empty() {
                        0.0
                    } else {
                        a_idxs.iter().map(|&i| atoms[i].b_factor).sum::<f64>()
                            / (a_idxs.len() as f64)
                    };
                    seq.push(*aa1);
                    residues.push(Residue3D {
                        chain: cid,
                        res_seq: rseq,
                        icode,
                        res_name: rname.clone(),
                        aa1: *aa1,
                        atom_indices: a_idxs.clone(),
                        ca_idx,
                        cb_idx,
                        mean_bfactor: mean_b,
                    });
                }
            }
        }
        chains.push(Chain3D {
            id: cid,
            sequence: seq,
            residues,
        });
    }

    // Detect covalent disulfide bonds from SG–SG geometry (<= 2.65 A)
    let sg_atoms: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.res_aa1 == 'C' && (a.name == "SG" || a.element == "S"))
        .map(|(i, _)| i)
        .collect();
    let mut disulfides = Vec::new();
    for i in 0..sg_atoms.len() {
        for j in (i + 1)..sg_atoms.len() {
            let a = &atoms[sg_atoms[i]];
            let b = &atoms[sg_atoms[j]];
            let d = dist3(a.coords, b.coords);
            if (1.5..=2.65).contains(&d) {
                disulfides.push(DisulfideBond {
                    chain_a: a.chain,
                    res_seq_a: a.res_seq,
                    chain_b: b.chain,
                    res_seq_b: b.res_seq,
                    distance_a: (d * 100.0).round() / 100.0,
                });
            }
        }
    }

    Ok(ProteinStructure {
        title,
        atoms,
        chains,
        disulfides,
        is_alphafold_plddt,
    })
}

/// Parse a standard RCSB / AlphaFold PDB file into `ProteinStructure` (resolving altlocs by
/// highest occupancy and keeping standard heavy protein atoms only).
pub fn parse_pdb(pdb_text: &str) -> Result<ProteinStructure, BioError> {
    if pdb_text.trim().is_empty() {
        return Err(BioError::EmptyInput);
    }
    let mut title = String::new();
    let mut is_af = false;
    // Key: (chain, res_seq, icode, atom_name) -> best Atom3D (highest occupancy)
    let mut best_atoms: HashMap<(char, i32, char, String), Atom3D> = HashMap::new();
    let mut order_keys: Vec<(char, i32, char, String)> = Vec::new();

    for line in pdb_text.lines() {
        if line.starts_with("ENDMDL") && !order_keys.is_empty() {
            break; // First model only for NMR / multi-model bundles
        }
        if line.starts_with("TITLE") {
            let t = line.get(10..).unwrap_or("").trim();
            if !t.is_empty() {
                if !title.is_empty() {
                    title.push(' ');
                }
                title.push_str(t);
            }
            if t.to_ascii_uppercase().contains("ALPHAFOLD") {
                is_af = true;
            }
            continue;
        }
        if line.starts_with("REMARK") && line.to_ascii_uppercase().contains("ALPHAFOLD") {
            is_af = true;
            continue;
        }
        let is_atom = line.starts_with("ATOM  ");
        let is_het = line.starts_with("HETATM");
        if (!is_atom && !is_het) || line.len() < 54 {
            continue;
        }
        let res_name = line.get(17..20).unwrap_or("").trim();
        let Some(aa1) = aa3_to_aa1(res_name) else {
            continue;
        };
        let atom_name = line.get(12..16).unwrap_or("").trim().to_string();
        let raw_elem = line.get(76..78).unwrap_or("");
        let elem = infer_element(&atom_name, raw_elem);
        if elem == "H" || elem == "D" {
            continue;
        }
        let alt = line.as_bytes().get(16).copied().unwrap_or(b' ') as char;
        let chain = line.as_bytes().get(21).copied().unwrap_or(b'A') as char;
        let chain = if chain == ' ' { 'A' } else { chain };
        let res_seq: i32 = line.get(22..26).unwrap_or("0").trim().parse().unwrap_or(0);
        let icode = line.as_bytes().get(26).copied().unwrap_or(b' ') as char;
        let x: f64 = line
            .get(30..38)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(f64::NAN);
        let y: f64 = line
            .get(38..46)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(f64::NAN);
        let z: f64 = line
            .get(46..54)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(f64::NAN);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        let occ: f64 = line
            .get(54..60)
            .unwrap_or("1.0")
            .trim()
            .parse()
            .unwrap_or(1.0);
        let bf: f64 = line
            .get(60..66)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let key = (chain, res_seq, icode, atom_name.clone());
        let candidate = Atom3D {
            serial: 0,
            name: atom_name,
            element: elem,
            res_name: res_name.to_ascii_uppercase(),
            res_aa1: aa1,
            chain,
            res_seq,
            icode,
            coords: [x, y, z],
            occupancy: occ,
            b_factor: bf,
            is_hetatm: is_het,
        };
        if let Some(prev) = best_atoms.get(&key) {
            if occ > prev.occupancy || (occ == prev.occupancy && (alt == ' ' || alt == 'A')) {
                best_atoms.insert(key, candidate);
            }
        } else {
            if order_keys.len() >= MAX_BIO_ATOMS {
                return Err(BioError::TooManyAtoms(order_keys.len() + 1));
            }
            order_keys.push(key.clone());
            best_atoms.insert(key, candidate);
        }
    }

    let mut atoms = Vec::with_capacity(order_keys.len());
    for (i, k) in order_keys.into_iter().enumerate() {
        if let Some(mut a) = best_atoms.remove(&k) {
            a.serial = i + 1;
            atoms.push(a);
        }
    }
    if title.is_empty() {
        title = "PDB Structure".to_string();
    }
    finalize_structure(title, atoms, is_af)
}

/// Parse an mmCIF file (`_atom_site` loop) into `ProteinStructure`.
pub fn parse_mmcif(cif_text: &str) -> Result<ProteinStructure, BioError> {
    if cif_text.trim().is_empty() {
        return Err(BioError::EmptyInput);
    }
    let mut cols: Vec<String> = Vec::new();
    let mut in_atom_site = false;
    let mut atoms = Vec::new();

    for raw in cif_text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            if in_atom_site && !atoms.is_empty() {
                break;
            }
            continue;
        }
        if line.starts_with("loop_") {
            if in_atom_site && !atoms.is_empty() {
                break;
            }
            cols.clear();
            in_atom_site = false;
            continue;
        }
        if line.starts_with("_atom_site.") {
            in_atom_site = true;
            cols.push(line.to_string());
            continue;
        }
        if in_atom_site && line.starts_with('_') {
            if !atoms.is_empty() {
                break;
            }
            in_atom_site = false;
            continue;
        }
        if !in_atom_site {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < cols.len() {
            continue;
        }
        let col_val = |name: &str| -> Option<&str> {
            cols.iter()
                .position(|c| c == name)
                .and_then(|i| toks.get(i).copied())
        };
        let group = col_val("_atom_site.group_PDB").unwrap_or("ATOM");
        if group != "ATOM" && group != "HETATM" {
            continue;
        }
        let res_name = col_val("_atom_site.label_comp_id")
            .or_else(|| col_val("_atom_site.auth_comp_id"))
            .unwrap_or("");
        let Some(aa1) = aa3_to_aa1(res_name) else {
            continue;
        };
        let alt = col_val("_atom_site.label_alt_id").unwrap_or(".");
        if alt != "." && alt != "?" && alt != "A" {
            continue;
        }
        let aname = col_val("_atom_site.label_atom_id")
            .or_else(|| col_val("_atom_site.auth_atom_id"))
            .unwrap_or("CA")
            .trim_matches('"')
            .to_string();
        let elem = infer_element(&aname, col_val("_atom_site.type_symbol").unwrap_or(""));
        if elem == "H" || elem == "D" {
            continue;
        }
        let chain = col_val("_atom_site.auth_asym_id")
            .or_else(|| col_val("_atom_site.label_asym_id"))
            .and_then(|s| s.chars().next())
            .unwrap_or('A');
        let res_seq: i32 = col_val("_atom_site.auth_seq_id")
            .or_else(|| col_val("_atom_site.label_seq_id"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let x: f64 = col_val("_atom_site.Cartn_x")
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);
        let y: f64 = col_val("_atom_site.Cartn_y")
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);
        let z: f64 = col_val("_atom_site.Cartn_z")
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        let occ: f64 = col_val("_atom_site.occupancy")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let bf: f64 = col_val("_atom_site.B_iso_or_equiv")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        if atoms.len() >= MAX_BIO_ATOMS {
            return Err(BioError::TooManyAtoms(atoms.len() + 1));
        }
        let serial = atoms.len() + 1;
        atoms.push(Atom3D {
            serial,
            name: aname,
            element: elem,
            res_name: res_name.to_ascii_uppercase(),
            res_aa1: aa1,
            chain,
            res_seq,
            icode: ' ',
            coords: [x, y, z],
            occupancy: occ,
            b_factor: bf,
            is_hetatm: group == "HETATM",
        });
    }

    finalize_structure("mmCIF Structure".to_string(), atoms, false)
}

/// Auto-detect PDB vs mmCIF and parse.
pub fn parse_structure_auto(text: &str) -> Result<ProteinStructure, BioError> {
    if text.contains("_atom_site.") {
        parse_mmcif(text)
    } else {
        parse_pdb(text)
    }
}

/// Euclidean distance between two 3D coordinates (Angstroms).
#[must_use]
pub fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Bondi Van der Waals radius (Angstroms) for Shrake-Rupley SASA.
#[must_use]
pub fn vdw_radius(elem: &str) -> f64 {
    match elem.to_ascii_uppercase().as_str() {
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "CL" => 1.75,
        _ => 1.70,
    }
}

/// Generate evenly spaced unit-sphere Fibonacci lattice points for Shrake-Rupley SASA.
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let y = 1.0 - (2.0 * (i as f64) + 1.0) / (n as f64);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * (i as f64);
        pts.push([theta.cos() * r, y, theta.sin() * r]);
    }
    pts
}

/// Compute per-atom Solvent Accessible Surface Area (Å²) via Rayon-parallel Shrake-Rupley
/// with spatial cell-list acceleration (`probe_radius = 1.4 Å`, `42` Fibonacci sphere points).
#[must_use]
pub fn compute_shrake_rupley_sasa(
    atoms: &[Atom3D],
    probe_radius: f64,
    n_points: usize,
) -> Vec<f64> {
    let n = atoms.len();
    if n == 0 {
        return Vec::new();
    }
    let sphere = fibonacci_sphere(n_points.max(16));
    let radii: Vec<f64> = atoms
        .iter()
        .map(|a| vdw_radius(&a.element) + probe_radius)
        .collect();
    let max_r = radii.iter().copied().fold(0.0_f64, f64::max);
    let cell_size = (2.0 * max_r).max(6.4);

    // Spatial hash grid for O(N) neighbor lookup
    let cell_of = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell_size).floor() as i32,
            (c[1] / cell_size).floor() as i32,
            (c[2] / cell_size).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(n / 4 + 1);
    for (idx, a) in atoms.iter().enumerate() {
        grid.entry(cell_of(a.coords)).or_default().push(idx);
    }

    (0..n)
        .into_par_iter()
        .map(|i| {
            let pi = atoms[i].coords;
            let ri = radii[i];
            let (cx, cy, cz) = cell_of(pi);
            let mut nbrs: Vec<(usize, f64)> = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if j == i {
                                    continue;
                                }
                                let rsum = ri + radii[j];
                                let d2 = (atoms[j].coords[0] - pi[0]).powi(2)
                                    + (atoms[j].coords[1] - pi[1]).powi(2)
                                    + (atoms[j].coords[2] - pi[2]).powi(2);
                                if d2 < rsum * rsum {
                                    nbrs.push((j, radii[j] * radii[j]));
                                }
                            }
                        }
                    }
                }
            }

            let mut exposed = 0usize;
            for &u in &sphere {
                let q = [pi[0] + ri * u[0], pi[1] + ri * u[1], pi[2] + ri * u[2]];
                let mut occluded = false;
                for &(j, rj2) in &nbrs {
                    let pj = atoms[j].coords;
                    let d2 =
                        (q[0] - pj[0]).powi(2) + (q[1] - pj[1]).powi(2) + (q[2] - pj[2]).powi(2);
                    if d2 < rj2 {
                        occluded = true;
                        break;
                    }
                }
                if !occluded {
                    exposed += 1;
                }
            }
            let frac = (exposed as f64) / (sphere.len() as f64);
            4.0 * PI * ri * ri * frac
        })
        .collect()
}

/// Kyte-Doolittle hydropathy scale.
#[must_use]
pub fn kyte_doolittle(aa: char) -> f64 {
    match aa.to_ascii_uppercase() {
        'I' => 4.5,
        'V' => 4.2,
        'L' => 3.8,
        'F' => 2.8,
        'C' => 2.5,
        'M' => 1.9,
        'A' => 1.8,
        'G' => -0.4,
        'T' => -0.7,
        'S' => -0.8,
        'W' => -0.9,
        'Y' => -1.3,
        'P' => -1.6,
        'H' => -3.2,
        'E' | 'Q' | 'D' | 'N' => -3.5,
        'K' => -3.9,
        'R' => -4.5,
        _ => 0.0,
    }
}

/// Black & Mould (1991) normalized hydrophobicity scale centered around Gly = 0.0 (used in 3D SAP,
/// Chennamsetty et al. *PNAS* 2009: positive = hydrophobic, negative = hydrophilic).
#[must_use]
pub fn black_mould_sap_hydro(aa: char) -> f64 {
    // Raw Black & Mould (0..1) shifted by -0.501 (Gly = 0.0)
    let raw = match aa.to_ascii_uppercase() {
        'F' => 1.000,
        'L' => 0.943,
        'I' => 0.943,
        'Y' => 0.880,
        'W' => 0.878,
        'V' => 0.825,
        'M' => 0.738,
        'P' => 0.711,
        'C' => 0.680,
        'A' => 0.616,
        'G' => 0.501,
        'T' => 0.450,
        'S' => 0.359,
        'K' => 0.283,
        'Q' => 0.251,
        'N' => 0.236,
        'H' => 0.165,
        'E' => 0.043,
        'D' => 0.028,
        'R' => 0.000,
        _ => 0.501,
    };
    raw - 0.501
}

/// Reference fully-exposed sidechain SASA (Å², Tien et al. 2013) for SAP normalization.
#[must_use]
pub fn max_residue_sasa(aa: char) -> f64 {
    match aa.to_ascii_uppercase() {
        'A' => 129.0,
        'R' => 274.0,
        'N' => 195.0,
        'D' => 193.0,
        'C' => 167.0,
        'E' => 223.0,
        'Q' => 225.0,
        'G' => 104.0,
        'H' => 224.0,
        'I' => 197.0,
        'L' => 201.0,
        'K' => 236.0,
        'M' => 224.0,
        'F' => 240.0,
        'P' => 159.0,
        'S' => 155.0,
        'T' => 172.0,
        'W' => 285.0,
        'Y' => 263.0,
        'V' => 174.0,
        _ => 180.0,
    }
}

/// Average residue molecular weight (Da, anhydrous residue inside polypeptide chain).
#[must_use]
pub fn residue_avg_mw(aa: char) -> f64 {
    match aa.to_ascii_uppercase() {
        'A' => 71.0788,
        'R' => 156.1875,
        'N' => 114.1038,
        'D' => 115.0886,
        'C' => 103.1388,
        'E' => 129.1155,
        'Q' => 128.1307,
        'G' => 57.0519,
        'H' => 137.1411,
        'I' => 113.1594,
        'L' => 113.1594,
        'K' => 128.1741,
        'M' => 131.1926,
        'F' => 147.1766,
        'P' => 97.1167,
        'S' => 87.0782,
        'T' => 101.1051,
        'W' => 186.2132,
        'Y' => 163.1760,
        'V' => 99.1326,
        _ => 110.0,
    }
}
