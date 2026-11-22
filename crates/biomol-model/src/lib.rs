#![forbid(unsafe_code)]

//! `biomol-model` — Interactive 3D Biomolecular Modeling, PyMOL-Style Mutagenesis Wizard
//! (Dunbrack Rotamers + Live Steric Clash Discs + Local Repacking), Local MM Energy Minimizer
//! (ISOLDE-lite Steepest-Descent), and 3D Structure Preparation Audit.

use craftbio_scene::SphereOverlay;
use craftbio_struct::{
    Atom, BioStructError, Biomolecule, ResidueClass, aa1_to_aa3, approx_atom_charge, dist3,
    finalize_biomolecule, vdw_radius,
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

/// Result of a 3D point mutation with rotamer placement and clash diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutation3DReport {
    pub chain: char,
    pub res_seq: i32,
    pub wild_type: char,
    pub mutant: char,
    pub rotamer_index: usize,
    pub total_rotamers: usize,
    pub chi1_deg: f64,
    pub clash_count: usize,
    pub new_hbond_count: usize,
    pub clash_overlays: Vec<SphereOverlay>,
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

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> [f64; 3] {
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

/// Construct idealized heavy sidechain atom coordinates in the local `(N, CA, C)` frame for `aa` at `chi1_deg`.
fn build_sidechain_atoms(
    n_pos: [f64; 3],
    ca_pos: [f64; 3],
    c_pos: [f64; 3],
    aa: char,
    chi1_deg: f64,
) -> Vec<(&'static str, &'static str, [f64; 3])> {
    if aa == 'G' {
        return Vec::new();
    }
    // Build orthonormal local frame at CA with standard L-amino-acid CB direction
    let v_n = norm3(sub3(n_pos, ca_pos));
    let v_c = norm3(sub3(c_pos, ca_pos));
    let bis = norm3(add3(v_n, v_c));
    let perp = norm3(cross(v_n, v_c));
    // Standard tetrahedral L-chirality CB vector (~109.5 deg from both N and C)
    let cb_dir = norm3(add3(scale3(bis, -0.577), scale3(perp, -0.816)));
    let cb = add3(ca_pos, scale3(cb_dir, 1.53));
    if aa == 'A' {
        return vec![("CB", "C", cb)];
    }

    // Orthonormal basis around CA->CB bond for chi1 rotation
    let z_axis = cb_dir;
    let x_axis = norm3(sub3(
        v_n,
        scale3(
            z_axis,
            v_n[0] * z_axis[0] + v_n[1] * z_axis[1] + v_n[2] * z_axis[2],
        ),
    ));
    let y_axis = norm3(cross(z_axis, x_axis));

    let place = |origin: [f64; 3], len: f64, theta_deg: f64, phi_deg: f64| -> [f64; 3] {
        let th = theta_deg.to_radians();
        let ph = phi_deg.to_radians();
        let dir = add3(
            scale3(z_axis, th.cos()),
            scale3(
                add3(scale3(x_axis, ph.cos()), scale3(y_axis, ph.sin())),
                th.sin(),
            ),
        );
        add3(origin, scale3(norm3(dir), len))
    };

    let mut out = vec![("CB", "C", cb)];
    let c1 = chi1_deg;
    match aa {
        'S' => out.push(("OG", "O", place(cb, 1.43, 70.5, c1))),
        'C' => out.push(("SG", "S", place(cb, 1.81, 70.5, c1))),
        'T' => {
            out.push(("OG1", "O", place(cb, 1.43, 70.5, c1)));
            out.push(("CG2", "C", place(cb, 1.52, 70.5, c1 + 120.0)));
        }
        'V' => {
            out.push(("CG1", "C", place(cb, 1.52, 70.5, c1)));
            out.push(("CG2", "C", place(cb, 1.52, 70.5, c1 + 120.0)));
        }
        'I' => {
            let cg1 = place(cb, 1.52, 70.5, c1);
            out.push(("CG1", "C", cg1));
            out.push(("CG2", "C", place(cb, 1.52, 70.5, c1 - 120.0)));
            out.push(("CD1", "C", place(cg1, 1.52, 35.0, c1 + 180.0)));
        }
        'L' => {
            let cg = place(cb, 1.53, 70.5, c1);
            out.push(("CG", "C", cg));
            out.push(("CD1", "C", place(cg, 1.52, 42.0, c1 + 60.0)));
            out.push(("CD2", "C", place(cg, 1.52, 42.0, c1 - 60.0)));
        }
        'D' => {
            let cg = place(cb, 1.52, 70.5, c1);
            out.push(("CG", "C", cg));
            out.push(("OD1", "O", place(cg, 1.25, 45.0, c1 + 70.0)));
            out.push(("OD2", "O", place(cg, 1.25, 45.0, c1 - 70.0)));
        }
        'N' => {
            let cg = place(cb, 1.52, 70.5, c1);
            out.push(("CG", "C", cg));
            out.push(("OD1", "O", place(cg, 1.24, 45.0, c1 + 70.0)));
            out.push(("ND2", "N", place(cg, 1.33, 45.0, c1 - 70.0)));
        }
        'E' => {
            let cg = place(cb, 1.52, 70.5, c1);
            let cd = place(cg, 1.52, 35.0, c1 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("OE1", "O", place(cd, 1.25, 45.0, c1 + 70.0)));
            out.push(("OE2", "O", place(cd, 1.25, 45.0, c1 - 70.0)));
        }
        'Q' => {
            let cg = place(cb, 1.52, 70.5, c1);
            let cd = place(cg, 1.52, 35.0, c1 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("OE1", "O", place(cd, 1.24, 45.0, c1 + 70.0)));
            out.push(("NE2", "N", place(cd, 1.33, 45.0, c1 - 70.0)));
        }
        'M' => {
            let cg = place(cb, 1.52, 70.5, c1);
            let sd = place(cg, 1.81, 35.0, c1 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("SD", "S", sd));
            out.push(("CE", "C", place(sd, 1.79, 50.0, c1)));
        }
        'K' => {
            let cg = place(cb, 1.52, 70.5, c1);
            let cd = place(cg, 1.52, 35.0, c1 + 180.0);
            let ce = place(cd, 1.52, 35.0, c1);
            let nz = place(ce, 1.49, 35.0, c1 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("CE", "C", ce));
            out.push(("NZ", "N", nz));
        }
        'R' => {
            let cg = place(cb, 1.52, 70.5, c1);
            let cd = place(cg, 1.52, 35.0, c1 + 180.0);
            let ne = place(cd, 1.46, 35.0, c1);
            let cz = place(ne, 1.33, 35.0, c1 + 180.0);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
            out.push(("NE", "N", ne));
            out.push(("CZ", "C", cz));
            out.push(("NH1", "N", place(cz, 1.33, 45.0, c1 + 70.0)));
            out.push(("NH2", "N", place(cz, 1.33, 45.0, c1 - 70.0)));
        }
        'H' => {
            let cg = place(cb, 1.50, 70.5, c1);
            let nd1 = place(cg, 1.38, 48.0, c1 + 75.0);
            let cd2 = place(cg, 1.36, 48.0, c1 - 75.0);
            let ce1 = place(nd1, 1.33, 25.0, c1 - 20.0);
            let ne2 = place(cd2, 1.37, 25.0, c1 + 20.0);
            out.push(("CG", "C", cg));
            out.push(("ND1", "N", nd1));
            out.push(("CD2", "C", cd2));
            out.push(("CE1", "C", ce1));
            out.push(("NE2", "N", ne2));
        }
        'F' | 'Y' => {
            let cg = place(cb, 1.51, 70.5, c1);
            let cd1 = place(cg, 1.39, 55.0, c1 + 90.0);
            let cd2 = place(cg, 1.39, 55.0, c1 - 90.0);
            let ce1 = place(cd1, 1.39, 12.0, c1);
            let ce2 = place(cd2, 1.39, 12.0, c1);
            let cz = place(cg, 2.78, 12.0, c1);
            out.push(("CG", "C", cg));
            out.push(("CD1", "C", cd1));
            out.push(("CD2", "C", cd2));
            out.push(("CE1", "C", ce1));
            out.push(("CE2", "C", ce2));
            out.push(("CZ", "C", cz));
            if aa == 'Y' {
                out.push(("OH", "O", place(cz, 1.37, 12.0, c1)));
            }
        }
        'W' => {
            let cg = place(cb, 1.50, 70.5, c1);
            let cd1 = place(cg, 1.37, 50.0, c1 + 85.0);
            let cd2 = place(cg, 1.43, 50.0, c1 - 85.0);
            let ne1 = place(cd1, 1.38, 18.0, c1);
            let ce2 = place(cd2, 1.41, 18.0, c1 + 20.0);
            let ce3 = place(cd2, 1.40, 45.0, c1 - 110.0);
            let cz2 = place(ce2, 1.40, 15.0, c1 + 35.0);
            let cz3 = place(ce3, 1.39, 15.0, c1 - 35.0);
            let ch2 = place(cz2, 1.40, 20.0, c1 - 20.0);
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
            let cg = place(cb, 1.50, 65.0, 25.0);
            let cd = add3(scale3(add3(cg, n_pos), 0.5), [0.0, 0.0, 0.0]);
            out.push(("CG", "C", cg));
            out.push(("CD", "C", cd));
        }
        _ => {}
    }
    out
}

/// Mutate residue `chain:res_seq` in place to `target_aa` using Dunbrack rotamer `rotamer_idx`,
/// returning live steric clash overlays and H-bond diagnostics.
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

    let rots = dunbrack_chi1_rotamers(aa);
    let r_idx = rotamer_idx % rots.len();
    let chi1 = rots[r_idx];
    let new_sc = build_sidechain_atoms(n_pos, ca_pos, c_pos, aa, chi1);
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

    // Evaluate steric clashes & new polar H-bonds on placed sidechain
    let mut clash_count = 0usize;
    let mut hbond_count = 0usize;
    let mut clash_overlays = Vec::new();
    for &si in &inserted_indices {
        let sa = &new_atoms[si];
        let rsa = vdw_radius(&sa.element);
        for (oi, oa) in new_atoms.iter().enumerate() {
            if oi == si || (oa.chain == chain && oa.res_seq == res_seq) {
                continue;
            }
            let d = dist3(sa.coords, oa.coords);
            let clash_lim = (rsa + vdw_radius(&oa.element)) * 0.68;
            if d < clash_lim {
                clash_count += 1;
                if clash_overlays.len() < 12 {
                    clash_overlays.push(SphereOverlay {
                        label: format!(
                            "Clash {}{}:{}..{}{}",
                            sa.res_code, sa.res_seq, sa.name, oa.res_code, oa.res_seq
                        ),
                        category: "Clash".to_string(),
                        coords: [
                            0.5 * (sa.coords[0] + oa.coords[0]),
                            0.5 * (sa.coords[1] + oa.coords[1]),
                            0.5 * (sa.coords[2] + oa.coords[2]),
                        ],
                        radius_a: 0.65,
                        rgba: [0.96, 0.18, 0.22, 0.78],
                    });
                }
            } else if d <= 3.45
                && matches!(sa.element.as_str(), "N" | "O")
                && matches!(oa.element.as_str(), "N" | "O")
            {
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
        chi1_deg: chi1,
        clash_count,
        new_hbond_count: hbond_count,
        clash_overlays,
    })
}

/// Automatically evaluate all Dunbrack rotamers and apply the clash-minimum rotamer.
pub fn mutate_and_repack_best_rotamer(
    mol: &mut Biomolecule,
    chain: char,
    res_seq: i32,
    target_aa: char,
) -> Result<Mutation3DReport, ModelError> {
    let rots = dunbrack_chi1_rotamers(target_aa);
    let mut best_idx = 0usize;
    let mut best_score = i64::MAX;
    for r in 0..rots.len() {
        let mut probe = mol.clone();
        let rep = mutate_residue_3d(&mut probe, chain, res_seq, target_aa, r)?;
        let sc = (rep.clash_count as i64) * 10 - (rep.new_hbond_count as i64) * 3;
        if sc < best_score {
            best_score = sc;
            best_idx = r;
        }
    }
    mutate_residue_3d(mol, chain, res_seq, target_aa, best_idx)
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
}
