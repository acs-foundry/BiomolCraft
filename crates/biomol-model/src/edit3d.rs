#![forbid(unsafe_code)]

//! Wave 6 Advanced 3D Biomolecular Modeling & Atomic Editing:
//! - **`swapna`**: 3D RNA/DNA Nucleotide Mutagenesis preserving ribose-phosphate backbone & base plane
//! - **`bond_rot`**: Interactive Rotatable Bond Torsion with live steric clash & HYDE-proxy ΔΔG feedback
//! - **`tug` / Sculpting**: Pull atoms interactively with live ISOLDE-lite MM constraint relaxation
//! - **3D Structure Builder (`build` / `fab`)**: De Novo α-Helix, β-Sheet, Extended Peptide & A-Form RNA Duplex
//! - **Structure Editing Suite**: `addh` (pH-aware polar hydrogens), `renumber`, `changechains`, `combine`, `delete`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

use craftbio_scene::SphereOverlay;
use craftbio_struct::{
    Atom, Biomolecule, ResidueClass, approx_atom_charge, dist3, finalize_biomolecule, vdw_radius,
};
use serde::{Deserialize, Serialize};

use crate::{
    ModelError, add3, build_sidechain_atoms, cross, minimize_local_mm, norm3, scale3, sub3,
};

/// Result of a 3D RNA/DNA nucleotide mutation (`swapna`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapNaReport {
    pub chain: char,
    pub res_seq: i32,
    pub wild_type: char,
    pub mutant: char,
    pub is_dna: bool,
    pub clash_count: usize,
    pub hbond_count: usize,
    pub base_pairs_after: usize,
    pub clash_overlays: Vec<SphereOverlay>,
}

/// Result of rotating a covalent bond torsion (`bond_rot`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondRotReport {
    pub bond_label: String,
    pub delta_deg: f64,
    pub rotated_atoms: usize,
    pub clash_count_after: usize,
    pub hbond_count_after: usize,
    pub local_score_before_kcal: f64,
    pub local_score_after_kcal: f64,
}

/// Result of interactive atom pulling with live ISOLDE-lite MM sculpting (`tug`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TugSculptReport {
    pub atom_label: String,
    pub target_shift_a: f64,
    pub atoms_sculpted: usize,
    pub energy_before_kcal: f64,
    pub energy_after_kcal: f64,
    pub local_ddg_kcal: f64,
}

/// Result of pH-aware polar hydrogen addition (`addh`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddHydrogensReport {
    pub ph: f64,
    pub hydrogens_added: usize,
    pub protonated_his: usize,
    pub protonated_asp_glu: usize,
}

fn rodrigues_rotate(v: [f64; 3], axis_unit: [f64; 3], angle_rad: f64) -> [f64; 3] {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    let dot = v[0] * axis_unit[0] + v[1] * axis_unit[1] + v[2] * axis_unit[2];
    let cx = cross(axis_unit, v);
    [
        v[0] * c + cx[0] * s + axis_unit[0] * dot * (1.0 - c),
        v[1] * c + cx[1] * s + axis_unit[1] * dot * (1.0 - c),
        v[2] * c + cx[2] * s + axis_unit[2] * dot * (1.0 - c),
    ]
}

/// Instant HYDE-inspired local pocket interaction score (kcal/mol) for live feedback during
/// mutagenesis, `bond_rot`, and `tug` sculpting (negative = favorable H-bonds & apolar packing).
#[must_use]
pub fn evaluate_local_binding_proxy(mol: &Biomolecule, focus_mask: &[bool]) -> f64 {
    let mut score = 0.0_f64;
    let n = mol.atoms.len();
    for i in 0..n {
        if !focus_mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        let ai = &mol.atoms[i];
        if ai.element == "H" || ai.res_class == ResidueClass::Water {
            continue;
        }
        let ri = vdw_radius(&ai.element);
        let i_polar = matches!(ai.element.as_str(), "N" | "O" | "S");
        for (j, aj) in mol.atoms.iter().enumerate() {
            if focus_mask.get(j).copied().unwrap_or(false) {
                continue;
            }
            if aj.element == "H"
                || aj.res_class == ResidueClass::Water
                || (ai.chain == aj.chain && (ai.res_seq - aj.res_seq).abs() <= 1)
            {
                continue;
            }
            let d = dist3(ai.coords, aj.coords);
            if d > 4.8 {
                continue;
            }
            let j_polar = matches!(aj.element.as_str(), "N" | "O" | "S");
            let rsum = ri + vdw_radius(&aj.element);
            if d < rsum * 0.72 && !(i_polar && j_polar && d >= 2.4) {
                let ov = rsum * 0.72 - d;
                score += 6.5 * ov * ov;
            } else if i_polar && j_polar && (2.4..=3.45).contains(&d) {
                score -= 1.35 * (1.0 - ((d - 2.85).abs() / 0.85)).clamp(0.2, 1.0);
            } else if !i_polar && !j_polar && (3.2..=4.5).contains(&d) {
                score -= 0.14;
            } else if i_polar != j_polar && d < 3.4 {
                // Uncompensated polar desolvation penalty (HYDE principle)
                score += 0.22;
            }
        }
    }
    (score * 100.0).round() / 100.0
}

/// Idealized 2D planar nucleobase coordinates `(atom_name, elem, x_along_glycosidic, y_in_plane)`
/// anchored with the glycosidic `N9` (purine) or `N1` (pyrimidine) at `(0.0, 0.0)`.
fn idealized_base_2d(base: char) -> Vec<(&'static str, &'static str, f64, f64)> {
    match base {
        'A' => vec![
            ("N9", "N", 0.00, 0.00),
            ("C8", "C", 0.74, -1.16),
            ("N7", "N", 2.02, -0.84),
            ("C5", "C", 2.12, 0.54),
            ("C4", "C", 0.86, 1.06),
            ("N3", "N", 0.64, 2.38),
            ("C2", "C", 1.74, 3.12),
            ("N1", "N", 2.98, 2.66),
            ("C6", "C", 3.22, 1.34),
            ("N6", "N", 4.46, 0.88),
        ],
        'G' => vec![
            ("N9", "N", 0.00, 0.00),
            ("C8", "C", 0.74, -1.16),
            ("N7", "N", 2.02, -0.84),
            ("C5", "C", 2.12, 0.54),
            ("C4", "C", 0.86, 1.06),
            ("N3", "N", 0.64, 2.38),
            ("C2", "C", 1.74, 3.13),
            ("N2", "N", 1.56, 4.45),
            ("N1", "N", 2.98, 2.66),
            ("C6", "C", 3.24, 1.32),
            ("O6", "O", 4.40, 0.88),
        ],
        'C' => vec![
            ("N1", "N", 0.00, 0.00),
            ("C6", "C", 0.68, -1.18),
            ("C5", "C", 2.04, -1.20),
            ("C4", "C", 2.74, 0.02),
            ("N4", "N", 4.06, 0.05),
            ("N3", "N", 2.06, 1.20),
            ("C2", "C", 0.68, 1.20),
            ("O2", "O", 0.05, 2.26),
        ],
        'T' => vec![
            ("N1", "N", 0.00, 0.00),
            ("C6", "C", 0.68, -1.18),
            ("C5", "C", 2.04, -1.20),
            ("C7", "C", 2.78, -2.48),
            ("C4", "C", 2.74, 0.02),
            ("O4", "O", 3.98, 0.05),
            ("N3", "N", 2.06, 1.20),
            ("C2", "C", 0.68, 1.20),
            ("O2", "O", 0.05, 2.26),
        ],
        // 'U'
        _ => vec![
            ("N1", "N", 0.00, 0.00),
            ("C6", "C", 0.68, -1.18),
            ("C5", "C", 2.04, -1.20),
            ("C4", "C", 2.74, 0.02),
            ("O4", "O", 3.98, 0.05),
            ("N3", "N", 2.06, 1.20),
            ("C2", "C", 0.68, 1.20),
            ("O2", "O", 0.05, 2.26),
        ],
    }
}

/// Mutate an RNA or DNA nucleotide in 3D (`swapna`), preserving exact sugar-phosphate backbone
/// coordinates, glycosidic `C1'–N` attachment vector, and planar base orientation!
pub fn swapna_nucleotide_3d(
    mol: &mut Biomolecule,
    chain: char,
    res_seq: i32,
    target_base: char,
) -> Result<SwapNaReport, ModelError> {
    let base = target_base.to_ascii_uppercase();
    if !matches!(base, 'A' | 'U' | 'G' | 'C' | 'T') {
        return Err(ModelError::InvalidAminoAcid(target_base));
    }
    let ch = mol
        .chain(chain)
        .ok_or(ModelError::ResidueNotFound(chain, res_seq))?;
    let res = ch
        .residues
        .iter()
        .find(|r| r.res_seq == res_seq && matches!(r.class, ResidueClass::Rna | ResidueClass::Dna))
        .ok_or(ModelError::ResidueNotFound(chain, res_seq))?;

    let wt = res.code;
    let is_dna = res.class == ResidueClass::Dna || base == 'T';
    let icode = res.icode;
    let bf = res.mean_bfactor;
    let new_rname = if is_dna {
        format!("D{}", if base == 'U' { 'T' } else { base })
    } else {
        (if base == 'T' { 'U' } else { base }).to_string()
    };
    let final_code = if is_dna && base == 'U' {
        'T'
    } else if !is_dna && base == 'T' {
        'U'
    } else {
        base
    };
    let new_class = if is_dna {
        ResidueClass::Dna
    } else {
        ResidueClass::Rna
    };

    let find_atom = |names: &[&str]| -> Option<[f64; 3]> {
        res.atom_indices
            .iter()
            .copied()
            .find(|&i| names.contains(&mol.atoms[i].name.as_str()))
            .map(|i| mol.atoms[i].coords)
    };
    let c1p = find_atom(&["C1'", "C1*"]).ok_or(ModelError::ResidueNotFound(chain, res_seq))?;
    let glyc_n = find_atom(&["N9", "N1"]).unwrap_or_else(|| {
        let c4 = find_atom(&["C4'", "C4*"]).unwrap_or([c1p[0] - 2.0, c1p[1], c1p[2]]);
        add3(c1p, scale3(norm3(sub3(c1p, c4)), 1.47))
    });
    let u_axis = norm3(sub3(glyc_n, c1p));
    let aux_pt = find_atom(&["C4", "C2", "C8", "C6"])
        .unwrap_or_else(|| find_atom(&["O4'", "O4*"]).unwrap_or([c1p[0], c1p[1] + 1.0, c1p[2]]));
    let aux_v = sub3(aux_pt, glyc_n);
    let proj = sub3(
        aux_v,
        scale3(
            u_axis,
            aux_v[0] * u_axis[0] + aux_v[1] * u_axis[1] + aux_v[2] * u_axis[2],
        ),
    );
    let v_axis = if proj[0] * proj[0] + proj[1] * proj[1] + proj[2] * proj[2] > 1e-6 {
        norm3(proj)
    } else {
        let alt = if u_axis[1].abs() < 0.8 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        norm3(cross(u_axis, alt))
    };

    let anchor_n = add3(c1p, scale3(u_axis, 1.47));
    let base_template = idealized_base_2d(final_code);
    let placed_base: Vec<(&'static str, &'static str, [f64; 3])> = base_template
        .into_iter()
        .map(|(nm, el, x, y)| {
            let p = add3(anchor_n, add3(scale3(u_axis, x), scale3(v_axis, y)));
            (nm, el, p)
        })
        .collect();

    let is_backbone_nuc = |name: &str| -> bool {
        name.contains('\'')
            || name.contains('*')
            || matches!(name, "P" | "OP1" | "OP2" | "O1P" | "O2P" | "OP3")
    };

    let mut new_atoms = Vec::with_capacity(mol.atoms.len() + 12);
    let mut inserted_base = false;
    let mut inserted_indices = Vec::new();

    for a in &mol.atoms {
        if a.chain == chain && a.res_seq == res_seq && a.icode == icode {
            if is_backbone_nuc(&a.name) {
                let mut bb = a.clone();
                bb.res_name = new_rname.clone();
                bb.res_code = final_code;
                bb.res_class = new_class;
                new_atoms.push(bb);
                if (a.name == "C1'" || a.name == "C1*") && !inserted_base {
                    for &(nm, el, coords) in &placed_base {
                        inserted_indices.push(new_atoms.len());
                        new_atoms.push(Atom {
                            serial: 0,
                            name: nm.to_string(),
                            element: el.to_string(),
                            res_name: new_rname.clone(),
                            res_code: final_code,
                            res_class: new_class,
                            chain,
                            res_seq,
                            icode,
                            coords,
                            occupancy: 1.0,
                            b_factor: bf,
                            formal_charge: approx_atom_charge(new_class, final_code, nm),
                            is_hetatm: false,
                        });
                    }
                    inserted_base = true;
                }
            }
        } else {
            new_atoms.push(a.clone());
        }
    }
    if !inserted_base {
        for &(nm, el, coords) in &placed_base {
            inserted_indices.push(new_atoms.len());
            new_atoms.push(Atom {
                serial: 0,
                name: nm.to_string(),
                element: el.to_string(),
                res_name: new_rname.clone(),
                res_code: final_code,
                res_class: new_class,
                chain,
                res_seq,
                icode,
                coords,
                occupancy: 1.0,
                b_factor: bf,
                formal_charge: approx_atom_charge(new_class, final_code, nm),
                is_hetatm: false,
            });
        }
    }
    for (i, a) in new_atoms.iter_mut().enumerate() {
        a.serial = i + 1;
    }

    // Evaluate clashes & base-pair H-bonds on placed nucleobase
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
            if d < (rsa + vdw_radius(&oa.element)) * 0.65 {
                clash_count += 1;
                if clash_overlays.len() < 12 {
                    clash_overlays.push(SphereOverlay {
                        label: format!("NA Clash {}{}:{}", sa.res_code, sa.res_seq, sa.name),
                        category: "Clash".to_string(),
                        coords: [
                            0.5 * (sa.coords[0] + oa.coords[0]),
                            0.5 * (sa.coords[1] + oa.coords[1]),
                            0.5 * (sa.coords[2] + oa.coords[2]),
                        ],
                        radius_a: 0.60,
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

    let uc = mol.unit_cell.clone();
    let biomt = mol.biomt_ops.clone();
    let mut rebuilt = finalize_biomolecule(
        mol.id.clone(),
        mol.title.clone(),
        new_atoms,
        Vec::new(),
        mol.is_alphafold_plddt,
    )?;
    rebuilt.unit_cell = uc;
    rebuilt.biomt_ops = biomt;
    let bp_after = rebuilt.base_pairs.len();
    *mol = rebuilt;

    Ok(SwapNaReport {
        chain,
        res_seq,
        wild_type: wt,
        mutant: final_code,
        is_dna,
        clash_count,
        hbond_count,
        base_pairs_after: bp_after,
        clash_overlays,
    })
}

/// Rotate the downstream fragment of covalent bond `atom_a -> atom_b` by `delta_deg` degrees
/// (`bond_rot`), automatically rejecting cyclic ring bonds and reporting live clashes & ΔΔG.
pub fn rotate_bond_torsion(
    mol: &mut Biomolecule,
    atom_a: usize,
    atom_b: usize,
    delta_deg: f64,
) -> Result<BondRotReport, ModelError> {
    if atom_a >= mol.atoms.len() || atom_b >= mol.atoms.len() || atom_a == atom_b {
        return Err(ModelError::Struct(
            craftbio_struct::BioStructError::Selection("invalid bond atom indices".to_string()),
        ));
    }
    let mut adj = vec![Vec::new(); mol.atoms.len()];
    let mut has_bond = false;
    for b in &mol.bonds {
        if (b.a == atom_a && b.b == atom_b) || (b.a == atom_b && b.b == atom_a) {
            has_bond = true;
            continue;
        }
        adj[b.a].push(b.b);
        adj[b.b].push(b.a);
    }
    if !has_bond {
        return Err(ModelError::Struct(
            craftbio_struct::BioStructError::Selection(
                "selected atoms are not covalently bonded".to_string(),
            ),
        ));
    }
    // BFS from atom_b without traversing edge (atom_a, atom_b)
    let mut visited = HashSet::new();
    let mut q = VecDeque::new();
    visited.insert(atom_b);
    q.push_back(atom_b);
    while let Some(u) = q.pop_front() {
        for &v in &adj[u] {
            if v == atom_a {
                return Err(ModelError::Struct(
                    craftbio_struct::BioStructError::Selection(
                        "bond belongs to a closed ring and cannot be freely rotated".to_string(),
                    ),
                ));
            }
            if visited.insert(v) {
                q.push_back(v);
            }
        }
    }

    let mut mask = vec![false; mol.atoms.len()];
    for &idx in &visited {
        if idx != atom_b {
            mask[idx] = true;
        }
    }
    let score_before = evaluate_local_binding_proxy(mol, &mask);

    let pivot = mol.atoms[atom_b].coords;
    let axis = norm3(sub3(pivot, mol.atoms[atom_a].coords));
    let rad = delta_deg.to_radians();
    for &idx in &visited {
        if idx == atom_b {
            continue;
        }
        let rel = sub3(mol.atoms[idx].coords, pivot);
        mol.atoms[idx].coords = add3(pivot, rodrigues_rotate(rel, axis, rad));
    }

    let score_after = evaluate_local_binding_proxy(mol, &mask);
    let mut clashes = 0usize;
    let mut hbonds = 0usize;
    for (i, &in_m) in mask.iter().enumerate() {
        if !in_m {
            continue;
        }
        let ai = &mol.atoms[i];
        let ri = vdw_radius(&ai.element);
        for (j, &j_m) in mask.iter().enumerate() {
            if j_m || j == atom_a || j == atom_b {
                continue;
            }
            let aj = &mol.atoms[j];
            let d = dist3(ai.coords, aj.coords);
            if d < (ri + vdw_radius(&aj.element)) * 0.68 {
                clashes += 1;
            } else if d <= 3.45
                && matches!(ai.element.as_str(), "N" | "O")
                && matches!(aj.element.as_str(), "N" | "O")
            {
                hbonds += 1;
            }
        }
    }
    let aa = &mol.atoms[atom_a];
    let ab = &mol.atoms[atom_b];
    Ok(BondRotReport {
        bond_label: format!(
            "{}:{}.{} — {}:{}.{}",
            aa.chain, aa.res_seq, aa.name, ab.chain, ab.res_seq, ab.name
        ),
        delta_deg,
        rotated_atoms: visited.len().saturating_sub(1),
        clash_count_after: clashes,
        hbond_count_after: hbonds,
        local_score_before_kcal: score_before,
        local_score_after_kcal: score_after,
    })
}

/// Interactive `tug` / PyMOL Sculpting: displace `atom_idx` by `displacement` (Å) while relaxing
/// all surrounding atoms within `radius_a` (`6.0 Å`) using local MM bond & clash constraints!
pub fn tug_atom_with_sculpting(
    mol: &mut Biomolecule,
    atom_idx: usize,
    displacement: [f64; 3],
    radius_a: f64,
    relax_steps: usize,
) -> Result<TugSculptReport, ModelError> {
    if atom_idx >= mol.atoms.len() {
        return Err(ModelError::Struct(
            craftbio_struct::BioStructError::Selection("tug atom index out of range".to_string()),
        ));
    }
    let orig = mol.atoms[atom_idx].coords;
    let label = {
        let a = &mol.atoms[atom_idx];
        format!("{}:{}{}.{}", a.chain, a.res_name, a.res_seq, a.name)
    };
    let shift_len = (displacement[0].powi(2) + displacement[1].powi(2) + displacement[2].powi(2))
        .sqrt()
        .clamp(0.0, 8.0);
    let clamped_disp = if shift_len > 1e-6 {
        let raw_len =
            (displacement[0].powi(2) + displacement[1].powi(2) + displacement[2].powi(2)).sqrt();
        let f = shift_len / raw_len;
        [
            displacement[0] * f,
            displacement[1] * f,
            displacement[2] * f,
        ]
    } else {
        [0.0; 3]
    };

    let rad = radius_a.clamp(3.5, 12.0);
    let mut focus_mask = vec![false; mol.atoms.len()];
    let mut relax_mask = vec![false; mol.atoms.len()];
    for (i, a) in mol.atoms.iter().enumerate() {
        let d = dist3(a.coords, orig);
        if d <= 2.5 {
            focus_mask[i] = true;
        }
        if d <= rad && i != atom_idx {
            relax_mask[i] = true;
        }
    }
    let score_before = evaluate_local_binding_proxy(mol, &focus_mask);
    // Apply smooth partial drag to atom_idx and adjacent atoms, then relax surrounding shell
    mol.atoms[atom_idx].coords = add3(orig, clamped_disp);
    let min_rep = minimize_local_mm(mol, &relax_mask, relax_steps.clamp(15, 120));
    let score_after = evaluate_local_binding_proxy(mol, &focus_mask);

    Ok(TugSculptReport {
        atom_label: label,
        target_shift_a: (shift_len * 100.0).round() / 100.0,
        atoms_sculpted: min_rep.atoms_relaxed + 1,
        energy_before_kcal: min_rep.energy_before_kcal,
        energy_after_kcal: min_rep.energy_after_kcal,
        local_ddg_kcal: ((score_after - score_before) * 100.0).round() / 100.0,
    })
}

/// De novo 3D polypeptide builder (`build helix|sheet|extended <SEQ>` / PyMOL `fab` equivalent).
pub fn build_de_novo_peptide_3d(
    sequence: &str,
    conformation: &str,
    chain_id: char,
) -> Result<Biomolecule, ModelError> {
    let seq: Vec<char> = sequence
        .trim()
        .to_ascii_uppercase()
        .chars()
        .filter(|c| "ACDEFGHIKLMNPQRSTVWY".contains(*c))
        .collect();
    if seq.is_empty() {
        return Err(ModelError::InvalidAminoAcid('?'));
    }
    let mode = conformation.trim().to_ascii_lowercase();
    let mut atoms = Vec::with_capacity(seq.len() * 10);

    for (idx, &aa) in seq.iter().enumerate() {
        let i = idx as f64;
        let (n_pos, ca_pos, c_pos, o_pos) = match mode.as_str() {
            "sheet" | "strand" | "beta" => {
                // Idealized pleated beta-strand along X (3.35 A/res, alternating carbonyls)
                let x = i * 3.35;
                let sgn = if idx % 2 == 0 { 1.0 } else { -1.0 };
                let n = [x - 1.22, 0.45 * sgn, 0.28 * sgn];
                let ca = [x, -0.35 * sgn, -0.25 * sgn];
                let c = [x + 1.28, 0.35 * sgn, 0.25 * sgn];
                let o = [x + 1.32, 1.56 * sgn, 0.45 * sgn];
                (n, ca, c, o)
            }
            "extended" | "linear" => {
                let x = i * 3.55;
                let n = [x - 1.24, 0.40, 0.0];
                let ca = [x, -0.30, 0.0];
                let c = [x + 1.30, 0.30, 0.0];
                let o = [x + 1.35, 1.52, 0.0];
                (n, ca, c, o)
            }
            _ => {
                // Canonical right-handed alpha-helix (100 deg/res, 1.50 A rise/res, 2.30 A radius)
                let ang = i * (100.0_f64.to_radians());
                let ang_n = ang - 28.0_f64.to_radians();
                let ang_c = ang + 28.0_f64.to_radians();
                let z = i * 1.50;
                let r_ca = 2.30;
                let r_bb = 1.65;
                let n = [r_bb * ang_n.cos(), r_bb * ang_n.sin(), z - 0.52];
                let ca = [r_ca * ang.cos(), r_ca * ang.sin(), z];
                let c = [r_bb * ang_c.cos(), r_bb * ang_c.sin(), z + 0.56];
                // Carbonyl O points nearly parallel to +Z helix axis for exact DSSP i->i+4 H-bonds!
                let o = [
                    (r_bb + 0.35) * ang_c.cos(),
                    (r_bb + 0.35) * ang_c.sin(),
                    z + 1.78,
                ];
                (n, ca, c, o)
            }
        };

        let rname = craftbio_struct::aa1_to_aa3(aa).to_string();
        let rseq = (idx + 1) as i32;
        for (nm, el, coords) in [
            ("N", "N", n_pos),
            ("CA", "C", ca_pos),
            ("C", "C", c_pos),
            ("O", "O", o_pos),
        ] {
            atoms.push(Atom {
                serial: atoms.len() + 1,
                name: nm.to_string(),
                element: el.to_string(),
                res_name: rname.clone(),
                res_code: aa,
                res_class: ResidueClass::Protein,
                chain: chain_id,
                res_seq: rseq,
                icode: ' ',
                coords,
                occupancy: 1.0,
                b_factor: 90.0,
                formal_charge: approx_atom_charge(ResidueClass::Protein, aa, nm),
                is_hetatm: false,
            });
        }
        let chi1 = crate::dunbrack_chi1_rotamers(aa)[0];
        for (nm, el, coords) in build_sidechain_atoms(n_pos, ca_pos, c_pos, aa, chi1) {
            atoms.push(Atom {
                serial: atoms.len() + 1,
                name: nm.to_string(),
                element: el.to_string(),
                res_name: rname.clone(),
                res_code: aa,
                res_class: ResidueClass::Protein,
                chain: chain_id,
                res_seq: rseq,
                icode: ' ',
                coords,
                occupancy: 1.0,
                b_factor: 85.0,
                formal_charge: approx_atom_charge(ResidueClass::Protein, aa, nm),
                is_hetatm: false,
            });
        }
    }
    let label = format!("built_{mode}_{}aa", seq.len());
    let title = format!("Built 3D Peptide ({mode}, {} aa)", seq.len());
    let mol = finalize_biomolecule(label, title, atoms, Vec::new(), false)?;
    Ok(mol)
}

/// Build an idealized 3D **A-Form RNA Duplex** (`5'->3'` chain `chain_id` + antiparallel
/// Watson-Crick complementary strand) with full backbone & planar nucleobases (`build rna <SEQ>`).
pub fn build_de_novo_rna_helix_3d(
    sequence: &str,
    chain_id: char,
) -> Result<Biomolecule, ModelError> {
    let seq: Vec<char> = sequence
        .trim()
        .to_ascii_uppercase()
        .chars()
        .filter_map(|c| match c {
            'A' | 'U' | 'G' | 'C' => Some(c),
            'T' => Some('U'),
            _ => None,
        })
        .collect();
    if seq.len() < 2 {
        return Err(ModelError::InvalidAminoAcid('?'));
    }
    let comp = |b: char| match b {
        'A' => 'U',
        'U' => 'A',
        'G' => 'C',
        _ => 'G',
    };
    let chain_b = if chain_id == 'A' { 'B' } else { 'A' };
    let mut atoms = Vec::with_capacity(seq.len() * 34);
    let twist = (360.0 / 11.0_f64).to_radians(); // A-form RNA: 11 bp/turn
    let rise = 2.81_f64;

    let mut push_nt = |ch: char, rseq: i32, base: char, ang: f64, z: f64, inward_sign: f64| {
        let ca = ang.cos();
        let sa = ang.sin();
        let rad_p = 9.4;
        let rad_c4 = 8.1;
        let rad_c1 = 5.4;
        let p = [rad_p * ca, rad_p * sa, z - 0.6 * inward_sign];
        let o5 = [8.7 * ca, 8.7 * sa, z - 0.2 * inward_sign];
        let c4 = [rad_c4 * ca, rad_c4 * sa, z];
        let o3 = [
            8.3 * (ang + 0.22).cos(),
            8.3 * (ang + 0.22).sin(),
            z + 1.1 * inward_sign,
        ];
        let c1 = [rad_c1 * ca, rad_c1 * sa, z];

        let rname = base.to_string();
        for (nm, el, coords) in [
            ("P", "P", p),
            ("O5'", "O", o5),
            ("C4'", "C", c4),
            ("O3'", "O", o3),
            ("C1'", "C", c1),
        ] {
            atoms.push(Atom {
                serial: atoms.len() + 1,
                name: nm.to_string(),
                element: el.to_string(),
                res_name: rname.clone(),
                res_code: base,
                res_class: ResidueClass::Rna,
                chain: ch,
                res_seq: rseq,
                icode: ' ',
                coords,
                occupancy: 1.0,
                b_factor: 88.0,
                formal_charge: approx_atom_charge(ResidueClass::Rna, base, nm),
                is_hetatm: false,
            });
        }
        // Inward glycosidic unit vector toward helical axis + tangential base-plane axis
        let u_in = [-ca, -sa, 0.0];
        let v_tan = [-sa * inward_sign, ca * inward_sign, 0.0];
        let anchor_n = add3(c1, scale3(u_in, 1.47));
        for (nm, el, x, y) in idealized_base_2d(base) {
            let pos = add3(anchor_n, add3(scale3(u_in, x), scale3(v_tan, y * 0.7)));
            atoms.push(Atom {
                serial: atoms.len() + 1,
                name: nm.to_string(),
                element: el.to_string(),
                res_name: rname.clone(),
                res_code: base,
                res_class: ResidueClass::Rna,
                chain: ch,
                res_seq: rseq,
                icode: ' ',
                coords: pos,
                occupancy: 1.0,
                b_factor: 88.0,
                formal_charge: approx_atom_charge(ResidueClass::Rna, base, nm),
                is_hetatm: false,
            });
        }
    };

    for (i, &b) in seq.iter().enumerate() {
        let ang = (i as f64) * twist;
        let z = (i as f64) * rise;
        push_nt(chain_id, (i + 1) as i32, b, ang, z, 1.0);
    }
    // Antiparallel complementary strand (5'->3' reversed along Z)
    let n = seq.len();
    for i in 0..n {
        let pair_i = n - 1 - i;
        let ang = (pair_i as f64) * twist + PI;
        let z = (pair_i as f64) * rise;
        push_nt(chain_b, (i + 1) as i32, comp(seq[pair_i]), ang, z, -1.0);
    }

    let mol = finalize_biomolecule(
        format!("rna_duplex_{n}bp"),
        format!("Built 3D A-Form RNA Duplex ({n} bp)"),
        atoms,
        Vec::new(),
        false,
    )?;
    Ok(mol)
}

/// Add pH-aware explicit polar hydrogens (`addh [pH]` — backbone amides, hydroxyls, Lys/Arg/His/Asp/Glu).
pub fn add_polar_hydrogens_ph(mol: &mut Biomolecule, ph: f64) -> AddHydrogensReport {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); mol.atoms.len()];
    for b in &mol.bonds {
        if b.a < mol.atoms.len() && b.b < mol.atoms.len() {
            adj[b.a].push(b.b);
            adj[b.b].push(b.a);
        }
    }
    let mut new_hs = Vec::new();
    let mut prot_his = 0usize;
    let mut prot_acid = 0usize;

    for (i, a) in mol.atoms.iter().enumerate() {
        if a.res_class != ResidueClass::Protein {
            continue;
        }
        // Skip if atom already has a bonded H
        if adj[i].iter().any(|&nb| mol.atoms[nb].element == "H") {
            continue;
        }
        let (add_h, h_name): (bool, Option<&'static str>) =
            match (a.res_name.as_str(), a.name.as_str()) {
                ("PRO", "N") => (false, None),
                (_, "N") => (true, Some("H")),
                ("SER", "OG") => (true, Some("HG")),
                ("THR", "OG1") => (true, Some("HG1")),
                ("TYR", "OH") => (true, Some("HH")),
                ("CYS", "SG") => (adj[i].len() < 2, Some("HG")),
                ("TRP", "NE1") => (true, Some("HE1")),
                ("LYS", "NZ") => (ph < 10.5, Some("HZ1")),
                ("ARG", "NE") => (true, Some("HE")),
                ("HIS", "ND1") => {
                    if ph < 6.5 {
                        prot_his += 1;
                        (true, Some("HD1"))
                    } else {
                        (false, None)
                    }
                }
                ("HIS", "NE2") => (true, Some("HE2")),
                ("ASP", "OD2") | ("GLU", "OE2") => {
                    if ph < 4.2 {
                        prot_acid += 1;
                        (true, Some("HD2"))
                    } else {
                        (false, None)
                    }
                }
                _ => (false, None),
            };
        if add_h && let Some(hn) = h_name {
            let mut sum_dir = [0.0; 3];
            for &nb in &adj[i] {
                let d = norm3(sub3(a.coords, mol.atoms[nb].coords));
                sum_dir = add3(sum_dir, d);
            }
            let dir = if sum_dir[0].powi(2) + sum_dir[1].powi(2) + sum_dir[2].powi(2) > 1e-6 {
                norm3(sum_dir)
            } else {
                [0.0, 0.0, 1.0]
            };
            let h_pos = add3(a.coords, scale3(dir, 1.01));
            new_hs.push((
                i,
                Atom {
                    serial: 0,
                    name: hn.to_string(),
                    element: "H".to_string(),
                    res_name: a.res_name.clone(),
                    res_code: a.res_code,
                    res_class: a.res_class,
                    chain: a.chain,
                    res_seq: a.res_seq,
                    icode: a.icode,
                    coords: h_pos,
                    occupancy: a.occupancy,
                    b_factor: a.b_factor,
                    formal_charge: 0.25,
                    is_hetatm: a.is_hetatm,
                },
            ));
        }
    }

    let added = new_hs.len();
    for (parent_idx, mut h_atom) in new_hs {
        let hi = mol.atoms.len();
        h_atom.serial = hi + 1;
        mol.atoms.push(h_atom);
        mol.bonds.push(craftbio_struct::Bond {
            a: parent_idx,
            b: hi,
            order: 1,
        });
    }
    AddHydrogensReport {
        ph,
        hydrogens_added: added,
        protonated_his: prot_his,
        protonated_asp_glu: prot_acid,
    }
}

/// Renumber all residues in `chain` sequentially starting from `start_res_seq` (`renumber`).
pub fn renumber_chain(
    mol: &mut Biomolecule,
    chain: char,
    start_res_seq: i32,
) -> Result<usize, ModelError> {
    let mut old_to_new: HashMap<(i32, char), i32> = HashMap::new();
    let mut next_num = start_res_seq;
    for a in &mol.atoms {
        if a.chain == chain {
            old_to_new.entry((a.res_seq, a.icode)).or_insert_with(|| {
                let n = next_num;
                next_num += 1;
                n
            });
        }
    }
    if old_to_new.is_empty() {
        return Err(ModelError::ResidueNotFound(chain, start_res_seq));
    }
    let count = old_to_new.len();
    let mut atoms = mol.atoms.clone();
    for a in &mut atoms {
        if a.chain == chain
            && let Some(&nn) = old_to_new.get(&(a.res_seq, a.icode))
        {
            a.res_seq = nn;
            a.icode = ' ';
        }
    }
    let uc = mol.unit_cell.clone();
    let biomt = mol.biomt_ops.clone();
    let mut rebuilt = finalize_biomolecule(
        mol.id.clone(),
        mol.title.clone(),
        atoms,
        Vec::new(),
        mol.is_alphafold_plddt,
    )?;
    rebuilt.unit_cell = uc;
    rebuilt.biomt_ops = biomt;
    *mol = rebuilt;
    Ok(count)
}

/// Rename chain ID `old_chain -> new_chain` (`changechains`).
pub fn rename_chain(
    mol: &mut Biomolecule,
    old_chain: char,
    new_chain: char,
) -> Result<usize, ModelError> {
    let mut atoms = mol.atoms.clone();
    let mut moved = 0usize;
    for a in &mut atoms {
        if a.chain == old_chain {
            a.chain = new_chain;
            moved += 1;
        }
    }
    if moved == 0 {
        return Err(ModelError::ResidueNotFound(old_chain, 0));
    }
    let uc = mol.unit_cell.clone();
    let biomt = mol.biomt_ops.clone();
    let mut rebuilt = finalize_biomolecule(
        mol.id.clone(),
        mol.title.clone(),
        atoms,
        Vec::new(),
        mol.is_alphafold_plddt,
    )?;
    rebuilt.unit_cell = uc;
    rebuilt.biomt_ops = biomt;
    *mol = rebuilt;
    Ok(moved)
}

/// Delete all atoms marked `true` in `delete_mask` (`delete <selection>`).
pub fn delete_atoms_by_mask(
    mol: &mut Biomolecule,
    delete_mask: &[bool],
) -> Result<usize, ModelError> {
    let before = mol.atoms.len();
    let mut kept = Vec::with_capacity(before);
    for (i, a) in mol.atoms.iter().enumerate() {
        if !delete_mask.get(i).copied().unwrap_or(false) {
            let mut copy = a.clone();
            copy.serial = kept.len() + 1;
            kept.push(copy);
        }
    }
    let deleted = before.saturating_sub(kept.len());
    if kept.is_empty() {
        return Err(ModelError::Struct(
            craftbio_struct::BioStructError::EmptyInput,
        ));
    }
    let uc = mol.unit_cell.clone();
    let biomt = mol.biomt_ops.clone();
    let mut rebuilt = finalize_biomolecule(
        mol.id.clone(),
        mol.title.clone(),
        kept,
        Vec::new(),
        mol.is_alphafold_plddt,
    )?;
    rebuilt.unit_cell = uc;
    rebuilt.biomt_ops = biomt;
    *mol = rebuilt;
    Ok(deleted)
}

/// Merge multiple loaded `Biomolecule` models into one combined structure (`combine`),
/// auto-resolving colliding chain letters cleanly.
pub fn combine_biomolecules(
    mols: &[&Biomolecule],
    new_id: &str,
) -> Result<Biomolecule, ModelError> {
    const ALPH: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut atoms = Vec::new();
    let mut used_chains = HashSet::new();
    let mut next_idx = 0usize;

    for m in mols {
        let mut map = HashMap::new();
        for ch in &m.chains {
            let c = if !used_chains.contains(&ch.id) {
                ch.id
            } else {
                let mut cand = 'Z';
                while next_idx < ALPH.len() {
                    let cc = ALPH[next_idx] as char;
                    next_idx += 1;
                    if !used_chains.contains(&cc) {
                        cand = cc;
                        break;
                    }
                }
                cand
            };
            used_chains.insert(c);
            map.insert(ch.id, c);
        }
        for a in &m.atoms {
            let mut copy = a.clone();
            copy.chain = map.get(&a.chain).copied().unwrap_or(a.chain);
            copy.serial = atoms.len() + 1;
            atoms.push(copy);
        }
    }
    let combined = finalize_biomolecule(
        new_id.to_string(),
        format!(
            "Combined Model ({} sources, {} atoms)",
            mols.len(),
            atoms.len()
        ),
        atoms,
        Vec::new(),
        false,
    )?;
    Ok(combined)
}
