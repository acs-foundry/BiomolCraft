#![forbid(unsafe_code)]

//! Pure-Rust Generalized Born / Surface Area (`GB/SA-lite`) Implicit Solvation,
//! Electrostatic Dipole Moment, and Single-Trajectory Interface Binding Free Energy (`ΔG_bind`).

use craftbio_struct::{
    Biomolecule, DipoleMomentReport, ResidueClass, compute_dipole_moment, compute_sasa, dist3,
    vdw_radius,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Energy decomposition (kcal/mol) under Still GB/SA-lite implicit solvent (`ε_w = 78.5`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbSaEnergyTerms {
    pub e_coulomb_kcal: f64,
    pub e_gb_polar_kcal: f64,
    pub e_sa_nonpolar_kcal: f64,
    pub e_vdw_kcal: f64,
    pub total_kcal: f64,
    pub total_sasa_a2: f64,
}

/// Complete GB/SA + Dipole + optional interface `ΔG_bind` report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbSaReport {
    pub title: String,
    pub complex_energy: GbSaEnergyTerms,
    pub dipole: DipoleMomentReport,
    pub interface_chains_a: Vec<char>,
    pub interface_chains_b: Vec<char>,
    pub delta_g_bind_kcal: Option<f64>,
    pub delta_e_elec_gb_kcal: Option<f64>,
    pub delta_e_vdw_kcal: Option<f64>,
    pub delta_e_sa_kcal: Option<f64>,
    pub top_interface_hotspots: Vec<(String, f64)>,
}

fn eval_subset_gbsa(mol: &Biomolecule, indices: &[usize]) -> GbSaEnergyTerms {
    if indices.is_empty() {
        return GbSaEnergyTerms {
            e_coulomb_kcal: 0.0,
            e_gb_polar_kcal: 0.0,
            e_sa_nonpolar_kcal: 0.0,
            e_vdw_kcal: 0.0,
            total_kcal: 0.0,
            total_sasa_a2: 0.0,
        };
    }
    let subset_atoms: Vec<_> = indices.iter().map(|&i| mol.atoms[i].clone()).collect();
    let sasa_vec = compute_sasa(&subset_atoms, 1.4, 24);
    let total_sasa: f64 = sasa_vec.iter().sum();
    let e_sa = 0.0054 * total_sasa; // ACE nonpolar cavity term (kcal/mol/A^2)

    let n = subset_atoms.len();
    let radii: Vec<f64> = subset_atoms
        .iter()
        .map(|a| vdw_radius(&a.element).max(1.2))
        .collect();

    // Effective Born radii via fast Hawkins-Cramer-Truhlar burial integral (parallelized)
    let born_r: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let ri = radii[i];
            let pi = subset_atoms[i].coords;
            let mut inv_r = 1.0 / ri;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = dist3(pi, subset_atoms[j].coords);
                if d > 1.2 && d < 10.5 {
                    let rj = radii[j];
                    let v_eff = (rj * rj * rj) / (d * d * d * d * 4.5);
                    inv_r -= v_eff.min(0.06);
                }
            }
            (1.0 / inv_r.clamp(1.0 / 28.0, 1.0 / ri)).clamp(ri, 28.0)
        })
        .collect();

    // Pairwise Coulomb + Still GB polar solvation + soft-core LJ using consistent ε_in = 2.0
    // so intermolecular Coulomb attraction and Born desolvation penalty cancel physically!
    const EPS_IN: f64 = 2.0;
    const EPS_OUT: f64 = 78.5;
    let gb_pref = -0.5 * 332.0637 * (1.0 / EPS_IN - 1.0 / EPS_OUT);
    let (e_coul, e_gb, e_vdw) = (0..n)
        .into_par_iter()
        .map(|i| {
            let ai = &subset_atoms[i];
            let qi = ai.formal_charge;
            let bi = born_r[i];
            let ri = radii[i];
            let mut c_acc = 0.0_f64;
            let mut gb_acc = if qi.abs() > 1e-6 {
                gb_pref * qi * qi / bi
            } else {
                0.0
            };
            let mut vdw_acc = 0.0_f64;
            for j in (i + 1)..n {
                let aj = &subset_atoms[j];
                let d = dist3(ai.coords, aj.coords);
                if d > 14.0 {
                    continue;
                }
                let d_safe = d.max(1.35);
                let qj = aj.formal_charge;
                if qi.abs() > 1e-6 && qj.abs() > 1e-6 {
                    let same_res = ai.chain == aj.chain && ai.res_seq == aj.res_seq;
                    if !same_res {
                        c_acc += 332.0637 * qi * qj / (EPS_IN * d_safe);
                    }
                    let bj = born_r[j];
                    let bibj = bi * bj;
                    let f_gb =
                        (d_safe * d_safe + bibj * (-d_safe * d_safe / (4.0 * bibj)).exp()).sqrt();
                    gb_acc += 2.0 * gb_pref * qi * qj / f_gb;
                }
                if d < 8.5 && !(ai.chain == aj.chain && (ai.res_seq - aj.res_seq).abs() <= 1) {
                    let sigma = 0.5 * (ri + radii[j]);
                    let ratio = (sigma / d.max(0.76 * sigma)).min(1.35);
                    let r6 = ratio.powi(6);
                    let lj = 0.24 * (r6 * r6 - 2.0 * r6);
                    vdw_acc += lj.clamp(-0.85, 3.0);
                }
            }
            (c_acc, gb_acc, vdw_acc)
        })
        .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    let total = e_coul + e_gb + e_sa + e_vdw;
    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    GbSaEnergyTerms {
        e_coulomb_kcal: r2(e_coul),
        e_gb_polar_kcal: r2(e_gb),
        e_sa_nonpolar_kcal: r2(e_sa),
        e_vdw_kcal: r2(e_vdw),
        total_kcal: r2(total),
        total_sasa_a2: r2(total_sasa),
    }
}

/// Evaluate pure-Rust GB/SA solvation, electrostatic dipole, and (for multi-chain complexes)
/// single-trajectory interface binding free energy `ΔG_bind`.
#[must_use]
pub fn compute_gbsa_energy(
    mol: &Biomolecule,
    chains_a: Option<&[char]>,
    chains_b: Option<&[char]>,
) -> GbSaReport {
    let polymer_idx: Vec<usize> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            matches!(
                a.res_class,
                ResidueClass::Protein
                    | ResidueClass::Rna
                    | ResidueClass::Dna
                    | ResidueClass::Ligand
            )
        })
        .map(|(i, _)| i)
        .collect();

    let complex_energy = eval_subset_gbsa(mol, &polymer_idx);
    let dipole = compute_dipole_moment(mol);

    // Determine receptor/ligand chain split for ΔG_bind
    let poly_chains: Vec<char> = mol
        .chains
        .iter()
        .filter(|c| {
            matches!(
                c.dominant_class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            )
        })
        .map(|c| c.id)
        .collect();

    let (group_a, group_b): (Vec<char>, Vec<char>) = match (chains_a, chains_b) {
        (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => (a.to_vec(), b.to_vec()),
        _ if poly_chains.len() >= 2 => {
            // Default: last chain (often antigen in Ab-Ag complexes) vs rest
            let last = poly_chains[poly_chains.len() - 1];
            let rest = poly_chains[..poly_chains.len() - 1].to_vec();
            (rest, vec![last])
        }
        _ => (Vec::new(), Vec::new()),
    };

    let (dg_bind, d_elec_gb, d_vdw, d_sa, hotspots) = if !group_a.is_empty() && !group_b.is_empty()
    {
        let idx_a: Vec<usize> = polymer_idx
            .iter()
            .copied()
            .filter(|&i| group_a.contains(&mol.atoms[i].chain))
            .collect();
        let idx_b: Vec<usize> = polymer_idx
            .iter()
            .copied()
            .filter(|&i| group_b.contains(&mol.atoms[i].chain))
            .collect();
        let ga = eval_subset_gbsa(mol, &idx_a);
        let gb = eval_subset_gbsa(mol, &idx_b);
        let r2 = |v: f64| (v * 100.0).round() / 100.0;
        let de = r2(
            (complex_energy.e_coulomb_kcal + complex_energy.e_gb_polar_kcal)
                - (ga.e_coulomb_kcal + ga.e_gb_polar_kcal + gb.e_coulomb_kcal + gb.e_gb_polar_kcal),
        );
        let dv = r2(complex_energy.e_vdw_kcal - (ga.e_vdw_kcal + gb.e_vdw_kcal));
        let ds =
            r2(complex_energy.e_sa_nonpolar_kcal - (ga.e_sa_nonpolar_kcal + gb.e_sa_nonpolar_kcal));
        let dg = r2(de + dv + ds);

        // Per-residue interface energy contribution ranking
        let mut res_scores: Vec<(String, f64)> = Vec::new();
        for ch in &mol.chains {
            let in_a = group_a.contains(&ch.id);
            let in_b = group_b.contains(&ch.id);
            if !in_a && !in_b {
                continue;
            }
            let partner = if in_a { &idx_b } else { &idx_a };
            for r in &ch.residues {
                let mut e_res = 0.0_f64;
                for &ia in &r.atom_indices {
                    let aa = &mol.atoms[ia];
                    let ra = vdw_radius(&aa.element);
                    for &ib in partner {
                        let ab = &mol.atoms[ib];
                        let d = dist3(aa.coords, ab.coords);
                        if d < 6.5 {
                            let sigma = 0.5 * (ra + vdw_radius(&ab.element));
                            let ratio = (sigma / d.max(0.75 * sigma)).min(1.4);
                            let r6 = ratio.powi(6);
                            e_res += (0.14 * (r6 * r6 - 2.0 * r6)).clamp(-0.6, 3.0);
                            if aa.formal_charge.abs() > 1e-4 && ab.formal_charge.abs() > 1e-4 {
                                e_res += 332.0 * aa.formal_charge * ab.formal_charge
                                    / (16.0 * d.max(2.0));
                            }
                        }
                    }
                }
                if e_res.abs() > 0.4 {
                    res_scores.push((
                        format!("{}:{}{}", r.chain, r.res_name, r.res_seq),
                        (e_res * 100.0).round() / 100.0,
                    ));
                }
            }
        }
        res_scores.sort_by(|x, y| x.1.total_cmp(&y.1));
        res_scores.truncate(10);
        (Some(dg), Some(de), Some(dv), Some(ds), res_scores)
    } else {
        (None, None, None, None, Vec::new())
    };

    GbSaReport {
        title: mol.title.clone(),
        complex_energy,
        dipole,
        interface_chains_a: group_a,
        interface_chains_b: group_b,
        delta_g_bind_kcal: dg_bind,
        delta_e_elec_gb_kcal: d_elec_gb,
        delta_e_vdw_kcal: d_vdw,
        delta_e_sa_kcal: d_sa,
        top_interface_hotspots: hotspots,
    }
}
