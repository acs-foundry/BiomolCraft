#![forbid(unsafe_code)]

//! Computational Alanine Scanning (`ΔΔG_bind` virtual sidechain truncation beyond `Cβ`) and
//! Paratope Site Saturation Mutagenesis (19-aa steric/electrostatic/SAP/PTM screening).

use std::collections::{HashMap, HashSet};

use bio_core::{
    Atom3D, BioError, ProteinStructure, Residue3D, annotate_chain, black_mould_sap_hydro,
    compute_shrake_rupley_sasa, dist3,
};
use serde::{Deserialize, Serialize};

/// Alanine-scanning energetic contribution for a single interface residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlaScanResidue {
    pub side: String,
    pub chain: char,
    pub res_seq: i32,
    pub aa1: char,
    pub res_label: String,
    pub region: String,
    pub ddg_kcal_mol: f64,
    pub classification: String,
    pub lost_sidechain_contacts: usize,
    pub lost_hbonds: usize,
    pub lost_salt_bridges: usize,
    pub sidechain_dbsa_a2: f64,
    pub driver: String,
}

/// Complete computational Alanine Scanning report for paratope & epitope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlaScanReport {
    pub method: String,
    pub paratope_hotspots: Vec<AlaScanResidue>,
    pub epitope_hotspots: Vec<AlaScanResidue>,
    pub scientific_caveat: String,
}

/// Single amino-acid substitution evaluated during paratope site-saturation mutagenesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationProposal {
    pub mutation: String,
    pub target_aa: char,
    pub estimated_ddg_kcal_mol: f64,
    pub steric_clash_risk: String,
    pub delta_sap_hydro: f64,
    pub introduces_ptm: Option<String>,
    pub rationale: String,
}

/// Complete 19-amino-acid site-saturation report for one paratope residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationSiteReport {
    pub chain: char,
    pub res_seq: i32,
    pub wild_type_aa: char,
    pub region: String,
    pub nearby_antigen_residues: Vec<String>,
    pub ranked_mutations: Vec<MutationProposal>,
}

fn is_sidechain_beyond_cb(atom_name: &str) -> bool {
    !matches!(atom_name, "N" | "CA" | "C" | "O" | "OXT" | "CB")
}

fn is_acidic_sc(a: &Atom3D) -> bool {
    (a.res_aa1 == 'D' && matches!(a.name.as_str(), "OD1" | "OD2"))
        || (a.res_aa1 == 'E' && matches!(a.name.as_str(), "OE1" | "OE2"))
}

fn is_basic_sc(a: &Atom3D) -> bool {
    (a.res_aa1 == 'K' && a.name == "NZ")
        || (a.res_aa1 == 'R' && matches!(a.name.as_str(), "NH1" | "NH2" | "NE"))
        || (a.res_aa1 == 'H' && matches!(a.name.as_str(), "ND1" | "NE2"))
}

fn aa_volume_a3(aa: char) -> f64 {
    match aa {
        'G' => 60.1,
        'A' => 88.6,
        'S' => 89.0,
        'C' => 108.5,
        'D' => 111.1,
        'P' => 112.7,
        'N' => 114.1,
        'T' => 116.1,
        'E' => 138.4,
        'V' => 140.0,
        'Q' => 143.8,
        'H' => 153.2,
        'M' => 162.9,
        'I' => 166.7,
        'L' => 166.7,
        'K' => 168.6,
        'R' => 173.4,
        'F' => 189.9,
        'Y' => 193.6,
        'W' => 227.8,
        _ => 130.0,
    }
}

fn aa_formal_charge(aa: char) -> f64 {
    match aa {
        'K' | 'R' => 1.0,
        'H' => 0.2,
        'D' | 'E' => -1.0,
        _ => 0.0,
    }
}

/// Run rapid empirical 3D computational Alanine Scanning across all paratope and epitope residues.
/// Positive `ΔΔG_bind` (kcal/mol) indicates destabilization upon mutation to Ala (binding hotspot).
pub fn run_alanine_scan(
    structure: &ProteinStructure,
    antigen_chains: &[char],
    binder_chains: &[char],
) -> Result<AlaScanReport, BioError> {
    let ag_set: HashSet<char> = antigen_chains.iter().copied().collect();
    let bind_set: HashSet<char> = binder_chains.iter().copied().collect();

    let mut complex_atoms = Vec::new();
    let mut complex_map = HashMap::new();
    let mut ag_atoms = Vec::new();
    let mut ag_map = HashMap::new();
    let mut bind_atoms = Vec::new();
    let mut bind_map = HashMap::new();

    for (i, a) in structure.atoms.iter().enumerate() {
        if ag_set.contains(&a.chain) {
            complex_map.insert(i, complex_atoms.len());
            complex_atoms.push(a.clone());
            ag_map.insert(i, ag_atoms.len());
            ag_atoms.push(a.clone());
        } else if bind_set.contains(&a.chain) {
            complex_map.insert(i, complex_atoms.len());
            complex_atoms.push(a.clone());
            bind_map.insert(i, bind_atoms.len());
            bind_atoms.push(a.clone());
        }
    }
    if ag_atoms.is_empty() || bind_atoms.is_empty() {
        return Err(BioError::Structure(
            "cannot run alanine scan: missing antigen or binder atoms".to_string(),
        ));
    }

    let sasa_complex = compute_shrake_rupley_sasa(&complex_atoms, 1.4, 42);
    let sasa_ag = compute_shrake_rupley_sasa(&ag_atoms, 1.4, 42);
    let sasa_bind = compute_shrake_rupley_sasa(&bind_atoms, 1.4, 42);

    // IMGT annotation for binder chains
    let mut binder_regions: HashMap<(char, usize), String> = HashMap::new();
    for ch in &structure.chains {
        if bind_set.contains(&ch.id) {
            let ann = annotate_chain(&format!("Chain_{}", ch.id), &ch.sequence);
            for (ri, nr) in ann.numbered.iter().enumerate() {
                binder_regions.insert((ch.id, ri), nr.region.short().to_string());
            }
        }
    }

    let ag_atom_indices: Vec<usize> = structure
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| ag_set.contains(&a.chain))
        .map(|(i, _)| i)
        .collect();
    let bind_atom_indices: Vec<usize> = structure
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| bind_set.contains(&a.chain))
        .map(|(i, _)| i)
        .collect();

    let eval_residue = |r: &Residue3D,
                        side: &str,
                        region: String,
                        partner_indices: &[usize],
                        unbound_map: &HashMap<usize, usize>,
                        unbound_sasa: &[f64]|
     -> Option<AlaScanResidue> {
        let mut sc_dbsa = 0.0_f64;
        let mut sc_apolar_dbsa = 0.0_f64;
        let mut sc_contacts = 0usize;
        let mut lost_hb = 0usize;
        let mut lost_sb = 0usize;
        let mut has_any_interface_contact = false;

        for &idx in &r.atom_indices {
            let a = &structure.atoms[idx];
            let su = unbound_map.get(&idx).map_or(0.0, |&ui| unbound_sasa[ui]);
            let sb = complex_map.get(&idx).map_or(0.0, |&ci| sasa_complex[ci]);
            let da = (su - sb).max(0.0);
            if da >= 0.5 {
                has_any_interface_contact = true;
            }
            if !is_sidechain_beyond_cb(&a.name) {
                continue;
            }
            sc_dbsa += da;
            if a.element == "C" {
                sc_apolar_dbsa += da;
            }
            let mut atom_hb_counted = false;
            let mut atom_sb_counted = false;
            for &pi in partner_indices {
                let pa = &structure.atoms[pi];
                let d = dist3(a.coords, pa.coords);
                if d <= 4.3 {
                    sc_contacts += 1;
                    has_any_interface_contact = true;
                }
                if !atom_sb_counted
                    && d <= 4.0
                    && ((is_acidic_sc(a) && is_basic_sc(pa))
                        || (is_basic_sc(a) && is_acidic_sc(pa)))
                {
                    lost_sb += 1;
                    atom_sb_counted = true;
                } else if !atom_hb_counted
                    && d <= 3.45
                    && matches!(a.element.as_str(), "N" | "O")
                    && matches!(pa.element.as_str(), "N" | "O")
                {
                    lost_hb += 1;
                    atom_hb_counted = true;
                }
            }
        }

        if !has_any_interface_contact && sc_contacts == 0 {
            return None;
        }

        let (ddg, driver) = if matches!(r.aa1, 'A' | 'G') {
            (
                0.0,
                "Backbone / Gly-Ala (no sidechain beyond Cβ)".to_string(),
            )
        } else {
            let e_vdw = ((sc_contacts as f64) * 0.11).min(2.2);
            let e_solv = (sc_apolar_dbsa * 0.022).min(2.4);
            let e_hb = (lost_hb as f64) * 0.85;
            let e_sb = (lost_sb as f64) * 1.15;
            let arom_bonus = if matches!(r.aa1, 'W' | 'Y' | 'F') && sc_dbsa > 35.0 {
                0.45
            } else {
                0.0
            };
            let raw = (e_vdw + e_solv + e_hb + e_sb + arom_bonus).clamp(-0.5, 5.5);
            let mut parts = Vec::new();
            if lost_sb > 0 {
                parts.push(format!("{lost_sb} salt-bridge"));
            }
            if lost_hb > 0 {
                parts.push(format!("{lost_hb} H-bond"));
            }
            if sc_apolar_dbsa > 20.0 {
                parts.push(format!("{sc_apolar_dbsa:.0} Å² apolar ΔBSA"));
            }
            if parts.is_empty() {
                parts.push(format!("{sc_contacts} packing contacts"));
            }
            ((raw * 100.0).round() / 100.0, parts.join(" + "))
        };

        let classification = if ddg >= 1.5 {
            "Hotspot"
        } else if ddg >= 0.5 {
            "Support"
        } else {
            "Neutral"
        }
        .to_string();

        Some(AlaScanResidue {
            side: side.to_string(),
            chain: r.chain,
            res_seq: r.res_seq,
            aa1: r.aa1,
            res_label: format!("{}{}", r.aa1, r.res_seq),
            region,
            ddg_kcal_mol: ddg,
            classification,
            lost_sidechain_contacts: sc_contacts,
            lost_hbonds: lost_hb,
            lost_salt_bridges: lost_sb,
            sidechain_dbsa_a2: (sc_dbsa * 10.0).round() / 10.0,
            driver,
        })
    };

    let mut paratope_hotspots = Vec::new();
    let mut epitope_hotspots = Vec::new();

    for ch in &structure.chains {
        if bind_set.contains(&ch.id) {
            for (ri, r) in ch.residues.iter().enumerate() {
                let reg = binder_regions
                    .get(&(ch.id, ri))
                    .cloned()
                    .unwrap_or_else(|| "OTHER".to_string());
                if let Some(item) =
                    eval_residue(r, "Paratope", reg, &ag_atom_indices, &bind_map, &sasa_bind)
                {
                    paratope_hotspots.push(item);
                }
            }
        } else if ag_set.contains(&ch.id) {
            for r in &ch.residues {
                if let Some(item) = eval_residue(
                    r,
                    "Epitope",
                    "EPITOPE".to_string(),
                    &bind_atom_indices,
                    &ag_map,
                    &sasa_ag,
                ) {
                    epitope_hotspots.push(item);
                }
            }
        }
    }

    paratope_hotspots.sort_by(|a, b| b.ddg_kcal_mol.total_cmp(&a.ddg_kcal_mol));
    epitope_hotspots.sort_by(|a, b| b.ddg_kcal_mol.total_cmp(&a.ddg_kcal_mol));

    Ok(AlaScanReport {
        method: "Virtual Cβ-truncation empirical ΔΔG_bind (vdW + H-bond + salt-bridge + apolar ΔBSA)".to_string(),
        paratope_hotspots,
        epitope_hotspots,
        scientific_caveat: "Computational surrogate ranking for experimental mutagenesis prioritization; confirm hotspot ΔΔG via SPR/BLI alanine-mutant kinetics.".to_string(),
    })
}

/// Screen all 19 single-point amino-acid substitutions at a chosen paratope position for
/// steric complementarity, antigen electrostatic matching, SAP change, and PTM avoidance.
pub fn saturation_mutagenesis_site(
    structure: &ProteinStructure,
    antigen_chains: &[char],
    binder_chain: char,
    res_seq: i32,
) -> Result<SaturationSiteReport, BioError> {
    let ag_set: HashSet<char> = antigen_chains.iter().copied().collect();
    let ch = structure
        .chain(binder_chain)
        .ok_or_else(|| BioError::Structure(format!("binder chain '{binder_chain}' not found")))?;
    let (ri, res) = ch
        .residues
        .iter()
        .enumerate()
        .find(|(_, r)| r.res_seq == res_seq)
        .ok_or_else(|| {
            BioError::Structure(format!("residue {binder_chain}:{res_seq} not found"))
        })?;

    let ann = annotate_chain(&format!("Chain_{binder_chain}"), &ch.sequence);
    let region = ann
        .numbered
        .get(ri)
        .map_or("OTHER", |n| n.region.short())
        .to_string();

    let prev_aa = ri
        .checked_sub(1)
        .and_then(|k| ch.residues.get(k))
        .map(|r| r.aa1);
    let next_aa = ch.residues.get(ri + 1).map(|r| r.aa1);
    let next2_aa = ch.residues.get(ri + 2).map(|r| r.aa1);

    let site_coord = res
        .cb_idx
        .or(res.ca_idx)
        .map(|i| structure.atoms[i].coords)
        .ok_or_else(|| {
            BioError::Structure("target residue has no backbone coordinates".to_string())
        })?;

    // Inspect local antigen microenvironment within 8.5 A of site Cβ
    let mut local_ag_charge = 0.0_f64;
    let mut local_ag_hydro = 0.0_f64;
    let mut min_ag_dist = 99.0_f64;
    let mut nearby_res_set: Vec<String> = Vec::new();
    let mut seen_ag: HashSet<(char, i32)> = HashSet::new();

    for a in &structure.atoms {
        if !ag_set.contains(&a.chain) {
            continue;
        }
        let d = dist3(site_coord, a.coords);
        if d < min_ag_dist {
            min_ag_dist = d;
        }
        if d <= 8.5 && seen_ag.insert((a.chain, a.res_seq)) {
            nearby_res_set.push(format!("{}:{}{}", a.chain, a.res_aa1, a.res_seq));
            local_ag_charge += aa_formal_charge(a.res_aa1);
            local_ag_hydro += black_mould_sap_hydro(a.res_aa1);
        }
    }

    let wt_aa = res.aa1;
    let wt_vol = aa_volume_a3(wt_aa);
    let wt_sap = black_mould_sap_hydro(wt_aa);
    let mut ranked = Vec::new();

    for cand in "ACDEFGHIKLMNPQRSTVWY".chars() {
        if cand == wt_aa {
            continue;
        }
        let dvol = aa_volume_a3(cand) - wt_vol;
        let dsap = black_mould_sap_hydro(cand) - wt_sap;
        let q_cand = aa_formal_charge(cand);

        // Steric clash penalty when growing volume into tight cavity
        let (clash_pen, clash_risk) = if dvol > 55.0 && min_ag_dist < 4.5 {
            (1.6, "High (bulky in tight pocket)".to_string())
        } else if dvol > 30.0 && min_ag_dist < 4.8 {
            (0.65, "Moderate".to_string())
        } else {
            (0.0, "Low".to_string())
        };

        // Electrostatic complementarity with nearby antigen residues
        let elec_gain = if local_ag_charge.abs() >= 0.5 && q_cand != 0.0 {
            if local_ag_charge * q_cand < 0.0 {
                -0.85 // stabilizing opposite charge
            } else {
                0.90 // repulsive same charge
            }
        } else {
            0.0
        };

        // Hydrophobic/aromatic packing bonus near hydrophobic antigen patch
        let pack_gain = if local_ag_hydro > 0.3 && matches!(cand, 'Y' | 'W' | 'F' | 'L' | 'I') {
            -0.55
        } else if matches!(cand, 'Y' | 'Q' | 'N' | 'S' | 'T' | 'H') && min_ag_dist < 6.5 {
            -0.25 // versatile H-bond capacity
        } else if cand == 'P' || cand == 'G' {
            0.85 // backbone perturbation
        } else {
            0.0
        };

        // Check PTM liability introduction
        let ptm = if cand == 'C' {
            Some("Introduces unpaired Cys".to_string())
        } else if cand == 'M' {
            Some("Introduces Met oxidation site".to_string())
        } else if cand == 'N' && next_aa == Some('G') {
            Some("Creates NG deamidation motif".to_string())
        } else if cand == 'D' && matches!(next_aa, Some('G' | 'P')) {
            Some("Creates DG/DP isomerization/clipping motif".to_string())
        } else if cand == 'N' && next_aa != Some('P') && matches!(next2_aa, Some('S' | 'T')) {
            Some("Creates N-X-S/T Fab glycosylation sequon".to_string())
        } else if prev_aa == Some('N') && cand == 'G' {
            Some("Creates NG deamidation motif".to_string())
        } else {
            None
        };
        let ptm_pen = if ptm.is_some() { 0.8 } else { 0.0 };

        let raw_ddg: f64 = clash_pen + elec_gain + pack_gain + ptm_pen;
        let est_ddg = (raw_ddg * 100.0).round() / 100.0;
        let rationale = if elec_gain < 0.0 {
            "Complementary electrostatics to nearby antigen surface".to_string()
        } else if pack_gain < -0.4 {
            "Favorable aromatic/hydrophobic interface packing".to_string()
        } else if dsap < -0.2 && est_ddg <= 0.1 {
            "Improves colloidal solubility (lowers SAP) without clash".to_string()
        } else if clash_pen > 0.5 {
            "Steric bulk expansion risks clash".to_string()
        } else {
            "Conservative/neutral exploration candidate".to_string()
        };

        ranked.push(MutationProposal {
            mutation: format!("{binder_chain}:{wt_aa}{res_seq}{cand}"),
            target_aa: cand,
            estimated_ddg_kcal_mol: est_ddg,
            steric_clash_risk: clash_risk,
            delta_sap_hydro: (dsap * 1000.0).round() / 1000.0,
            introduces_ptm: ptm,
            rationale,
        });
    }

    ranked.sort_by(|a, b| {
        a.estimated_ddg_kcal_mol
            .total_cmp(&b.estimated_ddg_kcal_mol)
    });

    Ok(SaturationSiteReport {
        chain: binder_chain,
        res_seq,
        wild_type_aa: wt_aa,
        region,
        nearby_antigen_residues: nearby_res_set,
        ranked_mutations: ranked,
    })
}
