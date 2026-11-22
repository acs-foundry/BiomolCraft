#![forbid(unsafe_code)]

//! 3D Protein-Protein / Antibody-Antigen Interface Analysis:
//! Shrake-Rupley `ΔBSA` (total/polar/apolar), IMGT-annotated paratope & epitope mapping,
//! H-bonds, salt bridges, hydrophobic contacts, π-π / cation-π interactions, and packing proxy.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use bio_core::{
    AntibodyAnnotation, Atom3D, BioError, ProteinStructure, Residue3D, annotate_chain,
    compute_shrake_rupley_sasa, dist3,
};
use serde::{Deserialize, Serialize};

use crate::alascan::AlaScanReport;

/// Individual non-covalent atomic/residue interaction across the antigen–binder interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceContact {
    pub kind: String,
    pub binder_chain: char,
    pub binder_res: String,
    pub binder_region: String,
    pub binder_atom: String,
    pub antigen_chain: char,
    pub antigen_res: String,
    pub antigen_atom: String,
    pub distance_a: f64,
}

/// Summary of an epitope (antigen) or paratope (binder) interface residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceResidueSummary {
    pub chain: char,
    pub res_seq: i32,
    pub aa1: char,
    pub res_label: String,
    pub region: String,
    pub unbound_sasa_a2: f64,
    pub bound_sasa_a2: f64,
    pub delta_bsa_a2: f64,
    pub atomic_contacts: usize,
    pub hbond_count: usize,
    pub salt_bridge_count: usize,
}

/// Complete 3D PPI / Antibody–Antigen Interface Report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceReport {
    pub title: String,
    pub antigen_chains: Vec<char>,
    pub binder_chains: Vec<char>,
    pub total_bsa_a2: f64,
    pub polar_bsa_a2: f64,
    pub apolar_bsa_a2: f64,
    pub polar_bsa_fraction: f64,
    pub total_atomic_contacts: usize,
    pub hbond_count: usize,
    pub salt_bridge_count: usize,
    pub hydrophobic_contact_count: usize,
    pub pi_interaction_count: usize,
    pub shape_complementarity_proxy: f64,
    pub binder_annotations: Vec<AntibodyAnnotation>,
    pub cdr_bsa_breakdown: BTreeMap<String, f64>,
    pub paratope_residues: Vec<InterfaceResidueSummary>,
    pub epitope_residues: Vec<InterfaceResidueSummary>,
    pub key_contacts: Vec<InterfaceContact>,
    pub caveats: Vec<String>,
}

fn is_polar_atom(a: &Atom3D) -> bool {
    matches!(a.element.as_str(), "N" | "O" | "S")
}

fn is_hbond_partner(a: &Atom3D) -> bool {
    matches!(a.element.as_str(), "N" | "O")
}

fn is_acidic_sidechain_oxygen(a: &Atom3D) -> bool {
    (a.res_aa1 == 'D' && matches!(a.name.as_str(), "OD1" | "OD2"))
        || (a.res_aa1 == 'E' && matches!(a.name.as_str(), "OE1" | "OE2"))
}

fn is_basic_sidechain_nitrogen(a: &Atom3D) -> bool {
    (a.res_aa1 == 'K' && a.name == "NZ")
        || (a.res_aa1 == 'R' && matches!(a.name.as_str(), "NH1" | "NH2" | "NE"))
        || (a.res_aa1 == 'H' && matches!(a.name.as_str(), "ND1" | "NE2"))
}

fn aromatic_ring_center(atoms: &[Atom3D], r: &Residue3D) -> Option<[f64; 3]> {
    if !matches!(r.aa1, 'F' | 'Y' | 'W' | 'H') {
        return None;
    }
    let mut sum = [0.0_f64; 3];
    let mut cnt = 0usize;
    for &idx in &r.atom_indices {
        let nm = atoms[idx].name.as_str();
        let in_ring = match r.aa1 {
            'F' | 'Y' => matches!(nm, "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ"),
            'W' => matches!(
                nm,
                "CG" | "CD1" | "CD2" | "NE1" | "CE2" | "CE3" | "CZ2" | "CZ3" | "CH2"
            ),
            'H' => matches!(nm, "CG" | "ND1" | "CD2" | "CE1" | "NE2"),
            _ => false,
        };
        if in_ring {
            sum[0] += atoms[idx].coords[0];
            sum[1] += atoms[idx].coords[1];
            sum[2] += atoms[idx].coords[2];
            cnt += 1;
        }
    }
    if cnt >= 4 {
        let inv = 1.0 / (cnt as f64);
        Some([sum[0] * inv, sum[1] * inv, sum[2] * inv])
    } else {
        None
    }
}

/// Analyze a 3D antigen–binder complex: exact Shrake-Rupley `ΔBSA`, IMGT-annotated paratope/epitope,
/// H-bonds, salt bridges, hydrophobic contacts, and π-π / cation-π interactions.
pub fn analyze_ppi_interface(
    structure: &ProteinStructure,
    antigen_chains: &[char],
    binder_chains: &[char],
) -> Result<InterfaceReport, BioError> {
    if antigen_chains.is_empty() || binder_chains.is_empty() {
        return Err(BioError::Structure(
            "both antigen_chains and binder_chains must be non-empty".to_string(),
        ));
    }
    let ag_set: HashSet<char> = antigen_chains.iter().copied().collect();
    let bind_set: HashSet<char> = binder_chains.iter().copied().collect();

    if ag_set.iter().any(|c| bind_set.contains(c)) {
        return Err(BioError::Structure(
            "antigen_chains and binder_chains must not overlap".to_string(),
        ));
    }

    // Collect subset atom lists preserving mapping back to structure.atoms
    let mut complex_atoms = Vec::new();
    let mut complex_map: HashMap<usize, usize> = HashMap::new();
    let mut ag_atoms = Vec::new();
    let mut ag_map: HashMap<usize, usize> = HashMap::new();
    let mut bind_atoms = Vec::new();
    let mut bind_map: HashMap<usize, usize> = HashMap::new();

    for (orig_idx, a) in structure.atoms.iter().enumerate() {
        if ag_set.contains(&a.chain) {
            complex_map.insert(orig_idx, complex_atoms.len());
            complex_atoms.push(a.clone());
            ag_map.insert(orig_idx, ag_atoms.len());
            ag_atoms.push(a.clone());
        } else if bind_set.contains(&a.chain) {
            complex_map.insert(orig_idx, complex_atoms.len());
            complex_atoms.push(a.clone());
            bind_map.insert(orig_idx, bind_atoms.len());
            bind_atoms.push(a.clone());
        }
    }
    if ag_atoms.is_empty() || bind_atoms.is_empty() {
        return Err(BioError::Structure(format!(
            "missing atoms for requested chains (available chains: {:?})",
            structure.chain_ids()
        )));
    }

    // Shrake-Rupley SASA in complex and unbound states
    let sasa_complex = compute_shrake_rupley_sasa(&complex_atoms, 1.4, 42);
    let sasa_ag = compute_shrake_rupley_sasa(&ag_atoms, 1.4, 42);
    let sasa_bind = compute_shrake_rupley_sasa(&bind_atoms, 1.4, 42);

    let mut sum_ag_unbound = 0.0_f64;
    let mut sum_bind_unbound = 0.0_f64;
    let mut sum_complex = 0.0_f64;
    let mut pol_unbound = 0.0_f64;
    let mut pol_complex = 0.0_f64;

    for (orig_idx, a) in structure.atoms.iter().enumerate() {
        let Some(&ci) = complex_map.get(&orig_idx) else {
            continue;
        };
        let sc = sasa_complex[ci];
        sum_complex += sc;
        let is_pol = is_polar_atom(a);
        if is_pol {
            pol_complex += sc;
        }
        if let Some(&ai) = ag_map.get(&orig_idx) {
            let su = sasa_ag[ai];
            sum_ag_unbound += su;
            if is_pol {
                pol_unbound += su;
            }
        } else if let Some(&bi) = bind_map.get(&orig_idx) {
            let su = sasa_bind[bi];
            sum_bind_unbound += su;
            if is_pol {
                pol_unbound += su;
            }
        }
    }

    let total_bsa = (0.5 * (sum_ag_unbound + sum_bind_unbound - sum_complex)).max(0.0);
    let polar_bsa = (0.5 * (pol_unbound - pol_complex)).clamp(0.0, total_bsa);
    let apolar_bsa = (total_bsa - polar_bsa).max(0.0);
    let polar_frac = if total_bsa > 1e-6 {
        polar_bsa / total_bsa
    } else {
        0.0
    };

    // Annotate binder chains with IMGT/Kabat CDRs
    let mut binder_annotations = Vec::new();
    let mut binder_region_lookup: HashMap<(char, usize), String> = HashMap::new();
    for ch in &structure.chains {
        if bind_set.contains(&ch.id) {
            let ann = annotate_chain(&format!("Chain_{}", ch.id), &ch.sequence);
            for (ri, nr) in ann.numbered.iter().enumerate() {
                binder_region_lookup.insert((ch.id, ri), nr.region.short().to_string());
            }
            binder_annotations.push(ann);
        }
    }

    // Gather residues per side
    let mut ag_res_list: Vec<(&Residue3D, usize)> = Vec::new();
    let mut bind_res_list: Vec<(&Residue3D, usize)> = Vec::new();
    for ch in &structure.chains {
        if ag_set.contains(&ch.id) {
            for (ri, r) in ch.residues.iter().enumerate() {
                ag_res_list.push((r, ri));
            }
        } else if bind_set.contains(&ch.id) {
            for (ri, r) in ch.residues.iter().enumerate() {
                bind_res_list.push((r, ri));
            }
        }
    }

    // Contact classification across binder–antigen pairs
    let mut key_contacts: Vec<InterfaceContact> = Vec::new();
    let mut res_contact_counts: HashMap<(char, i32, char), (usize, usize, usize)> = HashMap::new();
    let mut total_atomic_contacts = 0usize;
    let mut hbond_count = 0usize;
    let mut salt_bridge_count = 0usize;
    let mut hydrophobic_contact_count = 0usize;
    let mut pi_interaction_count = 0usize;

    for &(br, bri) in &bind_res_list {
        let Some(bca) = br.ca_idx.or(br.atom_indices.first().copied()) else {
            continue;
        };
        let b_pos = structure.atoms[bca].coords;
        let b_region = binder_region_lookup
            .get(&(br.chain, bri))
            .cloned()
            .unwrap_or_else(|| "OTHER".to_string());
        let b_res_lbl = format!("{}{}", br.aa1, br.res_seq);

        for &(ar, _ari) in &ag_res_list {
            let Some(aca) = ar.ca_idx.or(ar.atom_indices.first().copied()) else {
                continue;
            };
            // Fast CA pre-filter (max sidechain reach ~14 A)
            if dist3(b_pos, structure.atoms[aca].coords) > 15.0 {
                continue;
            }
            let a_res_lbl = format!("{}{}", ar.aa1, ar.res_seq);

            for &bi in &br.atom_indices {
                let ba = &structure.atoms[bi];
                for &ai in &ar.atom_indices {
                    let aa = &structure.atoms[ai];
                    let d = dist3(ba.coords, aa.coords);
                    if d > 4.5 {
                        continue;
                    }
                    total_atomic_contacts += 1;
                    let be = res_contact_counts
                        .entry((br.chain, br.res_seq, br.icode))
                        .or_insert((0, 0, 0));
                    be.0 += 1;
                    let ae = res_contact_counts
                        .entry((ar.chain, ar.res_seq, ar.icode))
                        .or_insert((0, 0, 0));
                    ae.0 += 1;

                    let is_sb = d <= 4.0
                        && ((is_acidic_sidechain_oxygen(ba) && is_basic_sidechain_nitrogen(aa))
                            || (is_basic_sidechain_nitrogen(ba) && is_acidic_sidechain_oxygen(aa)));
                    let is_hb = !is_sb && d <= 3.5 && is_hbond_partner(ba) && is_hbond_partner(aa);
                    let is_hyd = d <= 4.2 && ba.element == "C" && aa.element == "C";

                    if is_sb {
                        salt_bridge_count += 1;
                        if let Some(e) =
                            res_contact_counts.get_mut(&(br.chain, br.res_seq, br.icode))
                        {
                            e.2 += 1;
                        }
                        if let Some(e) =
                            res_contact_counts.get_mut(&(ar.chain, ar.res_seq, ar.icode))
                        {
                            e.2 += 1;
                        }
                        key_contacts.push(InterfaceContact {
                            kind: "Salt Bridge".to_string(),
                            binder_chain: br.chain,
                            binder_res: b_res_lbl.clone(),
                            binder_region: b_region.clone(),
                            binder_atom: ba.name.clone(),
                            antigen_chain: ar.chain,
                            antigen_res: a_res_lbl.clone(),
                            antigen_atom: aa.name.clone(),
                            distance_a: (d * 100.0).round() / 100.0,
                        });
                    } else if is_hb {
                        hbond_count += 1;
                        if let Some(e) =
                            res_contact_counts.get_mut(&(br.chain, br.res_seq, br.icode))
                        {
                            e.1 += 1;
                        }
                        if let Some(e) =
                            res_contact_counts.get_mut(&(ar.chain, ar.res_seq, ar.icode))
                        {
                            e.1 += 1;
                        }
                        key_contacts.push(InterfaceContact {
                            kind: "H-Bond".to_string(),
                            binder_chain: br.chain,
                            binder_res: b_res_lbl.clone(),
                            binder_region: b_region.clone(),
                            binder_atom: ba.name.clone(),
                            antigen_chain: ar.chain,
                            antigen_res: a_res_lbl.clone(),
                            antigen_atom: aa.name.clone(),
                            distance_a: (d * 100.0).round() / 100.0,
                        });
                    } else if is_hyd {
                        hydrophobic_contact_count += 1;
                    }
                }
            }

            // Aromatic pi-pi and cation-pi detection
            let b_ring = aromatic_ring_center(&structure.atoms, br);
            let a_ring = aromatic_ring_center(&structure.atoms, ar);
            if let (Some(bc), Some(ac)) = (b_ring, a_ring) {
                let d = dist3(bc, ac);
                if d <= 5.5 {
                    pi_interaction_count += 1;
                    key_contacts.push(InterfaceContact {
                        kind: "Pi-Pi".to_string(),
                        binder_chain: br.chain,
                        binder_res: b_res_lbl.clone(),
                        binder_region: b_region.clone(),
                        binder_atom: "RING".to_string(),
                        antigen_chain: ar.chain,
                        antigen_res: a_res_lbl.clone(),
                        antigen_atom: "RING".to_string(),
                        distance_a: (d * 100.0).round() / 100.0,
                    });
                }
            } else if let Some(bc) = b_ring {
                for &ai in &ar.atom_indices {
                    if is_basic_sidechain_nitrogen(&structure.atoms[ai]) {
                        let d = dist3(bc, structure.atoms[ai].coords);
                        if d <= 5.4 {
                            pi_interaction_count += 1;
                            key_contacts.push(InterfaceContact {
                                kind: "Cation-Pi".to_string(),
                                binder_chain: br.chain,
                                binder_res: b_res_lbl.clone(),
                                binder_region: b_region.clone(),
                                binder_atom: "RING".to_string(),
                                antigen_chain: ar.chain,
                                antigen_res: a_res_lbl.clone(),
                                antigen_atom: structure.atoms[ai].name.clone(),
                                distance_a: (d * 100.0).round() / 100.0,
                            });
                            break;
                        }
                    }
                }
            } else if let Some(ac) = a_ring {
                for &bi in &br.atom_indices {
                    if is_basic_sidechain_nitrogen(&structure.atoms[bi]) {
                        let d = dist3(ac, structure.atoms[bi].coords);
                        if d <= 5.4 {
                            pi_interaction_count += 1;
                            key_contacts.push(InterfaceContact {
                                kind: "Cation-Pi".to_string(),
                                binder_chain: br.chain,
                                binder_res: b_res_lbl.clone(),
                                binder_region: b_region.clone(),
                                binder_atom: structure.atoms[bi].name.clone(),
                                antigen_chain: ar.chain,
                                antigen_res: a_res_lbl.clone(),
                                antigen_atom: "RING".to_string(),
                                distance_a: (d * 100.0).round() / 100.0,
                            });
                            break;
                        }
                    }
                }
            }
        }
    }

    // Build paratope and epitope per-residue ΔBSA summaries + CDR BSA breakdown
    let mut paratope_residues = Vec::new();
    let mut cdr_bsa_breakdown: BTreeMap<String, f64> = BTreeMap::new();
    for &(br, bri) in &bind_res_list {
        let mut su = 0.0_f64;
        let mut sb = 0.0_f64;
        for &idx in &br.atom_indices {
            if let Some(&bi) = bind_map.get(&idx) {
                su += sasa_bind[bi];
            }
            if let Some(&ci) = complex_map.get(&idx) {
                sb += sasa_complex[ci];
            }
        }
        let dbsa = (su - sb).max(0.0);
        let (nc, nhb, nsb) = res_contact_counts
            .get(&(br.chain, br.res_seq, br.icode))
            .copied()
            .unwrap_or((0, 0, 0));
        if dbsa >= 1.0 || nc > 0 {
            let reg = binder_region_lookup
                .get(&(br.chain, bri))
                .cloned()
                .unwrap_or_else(|| "OTHER".to_string());
            let key = format!("{}:{}", br.chain, reg);
            *cdr_bsa_breakdown.entry(key).or_insert(0.0) += dbsa;
            paratope_residues.push(InterfaceResidueSummary {
                chain: br.chain,
                res_seq: br.res_seq,
                aa1: br.aa1,
                res_label: format!("{}{}", br.aa1, br.res_seq),
                region: reg,
                unbound_sasa_a2: (su * 10.0).round() / 10.0,
                bound_sasa_a2: (sb * 10.0).round() / 10.0,
                delta_bsa_a2: (dbsa * 10.0).round() / 10.0,
                atomic_contacts: nc,
                hbond_count: nhb,
                salt_bridge_count: nsb,
            });
        }
    }
    for v in cdr_bsa_breakdown.values_mut() {
        *v = (*v * 10.0).round() / 10.0;
    }

    let mut epitope_residues = Vec::new();
    for &(ar, _ari) in &ag_res_list {
        let mut su = 0.0_f64;
        let mut sb = 0.0_f64;
        for &idx in &ar.atom_indices {
            if let Some(&ai) = ag_map.get(&idx) {
                su += sasa_ag[ai];
            }
            if let Some(&ci) = complex_map.get(&idx) {
                sb += sasa_complex[ci];
            }
        }
        let dbsa = (su - sb).max(0.0);
        let (nc, nhb, nsb) = res_contact_counts
            .get(&(ar.chain, ar.res_seq, ar.icode))
            .copied()
            .unwrap_or((0, 0, 0));
        if dbsa >= 1.0 || nc > 0 {
            epitope_residues.push(InterfaceResidueSummary {
                chain: ar.chain,
                res_seq: ar.res_seq,
                aa1: ar.aa1,
                res_label: format!("{}{}", ar.aa1, ar.res_seq),
                region: "EPITOPE".to_string(),
                unbound_sasa_a2: (su * 10.0).round() / 10.0,
                bound_sasa_a2: (sb * 10.0).round() / 10.0,
                delta_bsa_a2: (dbsa * 10.0).round() / 10.0,
                atomic_contacts: nc,
                hbond_count: nhb,
                salt_bridge_count: nsb,
            });
        }
    }

    paratope_residues.sort_by(|a, b| b.delta_bsa_a2.total_cmp(&a.delta_bsa_a2));
    epitope_residues.sort_by(|a, b| b.delta_bsa_a2.total_cmp(&a.delta_bsa_a2));
    key_contacts.sort_by(|a, b| a.distance_a.total_cmp(&b.distance_a));

    // Shape complementarity / packing density proxy (0..1; clinical Ab-Ag typically 0.62–0.76)
    let density = if total_bsa > 10.0 {
        (total_atomic_contacts as f64) / (total_bsa / 100.0)
    } else {
        0.0
    };
    let sc_proxy = if total_bsa < 20.0 {
        0.0
    } else {
        (0.45
            + 0.011 * density.min(30.0)
            + 0.008 * (hbond_count + salt_bridge_count).min(15) as f64)
            .clamp(0.35, 0.85)
    };

    let caveats = vec![
        "Heavy-atom geometry evaluated on input crystal/co-folded coordinates without explicit water bridges or dynamic loop sampling.".to_string(),
        "Shape complementarity (Sc proxy) reflects heavy-atom packing density across ΔBSA; confirm affinity kinetics experimentally via SPR/BLI.".to_string(),
    ];

    Ok(InterfaceReport {
        title: structure.title.clone(),
        antigen_chains: antigen_chains.to_vec(),
        binder_chains: binder_chains.to_vec(),
        total_bsa_a2: (total_bsa * 10.0).round() / 10.0,
        polar_bsa_a2: (polar_bsa * 10.0).round() / 10.0,
        apolar_bsa_a2: (apolar_bsa * 10.0).round() / 10.0,
        polar_bsa_fraction: (polar_frac * 1000.0).round() / 1000.0,
        total_atomic_contacts,
        hbond_count,
        salt_bridge_count,
        hydrophobic_contact_count,
        pi_interaction_count,
        shape_complementarity_proxy: (sc_proxy * 1000.0).round() / 1000.0,
        binder_annotations,
        cdr_bsa_breakdown,
        paratope_residues,
        epitope_residues,
        key_contacts,
        caveats,
    })
}

/// Render a clean Markdown 3D PPI interface + optional Alanine Scanning hotspot dossier.
#[must_use]
pub fn render_interface_markdown(rep: &InterfaceReport, alascan: Option<&AlaScanReport>) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# 3D Protein-Protein Interface Report: {}\n", rep.title);
    let _ = writeln!(
        md,
        "- **Antigen Chains:** `{:?}` | **Binder Chains:** `{:?}`",
        rep.antigen_chains, rep.binder_chains
    );
    let _ = writeln!(
        md,
        "- **Buried Surface Area (`ΔBSA`):** **`{:.1} Å²`** (Apolar `{:.1} Å²` / Polar `{:.1} Å²`, `{:.1}%` polar)",
        rep.total_bsa_a2,
        rep.apolar_bsa_a2,
        rep.polar_bsa_a2,
        rep.polar_bsa_fraction * 100.0
    );
    let _ = writeln!(
        md,
        "- **Contacts (`<= 4.5 Å`):** `{}` atomic | **H-Bonds:** `{}` | **Salt Bridges:** `{}` | **Hydrophobic:** `{}` | **π-π / Cation-π:** `{}` | **Packing Proxy:** `{:.2}`\n",
        rep.total_atomic_contacts,
        rep.hbond_count,
        rep.salt_bridge_count,
        rep.hydrophobic_contact_count,
        rep.pi_interaction_count,
        rep.shape_complementarity_proxy
    );

    if !rep.cdr_bsa_breakdown.is_empty() {
        let _ = writeln!(md, "## 1. Paratope BSA by Chain & IMGT Region\n");
        let _ = writeln!(md, "| Chain:Region | ΔBSA (Å²) |");
        let _ = writeln!(md, "|---|---:|");
        for (k, v) in &rep.cdr_bsa_breakdown {
            let _ = writeln!(md, "| `{k}` | {v:.1} |");
        }
        md.push('\n');
    }

    let _ = writeln!(md, "## 2. Top Paratope Residues (Binder)\n");
    let _ = writeln!(
        md,
        "| Chain | Residue | IMGT Region | ΔBSA (Å²) | Contacts | H-Bonds | Salt Bridges |"
    );
    let _ = writeln!(md, "|---|---|---|---:|---:|---:|---:|");
    for r in rep.paratope_residues.iter().take(15) {
        let _ = writeln!(
            md,
            "| {} | `{}` | {} | {:.1} | {} | {} | {} |",
            r.chain,
            r.res_label,
            r.region,
            r.delta_bsa_a2,
            r.atomic_contacts,
            r.hbond_count,
            r.salt_bridge_count
        );
    }

    let _ = writeln!(md, "\n## 3. Top Epitope Residues (Antigen)\n");
    let _ = writeln!(
        md,
        "| Chain | Residue | ΔBSA (Å²) | Contacts | H-Bonds | Salt Bridges |"
    );
    let _ = writeln!(md, "|---|---|---:|---:|---:|---:|");
    for r in rep.epitope_residues.iter().take(15) {
        let _ = writeln!(
            md,
            "| {} | `{}` | {:.1} | {} | {} | {} |",
            r.chain,
            r.res_label,
            r.delta_bsa_a2,
            r.atomic_contacts,
            r.hbond_count,
            r.salt_bridge_count
        );
    }

    if let Some(ala) = alascan {
        let _ = writeln!(
            md,
            "\n## 4. Computational Alanine Scanning Hotspots (`ΔΔG_bind`)\n"
        );
        let _ = writeln!(
            md,
            "- **Method:** {} | **Hotspots (`ΔΔG >= 1.5 kcal/mol`):** `{}` paratope, `{}` epitope\n",
            ala.method,
            ala.paratope_hotspots
                .iter()
                .filter(|h| h.classification == "Hotspot")
                .count(),
            ala.epitope_hotspots
                .iter()
                .filter(|h| h.classification == "Hotspot")
                .count()
        );
        let _ = writeln!(
            md,
            "| Side | Chain | Residue | Region | ΔΔG (kcal/mol) | Class | Key Driver |"
        );
        let _ = writeln!(md, "|---|---|---|---|---:|---|---|");
        for h in ala.paratope_hotspots.iter().take(10) {
            let _ = writeln!(
                md,
                "| Paratope | {} | `{}` | {} | **{:+.2}** | {} | {} |",
                h.chain, h.res_label, h.region, h.ddg_kcal_mol, h.classification, h.driver
            );
        }
        for h in ala.epitope_hotspots.iter().take(8) {
            let _ = writeln!(
                md,
                "| Epitope | {} | `{}` | EPITOPE | **{:+.2}** | {} | {} |",
                h.chain, h.res_label, h.ddg_kcal_mol, h.classification, h.driver
            );
        }
    }

    if !rep.key_contacts.is_empty() {
        let _ = writeln!(md, "\n## 5. Key Specific Non-Covalent Interactions\n");
        let _ = writeln!(md, "| Type | Binder | Region | Antigen | Distance (Å) |");
        let _ = writeln!(md, "|---|---|---|---|---:|");
        for c in rep.key_contacts.iter().take(20) {
            let _ = writeln!(
                md,
                "| {} | `{}:{}.{}` | {} | `{}:{}.{}` | {:.2} |",
                c.kind,
                c.binder_chain,
                c.binder_res,
                c.binder_atom,
                c.binder_region,
                c.antigen_chain,
                c.antigen_res,
                c.antigen_atom,
                c.distance_a
            );
        }
    }

    md
}
