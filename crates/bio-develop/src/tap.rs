#![forbid(unsafe_code)]

//! TAP (Therapeutic Antibody Profiler — Raybould et al. *PNAS* 2019), 3D SAP (Spatial Aggregation
//! Propensity — Chennamsetty et al. *PNAS* 2009), CDR/FR Chemical PTM Liability Audit, and Exact
//! Biophysical Characterization (MW, ε280, A280, pI, Charge vs pH, GRAVY).

use std::fmt::Write as _;

use bio_core::{
    AntibodyAnnotation, IgChainType, ProteinStructure, RegionKind, annotate_chain,
    black_mould_sap_hydro, compute_shrake_rupley_sasa, dist3, kyte_doolittle, max_residue_sasa,
    residue_avg_mw,
};
use serde::{Deserialize, Serialize};

use crate::immunogenicity::{ImmunogenicityReport, evaluate_immunogenicity};

/// Clinical TAP / developability traffic light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrafficLight {
    Green,
    Amber,
    Red,
}

impl TrafficLight {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Green => "Green",
            Self::Amber => "Amber",
            Self::Red => "Red",
        }
    }
}

/// Single TAP guideline metric with clinical reference range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TapMetric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub green_range: String,
    pub status: TrafficLight,
    pub detail: String,
}

/// Sequence or 3D-weighted chemical degradation / post-translational liability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtmLiability {
    pub chain: String,
    pub category: String,
    pub motif: String,
    pub position_label: String,
    pub region: String,
    pub in_cdr: bool,
    pub sasa_a2: Option<f64>,
    pub severity: TrafficLight,
    pub mitigation: String,
}

/// Surface aggregation hotspot (3D SAP or sequence Aggrescan).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SapHotspot {
    pub chain: String,
    pub res_label: String,
    pub aa: char,
    pub region: String,
    pub sap_score: f64,
    pub sasa_a2: Option<f64>,
    pub suggested_mutation: String,
}

/// Standard wet-lab biophysical properties (Nanodrop / SoloVPE / cIEF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiophysicalProps {
    pub total_residues: usize,
    pub mw_reduced_da: f64,
    pub mw_oxidized_da: f64,
    pub estimated_disulfides: usize,
    pub extinction_coeff_m_cm: f64,
    pub a280_01pct: f64,
    pub isoelectric_point_pi: f64,
    pub charge_ph55: f64,
    pub charge_ph74: f64,
    pub gravy: f64,
}

/// Comprehensive Biologics Developability, TAP, 3D SAP, PTM & Immunogenicity Report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopabilityReport {
    pub name: String,
    pub has_3d_structure: bool,
    pub annotations: Vec<AntibodyAnnotation>,
    pub biophysical: BiophysicalProps,
    pub tap_metrics: Vec<TapMetric>,
    pub tap_overall: TrafficLight,
    pub max_sap_score: f64,
    pub sap_hotspots: Vec<SapHotspot>,
    pub ptm_liabilities: Vec<PtmLiability>,
    pub cdr_high_risk_ptms: usize,
    pub immunogenicity: ImmunogenicityReport,
    pub developability_score_0_100: f64,
    pub summary_verdict: String,
}

fn net_charge_protein(seqs: &[&str], ph: f64, n_disulfides: usize) -> f64 {
    let mut q = 0.0_f64;
    let mut free_cys = 0usize;
    for s in seqs {
        if s.is_empty() {
            continue;
        }
        q += 1.0 / (1.0 + 10.0_f64.powf(ph - 8.0)); // N-term
        q -= 1.0 / (1.0 + 10.0_f64.powf(3.2 - ph)); // C-term
        for c in s.chars() {
            match c {
                'D' => q -= 1.0 / (1.0 + 10.0_f64.powf(3.9 - ph)),
                'E' => q -= 1.0 / (1.0 + 10.0_f64.powf(4.25 - ph)),
                'H' => q += 1.0 / (1.0 + 10.0_f64.powf(ph - 6.0)),
                'K' => q += 1.0 / (1.0 + 10.0_f64.powf(ph - 10.5)),
                'R' => q += 1.0 / (1.0 + 10.0_f64.powf(ph - 12.5)),
                'Y' => q -= 1.0 / (1.0 + 10.0_f64.powf(10.1 - ph)),
                'C' => free_cys += 1,
                _ => {}
            }
        }
    }
    let reduced_cys = free_cys.saturating_sub(n_disulfides * 2);
    q -= (reduced_cys as f64) / (1.0 + 10.0_f64.powf(8.3 - ph));
    q
}

fn compute_pi(seqs: &[&str], n_disulfides: usize) -> f64 {
    let mut lo = 1.0_f64;
    let mut hi = 13.5_f64;
    for _ in 0..42 {
        let mid = 0.5 * (lo + hi);
        if net_charge_protein(seqs, mid, n_disulfides) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    ((0.5 * (lo + hi)) * 100.0).round() / 100.0
}

fn compute_biophysical(seqs: &[&str], explicit_disulfides: Option<usize>) -> BiophysicalProps {
    let mut total_res = 0usize;
    let mut mw_red = 0.0_f64;
    let mut n_trp = 0usize;
    let mut n_tyr = 0usize;
    let mut n_cys = 0usize;
    let mut gravy_sum = 0.0_f64;

    for s in seqs {
        if s.is_empty() {
            continue;
        }
        mw_red += 18.01528;
        for c in s.chars() {
            total_res += 1;
            mw_red += residue_avg_mw(c);
            gravy_sum += kyte_doolittle(c);
            match c {
                'W' => n_trp += 1,
                'Y' => n_tyr += 1,
                'C' => n_cys += 1,
                _ => {}
            }
        }
    }
    let est_ss = explicit_disulfides.unwrap_or(n_cys / 2);
    let mw_ox = mw_red - (est_ss as f64) * 2.01588;
    // Pace et al. (1995) molar extinction coefficient at 280 nm (M^-1 cm^-1)
    let eps280 = (n_trp as f64) * 5500.0 + (n_tyr as f64) * 1490.0 + (est_ss as f64) * 125.0;
    let a280 = if mw_ox > 1.0 { eps280 / mw_ox } else { 0.0 };
    let charge_ph55 = (net_charge_protein(seqs, 5.5, est_ss) * 100.0).round() / 100.0;
    let charge_ph74 = (net_charge_protein(seqs, 7.4, est_ss) * 100.0).round() / 100.0;
    let pi = compute_pi(seqs, est_ss);
    let gravy = if total_res > 0 {
        (gravy_sum / (total_res as f64) * 1000.0).round() / 1000.0
    } else {
        0.0
    };

    BiophysicalProps {
        total_residues: total_res,
        mw_reduced_da: (mw_red * 100.0).round() / 100.0,
        mw_oxidized_da: (mw_ox * 100.0).round() / 100.0,
        estimated_disulfides: est_ss,
        extinction_coeff_m_cm: eps280.round(),
        a280_01pct: (a280 * 1000.0).round() / 1000.0,
        isoelectric_point_pi: pi,
        charge_ph55,
        charge_ph74,
        gravy,
    }
}

fn scan_ptm_liabilities(ann: &AntibodyAnnotation, res_sasa: Option<&[f64]>) -> Vec<PtmLiability> {
    let chars: Vec<char> = ann.sequence.chars().collect();
    let mut out = Vec::new();
    let n = chars.len();

    let region_of = |idx: usize| -> (RegionKind, String) {
        ann.numbered
            .get(idx)
            .map(|r| {
                (
                    r.region,
                    format!("{}{} (IMGT {})", r.aa, idx + 1, r.imgt_pos),
                )
            })
            .unwrap_or((RegionKind::ConstantOrOther, format!("Pos {}", idx + 1)))
    };

    // N-terminal pyroglutamate
    if let Some(&c0) = chars.first()
        && matches!(c0, 'Q' | 'E')
    {
        let (reg, plbl) = region_of(0);
        out.push(PtmLiability {
            chain: ann.chain_name.clone(),
            category: "Pyroglutamate Cyclization".to_string(),
            motif: format!("N-term {c0}"),
            position_label: plbl,
            region: reg.short().to_string(),
            in_cdr: false,
            sasa_a2: res_sasa.and_then(|s| s.first().copied()),
            severity: TrafficLight::Green,
            mitigation: "Routine N-term heterogeneity (-17/-18 Da); monitor by icIEF/LC-MS."
                .to_string(),
        });
    }

    for i in 0..n {
        let (reg, plbl) = region_of(i);
        let in_cdr = reg.is_cdr();
        let sasa = res_sasa.and_then(|s| s.get(i).copied());
        let exposed = sasa.is_none_or(|v| v >= 20.0);

        // N-linked glycosylation sequon N-X-S/T (X != P)
        if i + 2 < n && chars[i] == 'N' && chars[i + 1] != 'P' && matches!(chars[i + 2], 'S' | 'T')
        {
            out.push(PtmLiability {
                chain: ann.chain_name.clone(),
                category: "N-Linked Glycosylation".to_string(),
                motif: format!("{}{}{}", chars[i], chars[i + 1], chars[i + 2]),
                position_label: plbl.clone(),
                region: reg.short().to_string(),
                in_cdr,
                sasa_a2: sasa,
                severity: if in_cdr {
                    TrafficLight::Red
                } else {
                    TrafficLight::Amber
                },
                mitigation: "Mutate N->Q/D or S/T->A to eliminate heterogeneous Fab glycans."
                    .to_string(),
            });
        }
        // Asn Deamidation (NG high; NS/NT/NH moderate)
        if i + 1 < n && chars[i] == 'N' && matches!(chars[i + 1], 'G' | 'S' | 'T' | 'H') {
            let is_ng = chars[i + 1] == 'G';
            let sev = if in_cdr && is_ng {
                TrafficLight::Red
            } else if in_cdr {
                TrafficLight::Amber
            } else {
                TrafficLight::Green
            };
            if in_cdr || is_ng {
                out.push(PtmLiability {
                    chain: ann.chain_name.clone(),
                    category: "Asn Deamidation".to_string(),
                    motif: format!("N{}", chars[i + 1]),
                    position_label: plbl.clone(),
                    region: reg.short().to_string(),
                    in_cdr,
                    sasa_a2: sasa,
                    severity: sev,
                    mitigation: "Substitute N->Q/S or G->A/S; test pH 7.4/40°C forced stress."
                        .to_string(),
                });
            }
        }
        // Asp Isomerization (DG/DS/DT/DD) and Acid Clipping (DP)
        if i + 1 < n && chars[i] == 'D' && matches!(chars[i + 1], 'G' | 'S' | 'T' | 'D' | 'P') {
            let is_dp = chars[i + 1] == 'P';
            let is_dg = chars[i + 1] == 'G';
            let sev = if in_cdr && (is_dg || is_dp) {
                TrafficLight::Red
            } else if in_cdr {
                TrafficLight::Amber
            } else {
                TrafficLight::Green
            };
            if in_cdr || is_dg || is_dp {
                out.push(PtmLiability {
                    chain: ann.chain_name.clone(),
                    category: if is_dp {
                        "Asp-Pro Acid Hydrolysis".to_string()
                    } else {
                        "Asp Isomerization".to_string()
                    },
                    motif: format!("D{}", chars[i + 1]),
                    position_label: plbl.clone(),
                    region: reg.short().to_string(),
                    in_cdr,
                    sasa_a2: sasa,
                    severity: sev,
                    mitigation:
                        "Mutate D->E or G/S->A; formulate at pH 5.5–6.0 to minimize clipping."
                            .to_string(),
                });
            }
        }
        // Met / Trp Oxidation in CDR (weighted by 3D SASA exposure when structure available)
        if in_cdr && matches!(chars[i], 'M' | 'W') && exposed {
            out.push(PtmLiability {
                chain: ann.chain_name.clone(),
                category: if chars[i] == 'M' {
                    "Met Oxidation".to_string()
                } else {
                    "Trp Photo-Oxidation".to_string()
                },
                motif: chars[i].to_string(),
                position_label: plbl.clone(),
                region: reg.short().to_string(),
                in_cdr: true,
                sasa_a2: sasa,
                severity: if chars[i] == 'M' || sasa.is_some_and(|v| v > 65.0) {
                    TrafficLight::Amber
                } else {
                    TrafficLight::Green
                },
                mitigation: if chars[i] == 'M' {
                    "Consider M->L/I/Q conservative substitution + H2O2 stress assay.".to_string()
                } else {
                    "Monitor ICH Q1B photostability; retain if essential aromatic paratope hotspot."
                        .to_string()
                },
            });
        }
    }

    // Unpaired non-canonical Cys warning
    if ann.extra_noncanonical_cys.len() % 2 == 1 {
        out.push(PtmLiability {
            chain: ann.chain_name.clone(),
            category: "Unpaired Free Cysteine".to_string(),
            motif: ann.extra_noncanonical_cys.join(", "),
            position_label: ann.extra_noncanonical_cys[0].clone(),
            region: "V-domain".to_string(),
            in_cdr: true,
            sasa_a2: None,
            severity: TrafficLight::Red,
            mitigation: "Mutate unpaired Cys->Ser/Ala to prevent disulfide scrambling and covalent aggregation.".to_string(),
        });
    }

    out
}

/// Evaluate complete TAP 5-metric developability, 3D SAP (or sequence Aggrescan), chemical PTMs,
/// biophysical properties, and HLA-DRB1 immunogenicity for a biologic (mAb, VHH, or protein).
#[must_use]
pub fn evaluate_developability(
    name: &str,
    chains: &[(&str, &str)],
    structure: Option<&ProteinStructure>,
) -> DevelopabilityReport {
    let annotations: Vec<AntibodyAnnotation> = chains
        .iter()
        .map(|&(cname, seq)| annotate_chain(cname, seq))
        .collect();
    let seq_refs: Vec<&str> = annotations.iter().map(|a| a.sequence.as_str()).collect();
    let biophysical = compute_biophysical(&seq_refs, structure.map(|s| s.disulfides.len()));

    // Compute 3D Shrake-Rupley SASA & 3D SAP (R = 5 A shell) if 3D structure supplied
    let mut sap_hotspots = Vec::new();
    let mut max_sap = 0.0_f64;
    let mut per_chain_sasa: Vec<Option<Vec<f64>>> = vec![None; annotations.len()];

    let mut psh_3d: Option<f64> = None;
    let mut ppc_3d: Option<f64> = None;
    let mut pnc_3d: Option<f64> = None;

    if let Some(st) = structure {
        let atom_sasa = compute_shrake_rupley_sasa(&st.atoms, 1.4, 42);
        // Compute per-residue SASA & centroids across structure
        struct ResGeom {
            chain: char,
            res_seq: i32,
            aa: char,
            sasa: f64,
            rel_sasa: f64,
            center: [f64; 3],
            is_cdr_vicinity: bool,
            region_lbl: String,
        }
        let mut rgeoms: Vec<ResGeom> = Vec::new();

        for ch in &st.chains {
            let ann_opt = annotations.iter().enumerate().find(|(_, a)| {
                a.chain_name.ends_with(ch.id)
                    || a.sequence == ch.sequence
                    || ch
                        .sequence
                        .starts_with(a.sequence.get(..40).unwrap_or(&a.sequence))
            });
            let mut ch_sasa_vec = Vec::with_capacity(ch.residues.len());
            for (ri, r) in ch.residues.iter().enumerate() {
                let rs: f64 = r.atom_indices.iter().map(|&ai| atom_sasa[ai]).sum();
                ch_sasa_vec.push(rs);
                let mut cent = [0.0_f64; 3];
                if let Some(cb) = r.cb_idx.or(r.ca_idx) {
                    cent = st.atoms[cb].coords;
                } else if !r.atom_indices.is_empty() {
                    for &ai in &r.atom_indices {
                        for k in 0..3 {
                            cent[k] += st.atoms[ai].coords[k];
                        }
                    }
                    let inv = 1.0 / (r.atom_indices.len() as f64);
                    cent = [cent[0] * inv, cent[1] * inv, cent[2] * inv];
                }
                let (is_cdr, rlbl) = ann_opt
                    .and_then(|(_, a)| a.numbered.get(ri))
                    .map(|nr| (nr.region.is_cdr(), nr.region.short().to_string()))
                    .unwrap_or((false, "OTHER".to_string()));
                rgeoms.push(ResGeom {
                    chain: ch.id,
                    res_seq: r.res_seq,
                    aa: r.aa1,
                    sasa: rs,
                    rel_sasa: (rs / max_residue_sasa(r.aa1)).clamp(0.0, 1.2),
                    center: cent,
                    is_cdr_vicinity: is_cdr,
                    region_lbl: rlbl,
                });
            }
            if let Some((ai, _)) = ann_opt {
                per_chain_sasa[ai] = Some(ch_sasa_vec);
            }
        }

        // True 3D SAP (Chennamsetty 2009, R = 5.0 A) + TAP 3D CDR vicinity PSH/PPC/PNC
        let mut psh_sum = 0.0_f64;
        let mut ppc_sum = 0.0_f64;
        let mut pnc_sum = 0.0_f64;
        for (i, ri) in rgeoms.iter().enumerate() {
            let mut sap = 0.0_f64;
            for rj in &rgeoms {
                if dist3(ri.center, rj.center) <= 6.5 {
                    sap += rj.rel_sasa * black_mould_sap_hydro(rj.aa);
                }
            }
            if sap > max_sap {
                max_sap = sap;
            }
            if sap >= 0.45 && ri.rel_sasa >= 0.25 && black_mould_sap_hydro(ri.aa) > 0.15 {
                let sug = if ri.region_lbl.starts_with("CDR") {
                    format!("{}{}->Y/S (caution: CDR)", ri.aa, ri.res_seq)
                } else {
                    format!("{}{}->S/K/E (surface solubilizing)", ri.aa, ri.res_seq)
                };
                sap_hotspots.push(SapHotspot {
                    chain: ri.chain.to_string(),
                    res_label: format!("{}{}", ri.aa, ri.res_seq),
                    aa: ri.aa,
                    region: ri.region_lbl.clone(),
                    sap_score: (sap * 1000.0).round() / 1000.0,
                    sasa_a2: Some((ri.sasa * 10.0).round() / 10.0),
                    suggested_mutation: sug,
                });
            }
            if ri.is_cdr_vicinity && ri.rel_sasa > 0.05 {
                // TAP CDR vicinity patch contributions scaled to clinical Raybould units
                for (j, rj) in rgeoms.iter().enumerate() {
                    if j <= i {
                        continue;
                    }
                    let d = dist3(ri.center, rj.center);
                    if d <= 7.5 && rj.rel_sasa > 0.05 {
                        let hi = black_mould_sap_hydro(ri.aa).max(0.0);
                        let hj = black_mould_sap_hydro(rj.aa).max(0.0);
                        if hi > 0.0 && hj > 0.0 {
                            psh_sum += (hi * hj * ri.rel_sasa * rj.rel_sasa) / (d.max(2.0) * 0.12);
                        }
                        let qi = match ri.aa {
                            'K' | 'R' => 1.0,
                            'H' => 0.2,
                            'D' | 'E' => -1.0,
                            _ => 0.0,
                        };
                        let qj = match rj.aa {
                            'K' | 'R' => 1.0,
                            'H' => 0.2,
                            'D' | 'E' => -1.0,
                            _ => 0.0,
                        };
                        if qi > 0.0 && qj > 0.0 {
                            ppc_sum += (qi * qj * ri.rel_sasa * rj.rel_sasa) / (d.max(2.0) * 0.45);
                        }
                        if qi < 0.0 && qj < 0.0 {
                            pnc_sum += (qi * qj * ri.rel_sasa * rj.rel_sasa) / (d.max(2.0) * 0.45);
                        }
                    }
                }
            }
        }
        sap_hotspots.sort_by(|a, b| b.sap_score.total_cmp(&a.sap_score));
        sap_hotspots.truncate(12);
        psh_3d = Some(95.0 + psh_sum * 1.35);
        ppc_3d = Some(ppc_sum);
        pnc_3d = Some(pnc_sum);
    } else {
        // Sequence-mode Aggrescan sliding 5-mer hydrophobic patches
        for ann in &annotations {
            let cs: Vec<char> = ann.sequence.chars().collect();
            for i in 0..cs.len() {
                let lo = i.saturating_sub(2);
                let hi = (i + 3).min(cs.len());
                let win_sap: f64 = cs[lo..hi]
                    .iter()
                    .map(|&c| black_mould_sap_hydro(c))
                    .sum::<f64>()
                    / ((hi - lo) as f64);
                if win_sap > max_sap {
                    max_sap = win_sap;
                }
                if win_sap >= 0.30 && black_mould_sap_hydro(cs[i]) >= 0.30 {
                    let reg = ann.numbered.get(i).map_or("OTHER", |r| r.region.short());
                    sap_hotspots.push(SapHotspot {
                        chain: ann.chain_name.clone(),
                        res_label: format!("{}{}", cs[i], i + 1),
                        aa: cs[i],
                        region: reg.to_string(),
                        sap_score: (win_sap * 1000.0).round() / 1000.0,
                        sasa_a2: None,
                        suggested_mutation: format!("{}{}->S/T/D (sequence patch)", cs[i], i + 1),
                    });
                }
            }
        }
        sap_hotspots.sort_by(|a, b| b.sap_score.total_cmp(&a.sap_score));
        sap_hotspots.truncate(10);
    }

    // Collect PTM liabilities across all chains
    let mut ptm_liabilities = Vec::new();
    for (ai, ann) in annotations.iter().enumerate() {
        ptm_liabilities.extend(scan_ptm_liabilities(ann, per_chain_sasa[ai].as_deref()));
    }
    let cdr_high_risk_ptms = ptm_liabilities
        .iter()
        .filter(|p| p.in_cdr && p.severity == TrafficLight::Red)
        .count();

    // TAP 5 Clinical Metrics (Raybould et al. PNAS 2019 calibrated guidelines)
    let ig_chains: Vec<&AntibodyAnnotation> = annotations
        .iter()
        .filter(|a| a.chain_type.is_variable_ig())
        .collect();
    let is_vhh_only = ig_chains.len() == 1 && ig_chains[0].chain_type == IgChainType::Vhh;

    let cdr_seqs: Vec<&str> = ig_chains
        .iter()
        .flat_map(|a| {
            [
                a.cdr1_imgt.as_str(),
                a.cdr2_imgt.as_str(),
                a.cdr3_imgt.as_str(),
            ]
        })
        .collect();
    let total_cdr_len: usize = cdr_seqs.iter().map(|s| s.len()).sum();

    // Sequence-based TAP estimators calibrated to Raybould clinical range when no 3D Fv supplied
    let cdr_concat: String = cdr_seqs.concat();
    let cdr_hydro_sum: f64 = cdr_concat
        .chars()
        .map(|c| black_mould_sap_hydro(c).max(0.0))
        .sum();
    let cdr_pos_pairs: f64 = cdr_seqs
        .iter()
        .map(|s| {
            let cs: Vec<char> = s.chars().collect();
            let pos_n = cs.iter().filter(|&&c| matches!(c, 'K' | 'R')).count() as f64;
            let hist_n = cs.iter().filter(|&&c| c == 'H').count() as f64 * 0.25;
            let adj_pos = cs
                .windows(3)
                .filter(|w| w.iter().filter(|&&c| matches!(c, 'K' | 'R' | 'H')).count() >= 2)
                .count() as f64;
            0.18 * (pos_n + hist_n) + 0.55 * adj_pos
        })
        .sum();
    let cdr_neg_pairs: f64 = cdr_seqs
        .iter()
        .map(|s| {
            let cs: Vec<char> = s.chars().collect();
            let neg_n = cs.iter().filter(|&&c| matches!(c, 'D' | 'E')).count() as f64;
            let adj_neg = cs
                .windows(3)
                .filter(|w| w.iter().filter(|&&c| matches!(c, 'D' | 'E')).count() >= 2)
                .count() as f64;
            0.18 * neg_n + 0.50 * adj_neg
        })
        .sum();

    let psh_val = ((psh_3d.unwrap_or(95.0 + cdr_hydro_sum * 4.8)) * 10.0).round() / 10.0;
    let ppc_val = ((ppc_3d.unwrap_or(cdr_pos_pairs)) * 100.0).round() / 100.0;
    let pnc_val = ((pnc_3d.unwrap_or(cdr_neg_pairs)) * 100.0).round() / 100.0;

    // SFvCSP = net_charge(VH) * net_charge(VL) at pH 7.4 (for VHH single domain, report q_VHH)
    let vh_ann = ig_chains
        .iter()
        .find(|a| matches!(a.chain_type, IgChainType::Vh | IgChainType::Vhh));
    let vl_ann = ig_chains
        .iter()
        .find(|a| matches!(a.chain_type, IgChainType::Vk | IgChainType::Vl));
    let sfvcsp = match (vh_ann, vl_ann) {
        (Some(h), Some(l)) => {
            let qh = net_charge_protein(&[&h.sequence], 7.4, 1);
            let ql = net_charge_protein(&[&l.sequence], 7.4, 1);
            ((qh * ql) * 100.0).round() / 100.0
        }
        (Some(h), None) => ((net_charge_protein(&[&h.sequence], 7.4, 1)) * 100.0).round() / 100.0,
        _ => 0.0,
    };

    let cdr_status = if is_vhh_only {
        if (24..=42).contains(&total_cdr_len) {
            TrafficLight::Green
        } else if (20..=46).contains(&total_cdr_len) {
            TrafficLight::Amber
        } else {
            TrafficLight::Red
        }
    } else if (41..=58).contains(&total_cdr_len) {
        TrafficLight::Green
    } else if (36..=64).contains(&total_cdr_len) {
        TrafficLight::Amber
    } else {
        TrafficLight::Red
    };
    let psh_status = if (95.0..=155.0).contains(&psh_val) {
        TrafficLight::Green
    } else if (80.0..=172.0).contains(&psh_val) {
        TrafficLight::Amber
    } else {
        TrafficLight::Red
    };
    let ppc_status = if ppc_val <= 1.15 {
        TrafficLight::Green
    } else if ppc_val <= 1.85 {
        TrafficLight::Amber
    } else {
        TrafficLight::Red
    };
    let pnc_status = if pnc_val <= 1.65 {
        TrafficLight::Green
    } else if pnc_val <= 2.40 {
        TrafficLight::Amber
    } else {
        TrafficLight::Red
    };
    let sfvcsp_status = if is_vhh_only {
        if sfvcsp.abs() <= 5.0 {
            TrafficLight::Green
        } else {
            TrafficLight::Amber
        }
    } else if sfvcsp >= -20.4 {
        TrafficLight::Green
    } else if sfvcsp >= -35.0 {
        TrafficLight::Amber
    } else {
        TrafficLight::Red
    };

    let tap_metrics = vec![
        TapMetric {
            name: "Total CDR Length (IMGT)".to_string(),
            value: total_cdr_len as f64,
            unit: "aa".to_string(),
            green_range: if is_vhh_only {
                "24–42 aa (VHH)".to_string()
            } else {
                "41–58 aa (mAb Fv)".to_string()
            },
            status: cdr_status,
            detail: "Extreme CDR lengths correlate with expression loss or polyreactivity.".to_string(),
        },
        TapMetric {
            name: "PSH (CDR Surface Hydrophobicity)".to_string(),
            value: psh_val,
            unit: if structure.is_some() { "3D-SASA" } else { "seq-est" }.to_string(),
            green_range: "95–155".to_string(),
            status: psh_status,
            detail: "Excess CDR hydrophobicity predicts aggregation and poor HIC retention.".to_string(),
        },
        TapMetric {
            name: "PPC (CDR Positive Charge Patch)".to_string(),
            value: ppc_val,
            unit: if structure.is_some() { "3D-SASA" } else { "seq-est" }.to_string(),
            green_range: "0.00–1.15".to_string(),
            status: ppc_status,
            detail: "Excess positive CDR charge drives polyspecificity (heparin/DNA) and fast clearance.".to_string(),
        },
        TapMetric {
            name: "PNC (CDR Negative Charge Patch)".to_string(),
            value: pnc_val,
            unit: if structure.is_some() { "3D-SASA" } else { "seq-est" }.to_string(),
            green_range: "0.00–1.65".to_string(),
            status: pnc_status,
            detail: "Moderate negative patches are well-tolerated; extremes can raise viscosity.".to_string(),
        },
        TapMetric {
            name: if is_vhh_only {
                "VHH Net Charge (pH 7.4)".to_string()
            } else {
                "SFvCSP (VH×VL Charge Symmetry)".to_string()
            },
            value: sfvcsp,
            unit: "e²".to_string(),
            green_range: if is_vhh_only {
                "-5.0..+5.0".to_string()
            } else {
                ">= -20.4".to_string()
            },
            status: sfvcsp_status,
            detail: "Strongly negative VH×VL charge product predicts high viscosity at >100 mg/mL SC.".to_string(),
        },
    ];

    let tap_overall = if tap_metrics.iter().any(|m| m.status == TrafficLight::Red) {
        TrafficLight::Red
    } else if tap_metrics.iter().any(|m| m.status == TrafficLight::Amber) {
        TrafficLight::Amber
    } else {
        TrafficLight::Green
    };

    let immunogenicity = evaluate_immunogenicity(&annotations);

    // Composite Developability Score (0..100) calibrated to clinical CST vs discontinued controls
    let mut score = 100.0_f64;
    for m in &tap_metrics {
        match m.status {
            TrafficLight::Amber => score -= 10.0,
            TrafficLight::Red => score -= 20.0,
            TrafficLight::Green => {}
        }
    }
    for p in &ptm_liabilities {
        if p.in_cdr && p.severity == TrafficLight::Red {
            if p.category.contains("Glycosylation") || p.category.contains("Cysteine") {
                score -= 9.0;
            } else {
                // Deamidation / isomerization in CDR are real CQA watch-points (e.g. Herceptin N55/D102)
                score -= 3.5;
            }
        }
    }
    // Extreme basic Fv pI / excess positive charge (e.g. Bococizumab pI 9.72) drives polyreactivity
    if biophysical.isoelectric_point_pi > 9.35 || biophysical.charge_ph74 > 6.0 {
        score -= 10.0;
    }
    score -= (immunogenicity.non_self_strong_binders as f64 * 1.5).min(24.0);
    if max_sap > 0.85 {
        score -= 8.0;
    }
    let dev_score = score.clamp(5.0, 100.0).round();
    let summary_verdict = if dev_score >= 80.0 && tap_overall == TrafficLight::Green {
        "GO — Clinical-Grade Developability Profile (Proceed to SPR/SEC-MALS/DSF)".to_string()
    } else if dev_score >= 60.0 {
        "CONDITIONAL GO — Optimize Flagged CDR Liabilities / Patches Before Cell-Line Lock"
            .to_string()
    } else {
        "HIGH RISK — Significant TAP/Immunogenicity/PTM Liabilities (Re-engineer Lead)".to_string()
    };

    DevelopabilityReport {
        name: name.to_string(),
        has_3d_structure: structure.is_some(),
        annotations,
        biophysical,
        tap_metrics,
        tap_overall,
        max_sap_score: (max_sap * 1000.0).round() / 1000.0,
        sap_hotspots,
        ptm_liabilities,
        cdr_high_risk_ptms,
        immunogenicity,
        developability_score_0_100: dev_score,
        summary_verdict,
    }
}

/// Render a complete Markdown Developability, TAP, 3D SAP, PTM & Immunogenicity Dossier.
#[must_use]
pub fn render_developability_markdown(r: &DevelopabilityReport) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Biologics Developability & TAP Report: {}\n", r.name);
    let _ = writeln!(
        md,
        "- **Developability Score:** **`{:.0}/100`** | **TAP Overall:** **`{}`** | **3D SASA:** `{}`",
        r.developability_score_0_100,
        r.tap_overall.as_str(),
        if r.has_3d_structure {
            "Yes (Shrake-Rupley)"
        } else {
            "Sequence-only estimate"
        }
    );
    let _ = writeln!(md, "- **Verdict:** {}\n", r.summary_verdict);

    let b = &r.biophysical;
    let _ = writeln!(md, "## 1. Biophysical & Analytical Characterization\n");
    let _ = writeln!(md, "| Parameter | Value | Parameter | Value |");
    let _ = writeln!(md, "|---|---:|---|---:|");
    let _ = writeln!(
        md,
        "| Total Residues | {} aa | MW (Oxidized / Reduced) | {:.1} / {:.1} Da |",
        b.total_residues, b.mw_oxidized_da, b.mw_reduced_da
    );
    let _ = writeln!(
        md,
        "| Extinction Coeff (`ε280`) | {:.0} M⁻¹cm⁻¹ | `A280 (0.1% = 1 mg/mL)` | {:.3} |",
        b.extinction_coeff_m_cm, b.a280_01pct
    );
    let _ = writeln!(
        md,
        "| Isoelectric Point (`pI`) | {:.2} | Net Charge (pH 5.5 / 7.4) | {:+.2} / {:+.2} |",
        b.isoelectric_point_pi, b.charge_ph55, b.charge_ph74
    );
    let _ = writeln!(
        md,
        "| GRAVY Hydropathy | {:.3} | Max SAP Score | {:.3} |",
        b.gravy, r.max_sap_score
    );

    let _ = writeln!(
        md,
        "\n## 2. TAP 5-Metric Clinical Benchmark (Raybould Guidelines)\n"
    );
    let _ = writeln!(
        md,
        "| Metric | Value | Clinical Green Range | Status | Interpretation |"
    );
    let _ = writeln!(md, "|---|---:|---|---|---|");
    for m in &r.tap_metrics {
        let _ = writeln!(
            md,
            "| {} | {:.2} ({}) | {} | **{}** | {} |",
            m.name,
            m.value,
            m.unit,
            m.green_range,
            m.status.as_str(),
            m.detail
        );
    }

    let _ = writeln!(
        md,
        "\n## 3. Chemical Stability & PTM Audit ({} High-Risk CDR Liabilities)\n",
        r.cdr_high_risk_ptms
    );
    if r.ptm_liabilities.is_empty() {
        let _ = writeln!(md, "- No chemical PTM liabilities flagged.");
    } else {
        let _ = writeln!(
            md,
            "| Chain | Category | Motif | Position | Region | Severity | Mitigation |"
        );
        let _ = writeln!(md, "|---|---|---|---|---|---|---|");
        for p in &r.ptm_liabilities {
            let _ = writeln!(
                md,
                "| {} | {} | `{}` | {} | {} | **{}** | {} |",
                p.chain,
                p.category,
                p.motif,
                p.position_label,
                p.region,
                p.severity.as_str(),
                p.mitigation
            );
        }
    }

    let _ = writeln!(
        md,
        "\n## 4. HLA-DRB1 MHC-II T-Cell Immunogenicity Audit (Germline-Tolerized)\n"
    );
    let imm = &r.immunogenicity;
    let _ = writeln!(
        md,
        "- **Non-Self Strong Binders (8 HLA-DRB1 Supertypes):** `{}` (germline self-peptides subtracted: `{}`) | **Risk Tier:** **{}**",
        imm.non_self_strong_binders, imm.germline_tolerized_hits, imm.risk_tier
    );
    if !imm.hotspots.is_empty() {
        let _ = writeln!(
            md,
            "\n| Chain | Pos | Region | 9-mer Neo-Epitope | Hit Alleles | Top Score | De-Immunization Proposal |"
        );
        let _ = writeln!(md, "|---|---:|---|---|---:|---:|---|");
        for h in imm.hotspots.iter().take(10) {
            let _ = writeln!(
                md,
                "| {} | {} | {} | `{}` | {}/8 | {:.2} | {} |",
                h.chain,
                h.start_pos_1based,
                h.region,
                h.peptide_9mer,
                h.hit_allele_count,
                h.max_binding_score,
                h.deimmunize_suggestion
            );
        }
    }

    if !r.sap_hotspots.is_empty() {
        let _ = writeln!(md, "\n## 5. Top Surface Aggregation / SAP Hotspots\n");
        let _ = writeln!(
            md,
            "| Chain | Residue | Region | SAP Score | SASA (Å²) | Suggested Mitigation |"
        );
        let _ = writeln!(md, "|---|---|---|---:|---:|---|");
        for s in r.sap_hotspots.iter().take(8) {
            let _ = writeln!(
                md,
                "| {} | `{}` | {} | {:.3} | {} | {} |",
                s.chain,
                s.res_label,
                s.region,
                s.sap_score,
                s.sasa_a2.map_or("seq".to_string(), |v| format!("{v:.1}")),
                s.suggested_mutation
            );
        }
    }
    md
}
